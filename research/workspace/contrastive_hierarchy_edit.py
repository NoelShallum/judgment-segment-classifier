"""Write edits to the existing hierarchy's seven output rows using matched legal role cues."""
from collections import defaultdict
import itertools
import numpy as np
from scipy.linalg import solve
from safetensors.numpy import load_file,save_file
import torch

from associative_edit import correction
from fragment_trial import passages
from hier_model import CRF,EMITTER,HierModel,path_labels
from hier_trial import DONOR,OUT
from program_trial import save_json
from seven_paths import ROOT
from span_eval import counts,sets,summarize
from weight_edit import file_hash

TOPICS=(
    "the notice was sent before the appeal period ended",
    "the payment was transferred after the contract expired",
    "the tenant handed over the keys before the inspection",
    "the application was filed before the tribunal's deadline",
    "the authority supplied the records after the hearing",
    "the contractor completed the repairs before the inspection",
    "the clerk recorded the objection before the final hearing",
    "the employer withdrew the offer after the interview",
    "the commission closed the public consultation before the vote",
    "the seller returned the deposit after the cancellation",
    "the beneficiary received the order before the review period ended",
    "the agency notified the company after the assessment",
)
FRAMES=(
    ("Appeal concerning whether {topic}. List before the Division Bench.", "Court record concerning whether {topic}. Appearance entered."),
    ("The record states that {topic}.", "The agreed chronology shows that {topic}."),
    ("The question before us is whether {topic}.", "We must determine whether {topic}."),
    ("The appellant contends that {topic}.", "Learned counsel for the petitioner submits that {topic}."),
    ("The respondent contends that {topic}.", "Learned counsel for the opposing party submits that {topic}."),
    ("We conclude from the evidence that {topic}, which answers the objection.", "The reasoning is straightforward. The documents establish that {topic}."),
    ("We accordingly allow the appeal on the ground that {topic}.", "The court orders the application dismissed on the finding that {topic}."),
)


def matched_keys(model):
    groups=[]
    for topic, style in itertools.product(TOPICS, (0,1)):
        # Use a neutral frame at the edges. The competing roles share the same factual proposition.
        lead=["The proceedings were called for hearing.", "Both parties have had an opportunity to address the record."]
        rest=["We have considered the material placed before us."]
        for label, choices in enumerate(FRAMES):
            phrase=choices[style].format(topic=topic)
            sequence=lead+[phrase]+rest
            h=model.contextualize(model.words(sequence))[len(lead)]
            groups.append((topic,style,label,h))
    keys=np.stack([r[3] for r in groups])
    labels=np.array([r[2] for r in groups])
    return keys,labels


def paired_delta(keys, labels, weight, bias, damping, strength):
    # Orthogonalize the correction against the shared topic and frame averages.
    keys=keys.astype(np.float64)
    grouped=keys.reshape(-1,7,keys.shape[-1])
    x=(grouped-grouped.mean(1,keepdims=True)).reshape(keys.shape)
    vector=x
    gram=vector.T@vector
    reg=np.eye(len(gram))*damping*(gram.trace()/len(gram))
    desired=np.eye(7)[labels]
    logits=x@weight.T
    beta=solve(gram+reg,vector.T@(desired-logits),assume_a="pos").T
    return (weight+strength*beta).astype("f4"),bias


def main():
    torch.set_num_threads(2)
    model=HierModel(DONOR)
    keys,y=matched_keys(model)
    w=model.weight[3:];b=model.bias[3:]
    candidates=[("original",w,b)]
    for damping,gain in itertools.product((.1,.5,1.),(.25,.5,1.)):
        x,z=paired_delta(keys,y,w,b,damping,gain)
        candidates.append((f"same-topic-{damping}-{gain}",x,z))
    for margin,damp in itertools.product((.5,1.5,3.),(.1,.5)):
        x,z=correction(keys,y,w,b,margin,damp)
        for gain in (.25,.5,1.):
            candidates.append((f"margin-{margin}-{damp}-{gain}",w+gain*(x-w),b+gain*(z-b)))
    report={}
    for task,groups in (("whole",sets()),("fragment",passages())):
        f=load_file(str(OUT/(("document" if task=="whole" else "fragment")+".safetensors")))
        tr=f[CRF] if task=="whole" else f["transition"]
        if task=="fragment" and not np.isclose(f["classifier.weight"][:,200:],0).all():
            raise ValueError("Incumbent fragment requires additional features.")
        cache={}
        for name,docs in groups.items():
            cache[name]=[model.contextualize(model.words([d["text"][a:b] for a,b in d["units"]])) for d in docs]
        def evaluate(fold,ww,bb):
            out=[]
            for d,x in zip(groups[fold],cache[fold]):
                out.append(counts(d,d["units"],path_labels(x@ww.T+bb,tr)))
            return np.stack(out)
        grid=[]
        best=None
        original=summarize(evaluate("select",w,b).sum(0))
        for method,ww,bb in candidates:
            q=summarize(evaluate("select",ww,bb).sum(0))
            rank=q["exact_labeled_spans"]["f1"],q["token_accuracy"]
            grid.append([method,*rank])
            if best is None or rank>best[0]:
                best=(rank,method,ww,bb)
        _,method,ww,bb=best
        mx=model.weight.copy();mb=model.bias.copy();mx[3:]=ww;mb[3:]=bb
        path=OUT/(f"contrastive-{task}.safetensors")
        save_file({CRF:tr,EMITTER+"weight":mx,EMITTER+"bias":mb},str(path))
        records={}
        for fold in groups:
            c=evaluate(fold,ww,bb);p=evaluate(fold,w,b)
            rng=np.random.default_rng(12102);differences=[]
            for ix in rng.integers(len(c),size=(1200,len(c))):
                a=summarize(c[ix].sum(0));z=summarize(p[ix].sum(0))
                differences.append(a["exact_labeled_spans"]["f1"]-z["exact_labeled_spans"]["f1"])
            records[fold]={"candidate":summarize(c.sum(0)),"baseline":summarize(p.sum(0)),
                           "span_gap_95pct":np.quantile(differences,[.025,.975]).tolist(),
                           "cases":len(c)}
        report[task]={"selected":method,"grid":grid,"scores":records,"patch_sha256":file_hash(path)}
        print(task,method,records["test"]["candidate"]["exact_labeled_spans"]["f1"],records["test"]["baseline"]["exact_labeled_spans"]["f1"],flush=True)
        save_json(ROOT/"results"/"contrastive-hierarchy-edit.json",{"keys":len(keys),"topics":len(TOPICS),
            "protocol":"The seven role frames share their underlying proposition so role cues vary independently of the topic. Search two analytic weight writes and the fixed parent with no optimizer. Only existing seven classifier rows and biases are eligible for updates. Exact span F1 selects on validation.",
            "tasks":report})

if __name__=="__main__":
    main()
