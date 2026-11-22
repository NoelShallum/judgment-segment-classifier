"""Write discourse and document-position residual weights onto role emissions."""
from itertools import product
import numpy as np
from safetensors.numpy import load_file,save_file
import torch

from fragment_trial import passages
from hier_full import CACHE,gold_cases
from hier_model import CRF,HierModel,path_labels
from hier_trial import DONOR,OUT
from program_trial import save_json
from role_cues import FEATURES,cue_features,programmed_rows
from seven_paths import ROOT
from span_eval import counts,document,summarize
from transition_span_edit import features,sample_whole
from weight_edit import file_hash

AUTHOR=programmed_rows()
POSITION=np.array([[.85,.45,.50,0,0,-.3,-.6],
                   [-.1,.3,.35,.15,.15,-.05,-.5],
                   [-.4,-.15,-.15,.4,.4,.35,-.3],
                   [-.15,-.2,-.15,.02,.02,.55,.25],
                   [-.15,-.55,-.3,-.5,-.5,.2,1.2]],np.float32)
POSITION-=POSITION.mean(1,keepdims=True)

def text_features(d):
    texts=[d["text"][a:b] for a,b in d["units"]]
    q=np.array([(a+b)*.5/len(d["text"]) for a,b in d["units"]])
    loc=np.eye(5)[np.digitize(q,[.05,.25,.60,.91])]
    return cue_features(texts),loc

def main():
    torch.set_num_threads(2)
    model=HierModel(DONOR)
    final={}
    for task,groups in (("whole",sample_whole()),("fragment",passages())):
        if task=="whole":groups["test"]=[document(k,rows) for k,rows in gold_cases().items()]
        base_art=load_file(str(OUT/(("document" if task=="whole" else "fragment")+".safetensors")))
        incumbent=base_art[CRF] if task=="whole" else base_art["transition"]
        if task=="fragment" and not np.isclose(base_art["classifier.weight"][:,200:],0).all():
            raise ValueError("Fixture decoder requires standalone feature inference.")
        variants={"incumbent":incumbent,
                  "role_specific":load_file(str(OUT/("role-specific-"+task+".safetensors")))[CRF]}
        if task=="whole":variants["balanced"]=load_file(str(OUT/"role-specific-balanced.safetensors"))[CRF]
        data={}
        for split,docs in groups.items():
            ux=[text_features(d) for d in docs]
            if task=="whole" and split=="test":
                emb=[]
                for d in docs:
                    with np.load(CACHE/(d["id"]+".npz")) as f:emb.append(f["raw"])
            else:
                emb=features(model,docs)
            data[split]=[(d,prior,h@AUTHOR,p@POSITION) for d,prior,(h,p) in zip(docs,emb,ux)]
        def collect(split,tr,cue,edge):
            return np.stack([counts(d,d["units"],path_labels(a+cue*b+edge*c,tr)) for d,a,b,c in data[split]])
        best=None;balanced=None;grid=[]
        control=summarize(collect("select",incumbent,0,0).sum(0))
        for (name,trans),cue,edge in product(variants.items(),(0.,.5,1.,2.),((0.,.5,1.25) if task=="whole" else (0.,))):
            score=summarize(collect("select",trans,cue,edge).sum(0))
            key=score["exact_labeled_spans"]["f1"],score["token_accuracy"]
            grid.append([name,cue,edge,*key]);x=(key,name,cue,edge,trans)
            if best is None or key>best[0]:best=x
            if score["token_accuracy"]>=control["token_accuracy"]-.01 and (balanced is None or key>balanced[0]):balanced=x
        profiles={}
        for profile,seq in (("exact",best),("balanced",balanced)):
            _,name,cue,edge,trans=seq
            path=OUT/(f"{task}-discourse-{profile}.safetensors")
            save_file({CRF:trans,"cue_head":(cue*AUTHOR).astype("f4"),
                      "position_head":(edge*POSITION).astype("f4")},str(path))
            records={}
            for split in data:
                a=collect(split,trans,cue,edge);b=collect(split,incumbent,0,0)
                deltas=[];rng=np.random.default_rng(10854)
                for ix in rng.integers(len(a),size=(1400,len(a))):
                    x=summarize(a[ix].sum(0));y=summarize(b[ix].sum(0))
                    deltas.append([x["exact_labeled_spans"]["f1"]-y["exact_labeled_spans"]["f1"],x["token_accuracy"]-y["token_accuracy"]])
                records[split]={"candidate":summarize(a.sum(0)),"incumbent":summarize(b.sum(0)),"cases":len(a),
                                "span_f1_token_accuracy_gap_ci95":np.quantile(deltas,[.025,.975],axis=0).T.tolist()}
            profiles[profile]={"transition":name,"cue_gain":cue,"position_gain":edge,"weights_sha256":file_hash(path),"scores":records}
            q=records["test"]["candidate"]
            print(task,profile,name,cue,edge,q["exact_labeled_spans"]["f1"],q["token_accuracy"],flush=True)
        final[task]={"grid":grid,"profiles":profiles}
        save_json(ROOT/"results/discourse-span-weights.json",{"scope":"Existing authored phrase cues and published model with direct residual weights. The optional position term uses raw text length and is eligible only on complete judgments. No gradients. Tuned on separate validation cases; full benchmark cases have prior aggregate comparisons.",
            "cue_names":[name for name,_,_ in FEATURES],"tasks":final})

if __name__=="__main__":
    main()
