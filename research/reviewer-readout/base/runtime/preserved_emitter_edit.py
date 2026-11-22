"""Closed-form edits to a pretrained role emitter with an explicit retention objective."""
import csv
from collections import defaultdict
import itertools
import random
import numpy as np
from scipy.linalg import solve
from safetensors.numpy import load_file,save_file
import torch

from fragment_trial import passages
from hier_model import CRF,EMITTER,HierModel,path_labels
from hier_trial import DONOR,OUT
from program_trial import TO_ID,save_json
from seven_paths import ROOT
from span_eval import counts,document,sets,summarize
from weight_edit import file_hash

CACHE=ROOT/"cache/hier-emitter-edits-v1"
SOURCE=ROOT/"references/official"


def edit_documents():
    """Choose 144 reproducible, evenly distributed training case IDs."""
    ids=set(random.Random(176529).sample(range(4984),144))
    rows=defaultdict(list)
    with (SOURCE/"train.csv").open(newline="",encoding="utf-8-sig") as f:
        for i,r in enumerate(csv.DictReader(f)):
            if int(r["Index"]) in ids and r["Text"].strip():
                rows[r["Index"]].append((i,r["Text"],TO_ID[r["Label"]]))
    return [document(k,rows[k]) for k in sorted(rows)]



def embeddings(model, docs, kind):
    CACHE.mkdir(parents=True,exist_ok=True)
    values=[]
    for d in docs:
        p=CACHE/(kind+"-"+str(d["id"])+".npy")
        if not p.exists():
            cut=d["reference_units"] if kind=="aligned-source-train" else d["units"]
            result=model.contextualize(model.words([d["text"][l:r] for l,r in cut]))
            a=p.with_suffix(".part")
            with a.open("wb") as f:np.save(f,result,allow_pickle=False)
            a.replace(p)
        values.append(np.load(p,allow_pickle=False))
    return values


def patch_candidates(model,x,y):
    base=model.weight[3:].T.astype(np.float64);bias=model.bias[3:].astype(np.float64)
    base=np.vstack((base,bias))
    width=x.shape[1]
    x=np.column_stack((x,np.ones(len(x))))
    a=x@base
    pred=a.argmax(1)
    match=pred==y
    masked=a.copy()
    masked[np.arange(len(a)),y]=-np.inf
    other=masked.argmax(1)
    delta_target=np.zeros_like(a)
    needed=np.maximum(0.,1.+a[np.arange(len(a)),other]-a[np.arange(len(a)),y])
    delta_target[np.arange(len(a)),y]=needed/2
    delta_target[np.arange(len(a)),other]=-needed/2
    # Only the selected wrong predictions receive an association write. Correct outputs anchor retention.
    ew=x[~match];target=delta_target[~match];ky=y[~match]
    prior=np.bincount(ky,minlength=7)/len(ky)
    keep=x[match];bkeep=keep.T@keep/len(keep)
    I=np.eye(width+1)
    I[-1,-1]*=.25
    yield "parent",base[:-1].T,base[-1]
    for balance in (0.,.5):
        weighted=np.sqrt(np.maximum(prior[ky],1e-8)**-balance)
        keys=ew*weighted[:,None]
        targets=target*weighted[:,None]
        gram=keys.T@keys/len(keys);rhs=keys.T@targets/len(keys)
        for preserve,reg in itertools.product((.25,1.,4.),(.025,.10)):
            step=solve(gram+preserve*bkeep+reg*np.mean(np.diag(gram))*I,rhs,assume_a="pos")
            for scale in (.25,.5,1.):
                values=base+scale*step
                yield [balance,preserve,reg,scale],values[:-1].T,values[-1]


def main():
    torch.set_num_threads(2)
    model=HierModel(DONOR)
    train=edit_documents()
    if len(train)!=144:
        raise ValueError(f"Unexpected train case IDs: loaded {len(train)}")
    x=np.concatenate(embeddings(model,train,"aligned-source-train"))
    y=np.concatenate([d["reference_roles"] for d in train])
    heads=list(patch_candidates(model,x,y))
    print("Programmed",len(heads)-1,"analytic candidates from",len(y),"rows",flush=True)
    groups={"whole":sets(),"fragment":passages()}
    result={}
    for task,sets_by_split in groups.items():
        path=OUT/(("document" if task=="whole" else "fragment")+".safetensors")
        n=load_file(str(path));tr=n[CRF] if task=="whole" else n["transition"]
        if task=="fragment" and not np.isclose(n["classifier.weight"][:,200:],0).all():
            raise ValueError("Incumbent fragment uses independent features")
        bank={k:embeddings(model,ds,task+"-"+k) for k,ds in sets_by_split.items()}
        def score(split,w,b,t):
            records=[]
            for d,xx in zip(sets_by_split[split],bank[split]):
                q=path_labels(xx@w.T+b,t)
                records.append(counts(d,d["units"],q))
            return summarize(np.sum(records,axis=0))
        all_scores=[];best=None
        for spec,w,b in heads:
            for factor in (1.,.7):
                choice=tr.copy();choice[3:,3:]*=factor
                r=score("select",w,b,choice)
                key=r["exact_labeled_spans"]["f1"],r["token_accuracy"]
                all_scores.append([spec,factor,*key])
                if best is None or key>best[0]:
                    best=(key,spec,w,b,choice,factor)
        _,spec,w,b,t,factor=best
        a=model.weight.copy();z=model.bias.copy();a[3:]=w;z[3:]=b
        patch=OUT/("preserved-emitter-"+task+".safetensors")
        save_file({CRF:t,EMITTER+"weight":a,EMITTER+"bias":z},str(patch))
        by_fold={k:{"candidate":score(k,w,b,t),"baseline":score(k,model.weight[3:],model.bias[3:],tr),
                    "cases":len(ds)} for k,ds in sets_by_split.items()}
        result[task]={"profile":{"analytic_edit":spec,"transition_scale":factor},"grid":all_scores,"scores":by_fold,"sha256":file_hash(patch)}
        save_json(ROOT/"results"/"preserved-emitter-edit.json",{"train_rows":len(y),"train_cases":len(train),"program":"Closed-form supervised residual write to a pretrained head. Correct training activations penalize output drift. No optimizer, encoder update, validation labels in keys, or gradient steps.","tasks":result})
        print(task,result[task]["profile"],by_fold["test"]["candidate"]["exact_labeled_spans"]["f1"],flush=True)
if __name__=="__main__":
    main()
