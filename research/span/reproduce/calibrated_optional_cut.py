"""A closed-form boundary readout for rare exact character cuts."""
import hashlib
from itertools import product
import json
import random

import numpy as np
from safetensors.numpy import load_file,save_file

from optional_boundary_study import (EDGE,PATCH,RESULT,CRF,DONOR,ROOT,HierModel,
    arrays,bootstrap,document,file_hash,full_score,gold_cases,independent_validation,
    metric,package,sample_whole,save_json,torch,training_cases)
from hier_full import CACHE

CHOICES=[None,*product((-3.1,-2.2,-1.3),(.45,.8),(.4,1.1))]


def observations(compiled):
    scorer,em,_,options,coordinates,_,_=compiled
    top=lambda arr:arr.argmax()
    for block,variants in enumerate(options):
        here=em[block]
        left=em[max(0,block-1)]
        right=em[min(len(em)-1,block+1)]
        lo,hi=coordinates[block]
        length=scorer.offsets[hi]-scorer.offsets[lo]
        for j,(prior,share,diff,pair,index) in enumerate(variants):
            displaced=scorer.offsets[index]-scorer.offsets[lo]
            row=np.r_[prior,share,np.log1p(displaced),np.log1p(length-displaced),
                np.linalg.norm(diff), top(left)!=top(here),top(here)!=top(right),
                diff,here-here.mean(), left-left.mean(),right-right.mean(),
                pair.max(),pair[top(left),top(right)],pair[top(here),top(right)]]
            yield block,j,row,scorer.gold_label[index]>=0


def fit(rows,inside,ridge):
    x=np.asarray(rows,dtype=np.float64)
    y=np.asarray(inside,dtype=bool)
    mean=x.mean(0)
    scale=np.maximum(x.std(0),.03)
    x=(x-mean)/scale
    a=x[y];b=x[~y]
    high=a.mean(0);low=b.mean(0)
    da=a-high;db=b-low
    covariance=(da.T@da/len(a)+db.T@db/len(b))*.5
    w=np.linalg.solve(covariance+ridge*np.eye(len(mean)), high-low)
    prior=np.log(len(a)/len(b))
    bias=prior-.5*(high+low)@w
    return {"boundary.weight":(w/scale).astype(np.float32),
            "boundary.bias":np.asarray(bias-mean@(w/scale),np.float32)}


def predictions(bank,program):
    return [np.array([float(np.clip(u@program["boundary.weight"]+program["boundary.bias"],-10,3))
                      for _,_,u,_ in observations(doc)]) for doc in bank]


def recut(bank,forecast,floor):
    ready=[]
    for comp,prob in zip(bank,forecast):
        scorer,emit,edge,options,coords,count,hit=comp
        p=iter(prob)
        shifts=[]
        for v in options:
            block=[]
            for old in v:
                score=next(p)
                if score>=floor:
                    block.append((score,*old[1:]))
            shifts.append(block)
        ready.append((scorer,emit,edge,shifts,coords,count,hit))
    return ready


def main():
    torch.set_num_threads(2)
    parent=load_file(str(EDGE/"role-edge-exact.safetensors"))
    prior=load_file(str(PATCH/"prior.safetensors"))
    lexical=json.loads((EDGE/"vocab.json").read_text())
    vocab=json.loads((PATCH/"vocab.json").read_text())
    groups=sample_whole()
    model=HierModel(DONOR)

    def encode(d,split):
        if split=="test":
            with np.load(CACHE/(d["id"]+".npz")) as fp:raw=fp["raw"]
        elif split=="independent":
            sig=hashlib.sha256(d["text"].encode()).hexdigest()
            raw=np.load(ROOT/"astra-run-penalty/validation-cache"/(sig+".npy"))
        else:
            raw=model.logits(model.contextualize(model.words([d["text"][a:b] for a,b in d["units"]])))
        return package(d,model,parent,raw,vocab,prior,lexical)

    existing=PATCH/"calibrated-prior.safetensors"
    train_receipt=RESULT/"calibrated-boundary-train.json"
    if existing.exists() and train_receipt.exists():
        model_prior=load_file(str(existing))
    else:
        ids={str(k) for k in random.Random(42713).sample(range(4984),220)}
        feats=[]
        labels=[]
        kept=[]
        for k,rr in training_cases(ROOT/"references/official/train.csv"):
            if k not in ids:
                continue
            for _,_,x,answer in observations(encode(document(k,rr),"train")):
                feats.append(x)
                labels.append(answer)
            kept.append(k)
            if len(kept)%50==0:print("Encoded calibrated training",len(kept),sum(labels),flush=True)
        model_prior=fit(feats,labels,.65)
        save_file(model_prior,str(existing))
        save_json(train_receipt,{"cases":kept,"proposals":len(labels),"positives":int(sum(labels)),
            "model_sha256":file_hash(existing),"ridge":.65})
        print("Frozen closed-form prior",len(kept),len(labels),sum(labels),flush=True)

    def prepare(split):
        return [encode(d,split) for d in groups[split]]

    def eval_cfg(bank,post,option):
        if option is None:
            return arrays(bank,parent,None)
        floor,alpha,cost=option
        return arrays(recut(bank,post,floor),parent,(alpha,.3,cost))

    train=prepare("select")
    predicted=predictions(train,model_prior)
    baseline=best=metric(arrays(train,parent,None)[0].sum(0))
    winner=None
    grid=[]
    for spec in CHOICES:
        val,n,good=eval_cfg(train,predicted,spec)
        key=metric(val.sum(0))
        grid.append([spec,*key,n,good])
        if key>best:
            best=key
            winner=spec
    frozen=RESULT/"calibrated-boundary-frozen.json"
    save_json(frozen,{"winner":winner,"grid":grid,"prior_sha256":file_hash(existing),
                      "selection_ids":[d["id"] for d in groups["select"]]})
    print("Calibrated choice",winner,best,"baseline",baseline,flush=True)
    if winner:
        save_file({**model_prior,"boundary.rule":np.asarray(winner,np.float32)},str(PATCH/"calibrated-selected.safetensors"))
    seen={d["id"] for dd in groups.values() for d in dd}
    omitted=set(json.loads((EDGE/"overlap.json").read_text())["independent_exclusions_for_text_disjoint_check"])
    if winner:
        groups["independent"]=[d for d in independent_validation(seen) if d["id"] not in omitted]
    report={"winner":winner,"freeze_sha256":file_hash(frozen),"splits":{},
            "method":"A regularized closed-form discriminant reweights lexical and frozen donor evidence, including neighboring scores, to separate genuine missing role boundaries from false proposals. Selection can disable optional cuts. Gradients are unused."}
    for split in ("select","confirm","independent","test") if winner else ("select",):
        if split=="test":
            groups[split]=[document(k,r) for k,r in gold_cases().items()]
        data=train if split=="select" else prepare(split)
        post=predicted if split=="select" else predictions(data,model_prior)
        q,n,g=eval_cfg(data,post,winner)
        z,_,_=arrays(data,parent,None)
        report["splits"][split]={"cases":len(data),"candidate":full_score(q),"parent":full_score(z),
                                 "opened":n,"opened_at_gold_change":g,"gap_ci95":bootstrap(q,z)}
        save_json(RESULT/"calibrated-boundary-lattice.json",report)
        print(split,metric(q.sum(0)),metric(z.sum(0)),n,g,flush=True)


if __name__=="__main__":
    main()
