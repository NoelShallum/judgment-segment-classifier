"""Compare cut proposal ranking on confirmation judgments kept out of the readout fit."""
import json

import numpy as np
from safetensors.numpy import load_file

from calibrated_optional_cut import (
    EDGE,PATCH,RESULT,DONOR,HierModel,package,observations,predictions,sample_whole,save_json,torch)
from preserved_emitter_edit import embeddings


def rank_summary(values,truth):
    order=np.argsort(-values,kind="stable")
    a=truth[order]
    end=np.r_[np.flatnonzero(np.diff(values[order]))+1,len(values)]
    found=np.cumsum(a)[end-1]
    tp=np.diff(np.r_[0,found])
    active={}
    for share in (.01,.05,.1):
        take=max(1,int(len(a)*share))
        threshold=values[order[take-1]]
        keep=values>=threshold
        active[str(share)]={"retained":int(keep.sum()),"precision":float(truth[keep].mean()),
                             "recall":float(truth[keep].sum()/truth.sum())}
    return {"average_precision":float(np.sum(tp*found/end)/truth.sum()),"top_fraction":active}


def main():
    torch.set_num_threads(2)
    p=load_file(str(EDGE/"role-edge-exact.safetensors"))
    prior=load_file(str(PATCH/"prior.safetensors"))
    head=load_file(str(PATCH/"calibrated-prior.safetensors"))
    vocab=json.loads((PATCH/"vocab.json").read_text())
    lexical=json.loads((EDGE/"vocab.json").read_text())
    model=HierModel(DONOR)
    cohort=sample_whole()["confirm"]
    bank=[]
    cached=embeddings(model,cohort,"raw-boundary-write-confirm")
    for i,(d,state) in enumerate(zip(cohort,cached),1):
        q=model.logits(state)
        bank.append(package(d,model,p,q,vocab,prior,lexical))
        if i%24==0:
            print("Confirmation score features",i,flush=True)
    predicted=np.concatenate(predictions(bank,head))
    annotated=[(float(x[0]),label) for entry in bank for _,_,x,label in observations(entry)]
    a,b=map(np.asarray,zip(*annotated))
    result={"cases":len(cohort),"options":len(a),"positives":int(b.sum()),
            "lexical_only":rank_summary(a,b),"closed_form":rank_summary(predicted,b)}
    save_json(RESULT/"calibrated-boundary-ranking.json",result)
    print(result,flush=True)


if __name__=="__main__":
    main()
