"""Substitute only the donor word table and run all downstream steps in the shipped pairwise sequence path."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from native_memory import HERE,ROOT,HierModel,DONOR,confusion_docs,metric,dump,file_hash,BASE,data,confusion
from embedding_write import plan,digest
from hier_model import LABELS
import predict_pairwise_judgment as pair
import sentence_phrase_sequence as seq

def run(split,limit):
    marker=HERE/f"production-word-table-{split}-{limit}.json"
    d=data(split,limit)
    model,*components=seq.phrase.load_predictor("target phrases")
    raw=model.word.clone()
    # Cache the unaltered production dependencies, but exercise the released entry point for every judgment.
    seq.phrase.load_predictor=lambda arm:(model,*components)
    hook=pair.predictor("direct")
    original_threads=torch.set_num_threads
    torch.set_num_threads=lambda _:original_threads(1)
    y=np.concatenate([c[2] for c in d]);ends=np.r_[0,np.cumsum([len(c[2]) for c in d])]
    keys=[c[0] for c in d]
    countmap={r:i for i,r in enumerate(LABELS)}
    out=dict(splits=split,rows=len(y),keys=keys,weights_sha256=file_hash(BASE),tensor="sent_encoder.emb.weight",gradient_steps=0,arms={})
    controls=None
    for mode,scale in (("base",0),("probe-res",.28),("probe-res-shuf",.28)):
        model.word=raw
        if scale:
            idx,delta=plan(model,mode)
            model.word=raw.clone();model.word[idx]+=torch.from_numpy(delta)*scale
        predictions=[];start=time.monotonic()
        for k,txt,gt,_ in d:
            result=seq.run_input(txt,False,"spans",hook)
            predictions.extend(countmap[label] for label in result["labels"])
        pred=np.asarray(predictions)
        if controls is None: controls=pred.copy()
        matrix=confusion_docs(y,pred,ends);orig=confusion_docs(y,controls,ends)
        rng=np.random.default_rng(713)
        ci=[]
        for j in rng.integers(len(keys),size=(1200,len(keys))):
            a,b=metric(matrix[j].sum(0)),metric(orig[j].sum(0))
            ci.append([a["macro_f1"]-b["macro_f1"],a["accuracy"]-b["accuracy"]])
        out["arms"][mode]=dict(**metric(matrix.sum(0)),weight_table_sha256=digest(model.word.numpy()),seconds=time.monotonic()-start,changed=int((pred!=controls).sum()),
                          corrected=int(((pred!=controls)&(pred==y)).sum()),harmed=int(((pred!=controls)&(controls==y)).sum()),
                          paired95=np.quantile(ci,[.025,.975],axis=0).T.tolist())
        dump(marker,out)
        np.savez(HERE/f"production-word-table-{split}-{limit}-{mode}.npz",keys=keys,ends=ends,gold=y,pred=pred)
        print(split,limit,mode,out["arms"][mode]["macro_f1"],out["arms"][mode]["accuracy"],out["arms"][mode]["changed"],round(time.monotonic()-start),flush=True)
    if split in ("select","confirm") and (HERE/("strong-"+split+".npz")).exists():
        f=np.load(HERE/("strong-"+split+".npz"));ix={k:(a,b) for k,a,b in zip(f["keys"],f["ends"],f["ends"][1:])}
        rows=np.concatenate([np.arange(*ix[k]) for k in keys]);np.testing.assert_array_equal(y,f["gold"][rows])
        out["incumbent_20pct"]=metric(confusion(y,f["pred"][rows]));dump(marker,out)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("split",choices=("select","confirm","shared"));p.add_argument("limit",type=int)
    a=p.parse_args();run(a.split,a.limit)
