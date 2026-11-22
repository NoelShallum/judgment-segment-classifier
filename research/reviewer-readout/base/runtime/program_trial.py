"""Select directly written weight edits on official validation documents only."""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
from encoder import components,features,write_config,fingerprint
from role_fixtures import ANCHORS,CHECKS
from weight_edit import LABELS,ORIGINAL,ROOT,prior_bias,tensor,patched_checkpoint,file_hash

NAMES=("None","Facts","Issue","Arguments of Petitioner","Arguments of Respondent","Reasoning","Decision")
TO_ID={s:i for i,s in enumerate(NAMES)}
CONTRASTS={0:[1,6],1:[0,5,6],2:[3,4,5],3:[4,5],4:[3,5],5:[1,2,6],6:[1,5]}
PRIOR_STRENGTHS=(0,.15,.35,.65)
MARGINS=(0,.5,1.,2.,4.)
SCHEMA="legalseg-current-sentence-fp32-v1"

def save_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+".part")
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n");tmp.replace(path)

def fold(doc):
    return int(hashlib.sha256(("attoflow-role-2026-11/"+doc).encode()).hexdigest()[:8],16)%3

def official_sample(kind,per_class):
    fname="train.csv" if kind=="train" else "val.csv"
    f=ROOT/"references/official"/fname
    if not f.exists():raise FileNotFoundError(f)
    rng=random.Random({"train":601,"select":602,"confirm":603}[kind])
    groups=defaultdict(list); counts=Counter();docs=set()
    with f.open(encoding="utf-8-sig",newline="") as fh:
        for j,row in enumerate(csv.DictReader(fh)):
            doc=row["Index"]
            if kind!="train" and (fold(doc)==0)!=(kind=="confirm"):continue
            y=TO_ID[row["Label"]];counts[y]+=1;docs.add(doc)
            item=(j,doc,row["Text"] or "",y)
            if len(groups[y])<per_class:groups[y].append(item)
            else:
                k=rng.randrange(counts[y])
                if k<per_class:groups[y][k]=item
    rows=sorted((item for grp in groups.values() for item in grp),key=lambda z:z[0])
    weights=np.array([counts[row[3]]/len(groups[row[3]]) for row in rows],dtype=np.float64)
    return rows,weights,{"population":sum(counts.values()),"population_by_role":dict(counts),"eligible_documents":len(docs),"sample_documents":len(set(r[1] for r in rows)),"sample":len(rows)}

def fixture_rows(groups):
    return [(i,key,text,y) for y,key in enumerate(LABELS) for i,text in enumerate(groups[key])]

def metric(y,yp,weights=None):
    y=np.asarray(y,dtype=int); yp=np.asarray(yp,dtype=int)
    cm=np.bincount(y*7+yp,weights=weights,minlength=49).reshape(7,7)
    tp=cm.diagonal();den=cm.sum(0)+cm.sum(1)
    f1=np.divide(2*tp,den,out=np.zeros(7),where=den!=0)
    precision=np.divide(tp,cm.sum(0),out=np.zeros(7),where=cm.sum(0)!=0)
    recall=np.divide(tp,cm.sum(1),out=np.zeros(7),where=cm.sum(1)!=0)
    return {"macro_f1":float(f1.mean()),"accuracy":float(tp.sum()/cm.sum()),"f1":dict(zip(LABELS,f1.tolist())),
            "precision":dict(zip(LABELS,precision.tolist())),"recall":dict(zip(LABELS,recall.tolist())),"confusion":cm.tolist()}

def anchored_edit(x,ys):
    # A fixed rank-one change to each classifier row: the chosen positive
    # centroid gains half a logit and its contrasting centroid loses half.
    # Scaling the head by a prescribed margin edits this difference directly.
    centers=np.stack([x[ys==i].mean(0) for i in range(7)]).astype("float64")
    neg=np.stack([centers[CONTRASTS[i]].mean(0) for i in range(7)])
    direction=centers-neg
    dw=direction/np.square(direction).sum(1,keepdims=True).clip(min=1e-9)
    db=-(dw*(centers+neg)/2).sum(1)
    return dw.astype("float32"),db.astype("float32")

def get_features(m,tok,device,rows,name):
    x=features(m,tok,device,[r[2] for r in rows],SCHEMA+"-"+name)
    return x,np.asarray([r[3] for r in rows],dtype=int)

def paired_document_interval(rows,y,a,b,weights,iterations=1000):
    docs=sorted({r[1] for r in rows});di={s:i for i,s in enumerate(docs)}
    ids=np.array([di[r[1]] for r in rows]);y=np.asarray(y);a=np.asarray(a);b=np.asarray(b)
    def by_doc(pred):
        return np.bincount(ids*49+y*7+pred,weights=weights,minlength=len(docs)*49).reshape(len(docs),7,7)
    ca,cb=by_doc(a),by_doc(b)
    def score(cm):
        den=cm.sum(axis=1)+cm.sum(axis=0);return np.divide(2*cm.diagonal(),den,out=np.zeros(7),where=den!=0).mean()
    rng=np.random.default_rng(98443);diff=[]
    for _ in range(iterations):
        ix=rng.integers(len(docs),size=len(docs));diff.append(score(cb[ix].sum(0))-score(ca[ix].sum(0)))
    return np.quantile(diff,[.025,.975]).tolist()

def run(args):
    results=ROOT/"results";results.mkdir(exist_ok=True);progress=results/"program-validation.json"
    model,tok,device=components()
    w=tensor(ORIGINAL,"classifier.weight"); bias=tensor(ORIGINAL,"classifier.bias")
    print("original checkpoint loaded on",device,flush=True)
    xa,ya=get_features(model,tok,device,fixture_rows(ANCHORS),"authored-anchors")
    xc,yc=get_features(model,tok,device,fixture_rows(CHECKS),"authored-checks")
    dw,db=anchored_edit(xa,ya)
    rows,weights,stats=official_sample("select",args.per_class)
    x,y=get_features(model,tok,device,rows,"select-"+str(args.per_class))
    reports=[];best=(-1,None)
    for p in PRIOR_STRENGTHS:
        for gain in MARGINS:
            ww=w+gain*dw;bb=bias+prior_bias(p)+gain*db
            pred=(x@ww.T+bb).argmax(1);check=(xc@ww.T+bb).argmax(1)
            entry={"prior":p,"margin":gain,"official_validation_sample":metric(y,pred,weights),"synthetic_development_check":metric(yc,check)}
            reports.append(entry)
            # Selection uses only the prevalence-weighted official development sample.
            value=entry["official_validation_sample"]["macro_f1"]
            if value>best[0]+1e-12:best=(value,(p,gain))
    centers=np.stack([xa[ya==i].mean(0) for i in range(7)])
    retr=(x@((centers/np.linalg.norm(centers,axis=1,keepdims=True)).T)).argmax(1)
    p,gain=best[1];ww=w+gain*dw;bb=bias+prior_bias(p)+gain*db
    report={"sampling":"Seeded per-role reservoir over two-thirds of validation judgments, weighted to their true class prevalences. One-third of validation judgments reserved for confirmation.",
            "stats":stats,"candidates":reports,"anchor_centroid_baseline":metric(y,retr,weights),"selected":{"prior":p,"margin":gain},
            "selected_source":"Highest official validation-sample macro-F1. Ties favor the simpler candidate listed first.","official_test_used":False}
    save_json(progress,report)
    target=ROOT/"checkpoints/selected-direct";target.mkdir(exist_ok=True,parents=True)
    changes=patched_checkpoint(ORIGINAL,target/"model.safetensors",{"classifier.weight":ww,"classifier.bias":bb})
    write_config(target)
    info={"method":"Direct float32 output-head edit","selected":report["selected"],"source_sha256":file_hash(ORIGINAL),"target_sha256":file_hash(target/"model.safetensors"),"changes":changes,"labels":LABELS,"anchors_source":"role_fixtures.py"}
    save_json(target/"edit.json",info)
    # Apply exactly the frozen selected model and the original to disjoint dev docs.
    crows,cweights,cstats=official_sample("confirm",args.per_class)
    assert not ({r[1] for r in rows}&{r[1] for r in crows})
    cx,cy=get_features(model,tok,device,crows,"confirm-"+str(args.per_class))
    pa=(cx@w.T+bias).argmax(1);pb=(cx@ww.T+bb).argmax(1)
    report["confirmation"]={"stats":cstats,"base":metric(cy,pa,cweights),"selected":metric(cy,pb,cweights),"paired_document_95pct_interval_macro_f1_delta":paired_document_interval(crows,cy,pa,pb,cweights),"changed_predictions":int((pa!=pb).sum())}
    save_json(progress,report)
    print(json.dumps({"selected":report["selected"],"selection_baseline":reports[0]["official_validation_sample"]["macro_f1"],"selection_edited":best[0],"confirmation_baseline":report["confirmation"]["base"]["macro_f1"],"confirmation_edited":report["confirmation"]["selected"]["macro_f1"],"delta_interval":report["confirmation"]["paired_document_95pct_interval_macro_f1_delta"]}),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--per-class",type=int,default=160)
    run(parser.parse_args())
