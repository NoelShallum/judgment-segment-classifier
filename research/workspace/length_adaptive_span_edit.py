"""Test a bounded-context emission write only on long judgments."""
from collections import defaultdict
import csv
import hashlib
import itertools
import json
import random

import numpy as np
from safetensors.numpy import load_file,save_file
import torch

from discourse_span_edit import text_features
from context_span_runtime import contextual
from hier_full import CACHE as BASE_CACHE,gold_cases
from hier_model import CRF,HierModel,path_labels
from hier_trial import DONOR,OUT
from program_trial import TO_ID,fold,save_json
from seven_paths import ROOT
from span_eval import counts,document,summarize
from weight_edit import file_hash

MIN_WORDS=8000
WIDTHS=(32,64,128)
CHOSEN=ROOT/"cache/length-adaptive-span-v1"

def validation_cohort():
    eligible=defaultdict(list)
    totals=defaultdict(int)
    with (ROOT/"references/official/val.csv").open(newline="",encoding="utf-8-sig") as f:
        for i,r in enumerate(csv.DictReader(f)):
            totals[r["Index"]]+=len(r["Text"].split())
    for k,n in totals.items():
        if n>=MIN_WORDS:eligible["confirm" if fold(k)==0 else "select"].append(k)
    selected={n:random.Random(s).sample(sorted(eligible[n]),min(36,len(eligible[n])))
              for n,s in (("select",905708),("confirm",905709))}
    ids=set(selected["select"]+selected["confirm"])
    docs=defaultdict(list)
    with (ROOT/"references/official/val.csv").open(newline="",encoding="utf-8-sig") as f:
        for i,r in enumerate(csv.DictReader(f)):
            if r["Index"] in ids:docs[r["Index"]].append((i,r["Text"],TO_ID[r["Label"]]))
    return {k:[document(i,docs[i]) for i in v] for k,v in selected.items()}

def compute(m,d,widths,base=None):
    CHOSEN.mkdir(parents=True,exist_ok=True)
    sig=hashlib.sha256(d["text"].encode()).hexdigest()
    dest=CHOSEN/(sig+".npz")
    computed={}
    if dest.exists():
        with np.load(dest,allow_pickle=False) as p:computed={k:p[k] for k in p.files}
    missing=[w for w in widths if str(w) not in computed]
    if missing:
        text=d["text"]
        words=m.words([text[l:r] for l,r in d["units"]])
        for w in missing:
            q=m.contextualize(words) if w==0 else contextual(m,words,w)
            computed[str(w)]=m.logits(q)
        scratch=dest.with_suffix(".part")
        with scratch.open("wb") as stream:np.savez(stream,**computed)
        scratch.replace(dest)
    if base is not None:computed["0"]=base
    return computed

def gap(c,b):
    rng=np.random.default_rng(5227);rows=[]
    for ix in rng.integers(len(c),size=(1300,len(c))):
        q=summarize(c[ix].sum(0));p=summarize(b[ix].sum(0))
        rows.append([q["exact_labeled_spans"]["f1"]-p["exact_labeled_spans"]["f1"],q["token_accuracy"]-p["token_accuracy"]])
    return np.quantile(rows,[.025,.975],axis=0).T.tolist()

def main():
    torch.set_num_threads(2)
    m=HierModel(DONOR)
    docs=validation_cohort()
    parents={k:load_file(str(OUT/f"whole-discourse-{k}.safetensors")) for k in ("exact","balanced")}
    def materialize(ds):
        res=[]
        for index,d in enumerate(ds):
            res.append((d,compute(m,d,(0,*WIDTHS)),text_features(d)))
            if (index+1)%8==0:print("Document context cache",index+1,"/",len(ds),flush=True)
        return res
    def score(table,k,w,s):
        profile=parents[k];ret=[]
        for d,x,(a,b) in table:
            y=x["0"] if not s else (1-s)*x["0"]+s*x[str(w)]
            z=y+a@profile["cue_head"]+b@profile["position_head"]
            ret.append(counts(d,d["units"],path_labels(z,profile[CRF])))
        return np.stack(ret)
    data=materialize(docs["select"]);report={}
    for k in parents:
        rank=None;spec=None;grid=[]
        B=summarize(score(data,k,0,0).sum(0))
        for w,s in [(0,0),*itertools.product(WIDTHS,(.25,.5,1.))]:
            q=summarize(score(data,k,w,s).sum(0));p=(q["exact_labeled_spans"]["f1"],q["token_accuracy"])
            grid.append([w,s,*p])
            if k=="balanced" and p[1] < B["token_accuracy"]-.01:continue
            if rank is None or p>rank:rank=p;spec=w,s
        w,s=spec
        dest=OUT/f"whole-length-adaptive-{k}.safetensors"
        save_file({**parents[k],"context.emission_weight":np.concatenate((m.weight[3:]*(1-s),m.weight[3:]*s),axis=1).astype("f4"),
                   "context.emission_bias":m.bias[3:].astype("f4"),"context.mix":np.array([MIN_WORDS,w,s],dtype="f4")},str(dest))
        report[k]={"width":w,"local_mix":s,"weights_sha256":file_hash(dest),"grid":grid,"scores":{}}
    out=ROOT/"results/length-adaptive-frozen.json"
    save_json(out,{"threshold_whitespace_tokens":MIN_WORDS,"profile":report,"case_ids":{k:[d["id"] for d in ds] for k,ds in docs.items()}})
    frozen=file_hash(out)
    print("Frozen long-document profiles",[(k,v["width"],v["local_mix"]) for k,v in report.items()],flush=True)
    confirm=materialize(docs["confirm"])
    for k,p in report.items():
        w,s=p["width"],p["local_mix"]
        for key,ds in (("select_long",data),("confirm_long",confirm)):
            c=score(ds,k,w,s);b=score(ds,k,0,0)
            p["scores"][key]={"cases":len(c),"candidate":summarize(c.sum(0)),"incumbent":summarize(b.sum(0)),
                              "paired_exact_span_and_token_gap_95pct":gap(c,b)}
    test={k:{"all":[[],[]],"long":[[],[]]} for k in report}
    relevant={p["width"] for p in report.values() if p["local_mix"] and p["width"]}
    for i,(d,rows) in enumerate(gold_cases().items(),1):
        doc= document(d,rows)
        is_long=len(doc["text"].split())>=MIN_WORDS
        with np.load(BASE_CACHE/(d+".npz")) as record:base=record["raw"]
        offsets=text_features(doc)
        above=compute(m,doc,sorted(relevant),base=base) if is_long and relevant else {"0":base}
        row=(doc,above,offsets)
        for k,p in report.items():
            w,s=p["width"],p["local_mix"]
            c=score([row],k,w,s if is_long else 0)[0];b=score([row],k,0,0)[0]
            test[k]["all"][0].append(c);test[k]["all"][1].append(b)
            if is_long:
                test[k]["long"][0].append(c);test[k]["long"][1].append(b)
        if i%72==0:print("Full test",i,"/",712,flush=True)
    for k,p in report.items():
        for name,(a,b) in test[k].items():
            a,b=np.stack(a),np.stack(b)
            p["scores"]["test_"+name]={"cases":len(a),"candidate":summarize(a.sum(0)),"incumbent":summarize(b.sum(0)),
                                        "paired_exact_span_and_token_gap_95pct":gap(a,b)}
        print(k,p["scores"]["test_all"]["candidate"]["exact_labeled_spans"]["f1"],
               p["scores"]["test_long"]["candidate"]["exact_labeled_spans"]["f1"],flush=True)
    save_json(ROOT/"results/length-adaptive-span-write.json",{"threshold_whitespace_tokens":MIN_WORDS,"validation_freeze_sha256":frozen,
                "protocol":"Use a written blend of the complete bidirectional document state and a second state from a local block with context on both sides. Under 8,000 whitespace tokens preserve the current profile. The threshold was fixed after descriptive earlier results. Fit blend hyperparameters only on long validation cases before running confirmation and test. No gradient updates.",
                "profiles":report})
if __name__=="__main__":
    main()
