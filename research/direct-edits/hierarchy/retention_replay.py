"""Evaluate the locked retention edit. The full official test requires a passing separate validation gate."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file
import torch
from native_memory import HERE,ROOT,DONOR,BASE,HierModel,TO_ID,native,confusion_docs,metric,decode,dump,file_hash

MANIFEST_SHA="eff466c74aa6caa2791315ce668339cd2236a910f08063818ceebb1d41567c8d"
def load_arm(fname,expected,head=None):
    m=HierModel(DONOR)
    for name,sha in ([(head[0],head[1])] if head else [])+([(fname,expected)] if fname else []):
        source=HERE/name;assert file_hash(source)==sha
        destination=native(m)
        for k,arr in load_file(str(source)).items():destination[k].numpy()[:]=arr
    return m

def run(official):
    manifest=HERE/"uniform-retention-gate.json";assert file_hash(manifest)==MANIFEST_SHA
    cfg=json.loads(manifest.read_text()); assert file_hash(BASE)==cfg["donor"]
    if official:
        result=HERE/"uniform-retention-reserved-score.json"
        check=json.loads(result.read_text());assert check["manifest_sha256"]==MANIFEST_SHA and check["gate_passed"]
        if (HERE/"uniform-retention-test-score.json").exists():raise RuntimeError("Official replay already complete; no further scoring")
    head=(cfg["head_file"],cfg["head_sha256"])
    models={"donor":load_arm(None,None),"head":load_arm(*head),"targeted":load_arm(cfg["combined_file"],cfg["combined_sha256"])}
    if not official:
        for name,control in cfg["controls"].items():models[name]=load_arm(control["file"],control["sha256"],head)
    src=ROOT/"references/official"/("test.csv" if official else "val.csv")
    pred={mode:[] for mode in models}
    cache={}
    if official:
        # Use immutable word-level states from the original donor; these two modified gates occur downstream.
        with np.load(ROOT/"cache/sentence-recurrent-inverse/test.npz") as f:
            assert f["backbone"].item()==file_hash(BASE);ids=f["keys"].tolist();off=f["ends"];state=f["words"]
            cache={k:torch.from_numpy(state[a:b]) for k,a,b in zip(ids,off,off[1:])}
        assert len(ids)==712
    else:ids=cfg["cases"]
    rows={k:([],[]) for k in ids};blank=0;count_raw=0
    with src.open(newline="",encoding="utf-8-sig") as fp:
        for item in csv.DictReader(fp):
            k=item["Index"]
            if k not in rows:continue
            count_raw+=1;t=item["Text"].strip()
            if t:
                rows[k][0].append(t);rows[k][1].append(TO_ID[item["Label"]])
            else:blank+=1
    label=[];ends=[0];origin=models["donor"]
    for i,k in enumerate(ids):
        t,g=rows[k];label.extend(g);ends.append(len(label))
        if k in cache:
            w=cache[k];assert len(w)==len(t)
        else:w=origin.words(t)
        old_state=origin.contextualize(w)
        for mode,m in models.items():
            z=m.logits(old_state if mode in ("head","donor") else m.contextualize(w))
            pred[mode].extend(decode(z,[0,len(g)],m.transitions))
        if (i+1)%8==0:print("test" if official else "reserved",i+1,"/",len(ids),"judgments",len(label),"sentences",flush=True)
    y=np.array(label);pred={k:np.array(v) for k,v in pred.items()}; scores={}; tables={}; stats={}
    for k,v in pred.items():
        tables[k]=confusion_docs(y,v,ends);scores[k]=metric(tables[k].sum(0))
    for k,arm in pred.items():
        if k=="targeted":continue
        new=pred["targeted"];changed=new!=arm
        rng=np.random.default_rng(cfg["bootstrap_seed"]); boot=[]
        for selection in rng.integers(len(ids),size=(cfg["bootstraps"],len(ids))):
            a=metric(tables["targeted"][selection].sum(0));b=metric(tables[k][selection].sum(0))
            boot.append((a["macro_f1"]-b["macro_f1"],a["accuracy"]-b["accuracy"]))
        stats[k]={**dict(changed=int(changed.sum()),corrected=int((changed&(new==y)).sum()),harmed=int((changed&(arm==y)).sum())),
               "boot95_F1_accuracy":np.quantile(boot,[.025,.975],axis=0).T.tolist()}
    if official:
        passed=None
    else:
        passed=bool(stats["head"]["boot95_F1_accuracy"][0][0]>0 and scores["targeted"]["macro_f1"]>scores["gate-shuffled"]["macro_f1"] and scores["targeted"]["accuracy"]>=scores["head"]["accuracy"])
    stem="uniform-retention-"+("test" if official else "reserved")+"-score"
    out={"manifest_sha256":MANIFEST_SHA,"split":"official test" if official else "separately reserved validation","cases":ids,"rows":len(y),"raw_csv_rows":count_raw,
         "skipped_blank_rows":blank,"source_csv_sha256":file_hash(src),"arms":scores,"paired":stats,"gate_passed":passed,"gate_spec":cfg["primary_gate"],"gradient_steps":0}
    tmp=HERE/(stem+".part")
    with tmp.open("wb") as f:np.savez(f,keys=ids,ends=ends,gold=y,**pred)
    tmp.replace(HERE/(stem+".npz"));dump(HERE/(stem+".json"),out)
    print(stem,{k:(s["macro_f1"],s["accuracy"]) for k,s in scores.items()},"gate",passed,stats["head"],flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--official",action="store_true");arg=p.parse_args();run(arg.official)
