"""Measure case-conditioned native emission edits against the stronger routed system."""
import argparse
import itertools
import json
import numpy as np
from safetensors.numpy import load_file
import experiment as trial


def current_cases(split):
    if split=="shared":
        ids=json.loads(trial.MANIFEST.read_text())["extended_validation_judgment_ids"]
        p=trial.HERE.parent/"pooler/shared-cases"
        return {k:np.asarray(json.loads((p/(k+".json")).read_text())["predictions"]["unchanged"]) for k in ids}
    file=trial.HERE.parent/"hierarchy"/("strong-"+split+".npz")
    with np.load(file,allow_pickle=False) as f:
        return {str(k):p for k,p in zip(f["keys"],np.split(f["pred"],f["ends"][1:-1]))}


def transfer(rows,system,model,source,gain,margin,mechanism,focus):
    counts=[]
    preds=[]
    for row,origin in zip(rows,system):
        bias=trial.adapted_bias(row,source,gain,model)[3:]-model.bias[3:]
        scores=row["emissions"]
        index=np.arange(len(origin))
        if mechanism=="native":
            new=trial.path_labels(scores+bias,model.transitions)
            use=(origin==row["prediction"])&(new!=origin)&((scores+bias)[index,new]>=(scores+bias)[index,origin]+margin)
        else:
            anchored=scores.copy()
            anchored[index,origin]=np.maximum(anchored[index,origin],anchored.max(1)+margin)
            new=(anchored+bias).argmax(1)
            use=new!=origin
        if focus=="none and reasoning":
            use&=np.isin(origin,[0,5])&np.isin(new,[0,5])
        pred=np.where(use,new,origin)
        preds.append(pred)
        counts.append(trial.cm(row["g"],pred))
    return np.asarray(counts),np.concatenate(preds)


def h95(before,after):
    seed=np.random.default_rng(52189)
    draws=seed.integers(len(before),size=(1600,len(before)))
    def macro(m):
        v=m[draws].sum(1)
        return (2*np.diagonal(v,axis1=1,axis2=2)/np.maximum(v.sum(1)+v.sum(2),1)).mean(1)
    return np.quantile(macro(after)-macro(before),[.025,.975]).tolist()


def evaluate(split):
    system=current_cases(split)
    shared=set(json.loads(trial.MANIFEST.read_text())["extended_validation_judgment_ids"])
    ids=[k for k in system if split=="shared" or k not in shared]
    spec=trial.HERE/"transfer-freeze.json"
    prior=load_file(str(trial.HERE/"priors.safetensors"))
    data,model=trial.make_cache(split,ids,prior)
    base=[system[k] for k in ids]
    assert all(r["g"].shape==p.shape for r,p in zip(data,base))
    counts=np.stack([trial.cm(r["g"],p) for r,p in zip(data,base)])
    ref=trial.stat(counts)
    if split=="select":
        runs={}
        for source in ("targeted","constant"):
            grid=[]
            for params in itertools.product((.06,.125,.25),(.05,.2),("native","anchored"),("all","none and reasoning")):
                c,p=transfer(data,base,model,source,*params)
                s=trial.stat(c)
                grid.append([list(params),s["macro_f1"],s["accuracy"],int(np.sum(p!=np.concatenate(base)))])
            eligible=[r for r in grid if r[2]>=ref["accuracy"]-.002]
            chosen=max(eligible or grid,key=lambda r:(r[1],r[2],-r[3]))
            runs[source]={"chosen":chosen,"grid":grid}
        trial.write(spec,{"selection_ids":ids,"base":ref,"arms":runs,"selection_rule":"Choose highest macro-F1 within 0.2 accuracy points of current. Identity reported separately.","prior_weights":trial.file_hash(trial.HERE/"priors.safetensors")})
        print("transfer selected",ref["macro_f1"],{s:r["chosen"] for s,r in runs.items()},flush=True)
        return
    frozen=json.loads(spec.read_text())
    assert not set(ids)&set(frozen["selection_ids"])
    gold=np.concatenate([r["g"] for r in data])
    start=np.concatenate(base)
    ans={"freeze_sha256":trial.file_hash(spec),"judgments":len(ids),"rows":len(gold),"ids":ids,"incumbent":ref,"arms":{}}
    for source,mode in (("targeted","targeted"),("shuffled","targeted"),("constant","constant")):
        settings=frozen["arms"][mode]["chosen"][0]
        c,p=transfer(data,base,model,source,*settings)
        changed=p!=start
        ans["arms"][source]={**trial.stat(c),"changed":int(changed.sum()),"fixed":int((changed&(p==gold)).sum()),
                             "broken":int((changed&(start==gold)).sum()),"ci95_vs_incumbent":h95(counts,c)}
    trial.write(trial.HERE/("transfer-"+split+".json"),ans)
    print("transfer",split,ans["rows"],ref["macro_f1"],{s:r["macro_f1"] for s,r in ans["arms"].items()},flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("split",choices=["select","shared","confirm"])
    evaluate(cli.parse_args().split)
