"""Directly write a disagreement readout targeting sentence-level macro-F1."""
import json
import hashlib
import sys
from pathlib import Path
import numpy as np
import torch
from safetensors.numpy import load_file,save_file
from scipy.special import softmax
import experiment as p
from system import current_cases,h95
sys.path.insert(0,str(p.HERE.parent/"hierarchy"))
from retention_replay import load_arm

DIR=p.HERE/"arbitration"
DIR.mkdir(exist_ok=True)
METRIC_SEED=573591


def archive(split):
    if split=="train":
        source=p.HERE.parent/"ultrabert/historical-incumbent-crossfit.npz"
        with np.load(source,allow_pickle=False) as f:
            return {str(k):(f["predicted"][b:e],f["gold"][b:e]) for k,b,e in zip(f["cases"],f["ends"],f["ends"][1:])}
    return {k:(v,None) for k,v in current_cases(split).items()}


def make(split,selected):
    torch.set_num_threads(1)
    cache=DIR/("features-"+split+".npz")
    info=json.loads((p.HERE/"composition-freeze.json").read_text())
    source=archive(split)
    ids=[k for k in source if k in selected]
    identity=hashlib.sha256(json.dumps([ids,info["head_and_gates"],info["prior"]]).encode()).hexdigest()
    if cache.exists():
        with np.load(cache,allow_pickle=False) as f:
            if f["signature"].item()==identity:
                return {k:f[k] for k in f.files if k!="signature"}
    gate=load_arm("uniform-retention-plus-head.safetensors",info["head_and_gates"])
    recipe=load_file(str(p.HERE/"priors.safetensors"))
    rows={}
    eye=np.eye(7)
    for key,text,g in p.cases("train" if split=="train" else "val",set(ids)):
        native=gate.logits(gate.contextualize(gate.words(text)))
        nativelab=p.path_labels(native,gate.transitions)
        tr=p.targets(text,recipe)
        shift=p.adapted_bias({**tr,"prediction":nativelab},"targeted",info["gain"],gate)-gate.bias
        edited=native+shift[3:]
        choice=p.path_labels(edited,gate.transitions)
        orig,oldgold=source[key]
        if oldgold is not None:np.testing.assert_array_equal(g,oldgold)
        assert len(orig)==len(g)
        n=len(orig)
        loc=np.arange(n)
        pr=softmax(native,axis=1)
        er=softmax(edited,axis=1)
        ix=np.maximum(loc-1,0);after=np.minimum(loc+1,n-1)
        pair=orig*7+choice
        gap=er[loc,choice]-er[loc,orig]
        neighbors=np.column_stack([pred==neighbor for pred in (choice,orig) for neighbor in (orig[ix],orig[after],choice[ix],choice[after])])
        frac=(loc+.5)/n
        x=np.column_stack((np.eye(49)[pair],np.eye(49)[pair]*gap[:,None],er,pr,eye[nativelab],
                neighbors,frac,frac*frac,np.full(n,np.log1p(n)),np.tile(tr["targeted"],(n,1)),
                gap,shift[3:][choice]-shift[3:][orig],(choice==edited.argmax(1)),(orig==native.argmax(1))))
        rows[key]=(g,orig,choice,x.astype(np.float32))
    assert set(rows)==set(ids)
    columns=list(zip(*[rows[k] for k in ids]))
    out={k:np.concatenate(t) for k,t in zip(("gold","base","alt","x"),columns)}
    out.update(keys=np.asarray(ids),offs=np.cumsum([0]+[len(t) for t in columns[0]]))
    cache_tmp=cache.with_suffix(".part")
    with cache_tmp.open("wb") as f:np.savez_compressed(f,signature=identity,**out)
    cache_tmp.replace(cache)
    return out


def matrices(d,pred):
    y=d["gold"];offset=d["offs"]
    return np.stack([p.cm(y[a:b],pred[a:b]) for a,b in zip(offset,offset[1:])])


def row_utility(d):
    y,src,dst=d["gold"],d["base"],d["alt"]
    z=p.cm(y,src)
    den=z.sum(0)+z.sum(1)
    hits=z.diagonal()
    # Exact one-row macro-F1 difference at the training baseline confusion.
    a=2*((hits[src]-(y==src))/(den[src]-1)-hits[src]/den[src])
    b=2*((hits[dst]+(y==dst))/(den[dst]+1)-hits[dst]/den[dst])
    return (a+b)*(len(y)/7)


def apply(d,w,threshold):
    x=np.clip((d["x"]-w["center"])*w["scale"],-5,5)
    score=x@w["coef"]+w["offset"]
    return np.where((d["alt"]!=d["base"])&(score>threshold),d["alt"],d["base"])


def main():
    protocol=DIR/"protocol.json"
    shared=set(json.loads(p.MANIFEST.read_text())["extended_validation_judgment_ids"])
    t=set(archive("train")); s=set(archive("select"))-shared; c=set(archive("confirm"))-shared
    assert not s&c and not t&(s|c|shared)
    recipe={"fit":sorted(t),"select":sorted(s),"confirm":sorted(c),"shared":sorted(shared),
            "fit_population":"144 official train cases, archived crossfitted narrower BERT-review predictions proxy. Public upstream models and all-corpus prior can have seen these cases. Selection and confirmation use exact current 20% outputs, all existing sentences.",
            "alternate":"Native gate plus text-conditioned emission bias, strength fixed by composition manifest.",
            "method":"A ridge readout on class-pair and logit agreement predicts exact one-row change of pooled seven-class F1. Write coefficients via normal equations. Shuffled utility control; no gold at inference.",
            "candidate_settings":{"ridge":[.1,.7],"threshold":[.0,.04,.10,.20,.4,1.e6]},
            "gate":"Pick on select by macro-F1 with accuracy loss at most 0.10 points. Compare on both untouched-by-fit confirmation and shared 48. Require positive lower paired F1 interval on both against incumbent, higher point than shuffled, and accuracy at least incumbent before any official use."}
    if not protocol.exists():p.write(protocol,recipe)
    else:assert json.loads(protocol.read_text())==recipe
    ds={k:make(k,ids) for k,ids in (("train",t),("select",s))}
    train=ds["train"]; sel=ds["select"]
    mask=train["alt"]!=train["base"]
    x=train["x"][mask].astype(float)
    y=row_utility(train)[mask]
    center=x.mean(0);scale=1/np.maximum(x.std(0),.1)
    z=np.column_stack([np.clip((x-center)*scale,-5,5),np.ones(len(x))])
    gram=z.T@z
    selbase=p.stat(matrices(sel,sel["base"]))
    perm=np.random.default_rng(METRIC_SEED).permutation(len(y))
    options=[]; models={}
    for penalty in recipe["candidate_settings"]["ridge"]:
        reg=np.full(len(gram),penalty*np.mean(gram.diagonal()[:-1]));reg[-1]*=.01
        weights=np.linalg.solve(gram+np.diag(reg),z.T@np.column_stack([y,y[perm]]))
        for i,name in enumerate(("targeted","shuffled")):
            models[penalty,name]={"center":center,"scale":scale,"coef":weights[:-1,i],"offset":weights[-1:,i]}
        for threshold in recipe["candidate_settings"]["threshold"]:
            pred=apply(sel,models[penalty,"targeted"],threshold)
            score=p.stat(matrices(sel,pred))
            if score["accuracy"]>=selbase["accuracy"]-.001:
                options.append((score["macro_f1"],score["accuracy"],-int((pred!=sel["base"]).sum()),penalty,threshold))
    chosen=max(options)
    winner={}
    for name in ("targeted","shuffled"):
        for key,value in models[chosen[3],name].items():winner[name+"."+key]=np.ascontiguousarray(value)
    freeze=DIR/"weights.safetensors"
    save_file(winner,str(freeze))
    spec={"select_baseline":selbase,"choice":chosen,"weights":p.file_hash(freeze),"protocol":p.file_hash(protocol),"train_disagreements":int(mask.sum())}
    p.write(DIR/"freeze.json",spec)
    out={}
    for k,ids in (("select",s),("confirm",c),("shared",shared)):
        d=sel if k=="select" else make(k,ids)
        base=d["base"]; gold=d["gold"]; orig=matrices(d,base)
        entry={"cases":len(d["keys"]),"rows":len(gold),"base":p.stat(orig),"alternate":p.stat(matrices(d,d["alt"]))}
        for name in ("targeted","shuffled"):
            pred=apply(d,models[chosen[3],name],chosen[4])
            diff=pred!=base
            cm=matrices(d,pred)
            entry[name]={**p.stat(cm),"delta_ci95":h95(orig,cm),"fixed":int((diff&(pred==gold)).sum()),"broken":int((diff&(base==gold)).sum()),"changed":int(diff.sum())}
        out[k]=entry
        p.write(DIR/"results.json",out)
        print("arbitration",k,{k:round(entry[k]["macro_f1"],5) for k in ("base","alternate","targeted","shuffled")},flush=True)


if __name__=="__main__":
    main()
