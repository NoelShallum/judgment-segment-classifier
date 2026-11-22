"""Validation-only tests of an explicitly gated native-role checkpoint versus the 20% reviewer."""
import argparse
import itertools
import json
import numpy as np
from native_memory import HERE,ROOT,DONOR,HierModel,data,decode,metric,dump,file_hash,confusion,confusion_docs
from replay_native import read_native

def get(split,kind="odds"):
    name="native-"+kind+"-240-71"
    donor="native-base-240-71"
    if split=="shared":
        f=np.load(HERE/f"replay-{name}-shared-16.npz")
        c=np.load(HERE/f"replay-{donor}-shared-16.npz")
        gold=f["gold"];ids=f["keys"];ends=f["ends"]
        np.testing.assert_array_equal(c["keys"],ids)
        np.testing.assert_array_equal(c["gold"],gold)
        path=HERE.parent/"pooler/shared-cases"
        s=np.concatenate([json.loads((path/(str(k)+".json")).read_text())["predictions"]["unchanged"] for k in ids])
        return ids,ends,gold,s,f["pred"],f["logits"],c["pred"],c["logits"]
    if split=="select":
        def cached(arm):
            states=np.load(HERE/("native-"+arm+"-select-71.npz"))
            h,_=read_native(HERE/("native-"+arm+"-240-71.safetensors"))
            z=h.logits(states["states"])
            return states["keys"],states["ends"],states["gold"],decode(z,states["ends"],h.transitions),z
        ids,ends,gold,new,z=cached(kind)
        _,_,gold2,control,cz=cached("base")
    else:
        f=np.load(HERE/f"replay-{name}-{split}-70.npz")
        c=np.load(HERE/f"replay-{donor}-{split}-70.npz")
        ids,ends,gold,new,z=(f[k] for k in ("keys","ends","gold","pred","logits"))
        gold2,control,cz=(c[k] for k in ("gold","pred","logits"))
    np.testing.assert_array_equal(gold,gold2)
    saved=np.load(HERE/("strong-"+split+".npz"))
    full={str(k):slice(a,b) for k,a,b in zip(saved["keys"],saved["ends"],saved["ends"][1:])}
    ix=np.concatenate([np.arange(full[str(k)].start,full[str(k)].stop) for k in ids])
    np.testing.assert_array_equal(gold,saved["gold"][ix])
    return ids,ends,gold,saved["pred"][ix],new,z,control,cz

def apply(pack,rule):
    ids,ends,gold,strong,new,z,control,cz=pack
    mode,margin,step,agree=rule
    a=np.arange(len(new)); output=strong.copy()
    candidate=new
    score=z[a,candidate]-z[a,strong]
    if mode=="counterfactual":
        improvement=score-(cz[a,candidate]-cz[a,strong])
    elif mode=="standalone":
        improvement=np.full(len(score),np.inf)
    else:
        score=cz[a,control]-cz[a,strong]
        candidate=control;improvement=np.full(len(score),np.inf)
    mask=(candidate!=strong)&(candidate==((z if mode!="readout" else cz).argmax(1)))&(score>=margin)&(improvement>=step)
    if agree:
        neigh=np.zeros(len(gold),bool)
        for a,b in zip(ends,ends[1:]):
            neigh[a+1:b]|=candidate[a+1:b]==candidate[a:b-1]
            neigh[a:b-1]|=candidate[a+1:b]==candidate[a:b-1]
        mask&=neigh
    output[mask]=candidate[mask]
    return output

def summary(pack,h):
    ids,ends,g,b,*_=pack
    table=confusion_docs(g,h,ends); old=confusion_docs(g,b,ends)
    rng=np.random.default_rng(499)
    delta=[]
    for j in rng.integers(len(ids),size=(1500,len(ids))):
        v,w=metric(table[j].sum(0)),metric(old[j].sum(0))
        delta.append((v["macro_f1"]-w["macro_f1"],v["accuracy"]-w["accuracy"]))
    return dict(**metric(table.sum(0)),baseline=metric(old.sum(0)),changed=int((b!=h).sum()),
        corrected=int(((h!=b)&(h==g)).sum()),harmed=int(((h!=b)&(b==g)).sum()),
        paired95_f1_accuracy=np.quantile(delta,[.025,.975],axis=0).T.tolist(),rows=len(g),cases=list(ids))

def main(split,kind):
    path=HERE/("fused-"+kind+"-freeze.json")
    pack=get(split,kind)
    if split=="select":
        choices={}
        for mode in ("standalone","counterfactual","readout"):
            arms=[]
            for margin,step,agree in itertools.product((3.,5.,8.,12.,18.,25.),((0.,.5,1.,2.,4.) if mode=="counterfactual" else (0.,)),(False,True)):
                rule=(mode,margin,step,agree)
                h=apply(pack,rule);score=metric(confusion(pack[2],h))
                arms.append((score["macro_f1"],score["accuracy"],-int((h!=pack[3]).sum()),rule))
            best=max(arms);choices[mode]=best[-1]
        dump(path,dict(kind=kind,checkpoints={s:file_hash(HERE/("native-"+s+"-240-71.safetensors")) for s in ("base",kind)},selection_cases=list(pack[0]),rules=choices))
    spec=json.loads(path.read_text())
    out={name:summary(pack,apply(pack,rule)) for name,rule in spec["rules"].items()}
    dump(HERE/f"fused-{kind}-{split}.json",out)
    print(split,{name:(s["macro_f1"],s["accuracy"],s["changed"]) for name,s in out.items()},"incumbent",out["standalone"]["baseline"]["macro_f1"],flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("split",choices=("select","shared","confirm"));p.add_argument("--kind",default="odds")
    a=p.parse_args();main(a.split,a.kind)
