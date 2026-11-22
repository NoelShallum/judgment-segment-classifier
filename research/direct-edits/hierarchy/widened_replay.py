"""Score every sentence through the saved widened PyTorch checkpoint without the feature shortcut."""
import argparse
import json
import numpy as np
from native_memory import HERE,ROOT,DONOR,HierModel,file_hash,data,decode,confusion_docs,metric,dump
from replay_native import read_native

def run(split,n):
    path=HERE/"widened-model"; m=HierModel(path); spec=json.loads((HERE/"widened-frozen.json").read_text())
    assert file_hash(path/"hier.safetensors")==spec["weights_sha256"]
    base,info=read_native(HERE/"native-base-240-71.safetensors")
    prior=HierModel(DONOR)
    docs=data(split,n);keys=[];ys=[];pred=[];ctl=[];raw=[];inc=[];end=[0];zs=[]
    if split=="confirm" and (HERE/"strong-confirm.npz").exists():
        pri=np.load(HERE/"strong-confirm.npz")
        table={k:pri["pred"][a:b] for k,a,b in zip(pri["keys"],pri["ends"],pri["ends"][1:])}
    else:table={}
    for k,txt,g,cache in docs:
        assert k not in set(spec["train_cases"])|set(spec["select_cases"])
        hh=m.contextualize(m.words(txt))
        z=m.logits(hh); pred.extend(decode(z,[0,len(g)],m.transitions));zs.extend(z)
        oh=prior.contextualize(prior.words(txt) if cache is None else cache)
        raw.extend(decode(prior.logits(oh),[0,len(g)],prior.transitions))
        ctl.extend(decode(base.logits(oh),[0,len(g)],base.transitions))
        if split=="shared":
            inc.extend(json.loads((HERE.parent/"pooler/shared-cases"/(k+".json")).read_text())["predictions"]["unchanged"])
        elif k in table: inc.extend(table[k])
        keys.append(k);ys.extend(g);end.append(len(ys))
    y,h,old,hctl,strong=map(np.asarray,(ys,pred,raw,ctl,inc))
    tmp=HERE/f"widened-{split}-{n}.part"
    with tmp.open("wb") as fp:
        np.savez(fp,gold=y,pred=h,base=old,control=hctl,incumbent=strong,keys=keys,ends=end,logits=np.asarray(zs))
    tmp.replace(tmp.with_suffix(".npz"))
    mats={};sc={};compar={}
    for label,p in (("widened",h),("donor",old),("readout",hctl),("incumbent",strong)):
        if len(p)==0:continue
        mats[label]=confusion_docs(y,p,end)
        sc[label]=metric(mats[label].sum(0))
        if label!="widened":
            rng=np.random.default_rng(989);de=[]
            for j in rng.integers(len(keys),size=(2000,len(keys))):
                nw=metric(mats["widened"][j].sum(0));bs=metric(mats[label][j].sum(0))
                de.append([nw["macro_f1"]-bs["macro_f1"],nw["accuracy"]-bs["accuracy"]])
            compar[label]=dict(corrected=int(((h!=p)&(h==y)).sum()),harmed=int(((h!=p)&(p==y)).sum()),changed=int((h!=p).sum()),ci95=np.quantile(de,[.025,.975],axis=0).T.tolist())
    output=dict(cases=keys,rows=len(y),metrics=sc,paired=compar,frozen_sha256=file_hash(HERE/"widened-frozen.json"),gradient_steps=0)
    dump(tmp.with_suffix(".json"),output);print(split,sc['widened']['macro_f1'],sc['widened']['accuracy'],sc['readout']['macro_f1'],sc.get('incumbent',{}).get('macro_f1'),flush=True)
if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("split",choices=("shared","confirm"));a.add_argument("n",type=int);p=a.parse_args();run(p.split,p.n)
