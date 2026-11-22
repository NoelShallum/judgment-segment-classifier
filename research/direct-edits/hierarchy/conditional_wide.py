"""Train-only associative write conditioned on the incumbent's proposed role."""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.special import softmax
from safetensors.numpy import load_file,save_file
from native_memory import HERE,ROOT,DONOR,HierModel,metric,confusion,confusion_docs,dump,file_hash,decode
from widened import bank
from replay_native import read_native
from fuse_native import summary
import audit_pairwise_referral_coverage as coverage
import sentence_pairwise_consensus as consensus
import sentence_pairwise_repair as repair

def slice_for(bank_,keys):
    names,ends=bank_[:2];lookup={str(key):(a,b) for key,a,b in zip(names,ends,ends[1:])}
    ix=[np.arange(*lookup[str(key)]) for key in keys]
    return np.r_[0,np.cumsum([len(g) for g in ix])],np.concatenate(ix)

def train_incumbent(keys):
    path=HERE/"strong-pair-train-240.npz"
    if not path.exists():
        bank_=repair.P.dense.prepare("fit");lex=repair.P.cached_scores("fit");ends,ix=slice_for(bank_,keys)
        reduced=(np.asarray(keys),ends,*(a[ix] for a in bank_[2:]))
        parent,weights,_=consensus.inputs();recipe=repair.load_recipe()
        current,_=coverage.routing(reduced,lex[ix],recipe,weights,parent["arms"]["sequence only"]["choice"])
        tmp=path.with_suffix(".part")
        with tmp.open("wb") as fp:np.savez(fp,keys=keys,ends=ends,gold=reduced[2],pred=current)
        tmp.replace(path)
    x=np.load(path);np.testing.assert_array_equal(keys,x["keys"]);return x["gold"],x["pred"]

def features(split):
    if split in ("fit","select"):
        limit=240 if split=="fit" else 71
        f,states=bank(split,limit)
        ends=f["ends"];keys=f["keys"].tolist();y=f["gold"]
        if split=="fit":
            yb,strong=train_incumbent(keys);np.testing.assert_array_equal(y,yb)
        else:
            s=np.load(HERE/"strong-select.npz");_,ix=slice_for((s["keys"],s["ends"]),keys);strong=s["pred"][ix];np.testing.assert_array_equal(y,s["gold"][ix])
        m=HierModel(HERE/"widened-model");z=m.logits(states)
        old,_=read_native(HERE/"native-base-240-71.safetensors")
        ho=np.column_stack((states[:,:100],states[:,114:214]));cz=old.logits(ho)
    else:
        f=np.load(HERE/"widened-shared-16.npz");c=np.load(HERE/"replay-native-base-240-71-shared-16.npz")
        keys=f["keys"].tolist();ends=f["ends"];y=f["gold"];z=f["logits"];cz=c["logits"];strong=f["incumbent"]
    eye=np.eye(7);base=eye[strong];left=base.copy();right=base.copy();phase=np.ones(len(y));doc=np.empty_like(base)
    for a,b in zip(ends,ends[1:]):
        left[a+1:b]=base[a:b-1];right[a:b-1]=base[a+1:b];doc[a:b]=base[a:b].mean(0);phase[a:b]=np.linspace(0,1,b-a)
    po=softmax(cz/2.,axis=1);pw=softmax(z/2.,axis=1)
    common=np.column_stack((po,cz-cz.mean(1,keepdims=True),left,right,doc,phase,phase**2,phase*(1-phase),po*np.maximum(left,right)))
    return keys,ends,y,strong,common,np.column_stack((pw,pw-po,z-z.mean(1,keepdims=True),pw*np.maximum(left,right)))

def fit(x,y,current):
    mu=x.mean(0);sd=np.maximum(x.std(0),.03);x=np.column_stack((np.clip((x-mu)/sd,-6,6),np.ones(len(y))))
    # Fit each conditional belief using only train cases. The parameter slots are seven direct linear heads.
    weight=[]; E=np.eye(7);freq=np.bincount(y,minlength=7)/len(y)
    for r in range(7):
        ok=current==r;xx=x[ok];mass=freq[y[ok]]**-.2;mass/=mass.mean()
        gram=(xx.T*mass)@xx
        d=.16*np.diag(gram).mean()
        reg=np.diag(np.r_[np.ones(xx.shape[1]-1)*d,.01*d])
        weight.append(np.linalg.solve(gram+reg,(xx.T*mass)@E[y[ok]]))
    return {"mean":mu.astype(np.float32),"scale":sd.astype(np.float32),"weight":np.asarray(weight,dtype=np.float32)}

def scores(x,p,w):
    v=np.column_stack((np.clip((x-w["mean"])/w["scale"],-6,6),np.ones(len(x))))
    out=np.empty((len(p),7))
    for r in range(7):out[p==r]=v[p==r]@w["weight"][r]
    return out

def gate(p,z,margin):
    best=z.argmax(1);idx=np.arange(len(p));yes=(z[idx,best]-z[idx,p]>margin)
    return np.where(yes,best,p)

def main(split):
    modelpath=HERE/"conditional-widened.safetensors";freeze=HERE/"conditional-widened-freeze.json"
    keys,ends,y,strong,common,extra=features(split)
    if split=="select":
        trkeys,trends,gt,base,x,c=features("fit");weights={}
        for name,mat in (("readout",x),("widened",np.column_stack((x,c)))):
            for k,v in fit(mat,gt,base).items():weights[name+"."+k]=v
        tmp=modelpath.with_suffix(".part");save_file(weights,str(tmp));tmp.replace(modelpath)
    vals=load_file(str(modelpath));settings={}
    for name,mat in (("readout",common),("widened",np.column_stack((common,extra)))):
        z=scores(mat,strong,{k.split(".",1)[1]:v for k,v in vals.items() if k.startswith(name+".")})
        if split=="select":
            choices=[]
            for margin in (0.,.075,.15,.25,.4,.6,.9,1.4):
                h=gate(strong,z,margin);m=metric(confusion(y,h))
                choices.append((m["macro_f1"],m["accuracy"],-int((h!=strong).sum()),margin))
            chosen=max(choices)[-1]
        else:chosen=json.loads(freeze.read_text())["gates"][name]
        settings[name]=chosen;h=gate(strong,z,chosen)
        if name=="readout":out={}
        out[name]=summary((keys,ends,y,strong),h)
    if split=="select":
        dump(freeze,dict(weights_sha256=file_hash(modelpath),train_cases=trkeys,select_cases=keys,gates=settings,gradient_steps=0,
             train_baseline="Frozen full pairwise production head on official train cases. Selection and shared replay use the full 20% BERT incumbent.",
             sources={"widened":file_hash(HERE/"widened-model/hier.safetensors"),"control":file_hash(HERE/"native-base-240-71.safetensors")}))
    dump(HERE/("conditional-widened-"+split+".json"),out)
    print(split,settings,{k:(v["macro_f1"],v["accuracy"],v["changed"]) for k,v in out.items()},out["widened"]["baseline"]["macro_f1"],flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("split",choices=("select","shared"));main(p.parse_args().split)
