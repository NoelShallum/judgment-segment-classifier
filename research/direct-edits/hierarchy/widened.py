"""Preserve the donor functions and append independent lexical-memory cells to both directions."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from safetensors.numpy import load_file,save_file
from scipy.special import expit
from scipy.signal import lfilter
from native_memory import HERE,ROOT,DONOR,BASE,HierModel,TO_ID,confusion,decode,dump,file_hash,fit_head,metric,native,data,store_state

WIDTH=114
GATES=((-0.6,2.6,4.0,.24),(2.,-1.4,4.,1.1),(2.,-.3,4.,1.1))

def book():
    mat=np.load(HERE/"counts.npz")["counts"].transpose(0,2,1).astype(np.float64)
    c=.65*mat[0]+.2*mat[1]+.15*mat[2];freq=c.sum(1)
    prior=c.sum(0)/c.sum()
    p=(c+20*prior)/(freq[:,None]+20)
    odds=np.clip(np.log(p/prior),-3,3);odds-=odds.mean(1,keepdims=True)
    odds*=np.minimum(1,freq/12)[:,None];odds[:2]=0
    beta=np.load(HERE/"residual-dictionary.npz")["word_scores"]
    return np.column_stack((odds,beta)).astype(np.float32)

def signal(toks,b):
    # Standard PyTorch LSTM equations, solved as independent scalar filters for the appended channels.
    si,sf,so,sc=GATES[0]; inp=expit(si); forget=expit(sf); out=expit(so)
    n=len(toks);fw=np.empty((n,14));bw=np.empty_like(fw)
    cache=np.tanh(sc*b)
    for j,keys in enumerate(toks):
        val=cache[keys]
        coeff=forget**np.arange(len(keys))
        fw[j]=out*np.tanh(inp*(coeff[::-1]@val))
        bw[j]=out*np.tanh(inp*(coeff@val))
    pieces=[]
    for i,(x,y) in enumerate(((fw,bw),(bw[::-1],fw[::-1]))):
        a,f,o,g=GATES[1+i]
        stim=expit(a)*np.tanh(g*(.80*x+.55*y))
        mem=lfilter([1.],[1.,-expit(f)],stim,axis=0)
        h=expit(o)*np.tanh(mem)
        pieces.append(h if i==0 else h[::-1])
    return np.column_stack(pieces).astype(np.float32)

def added(split,n):
    path=HERE/f"widened-channels-{split}-{n}.npz"
    if not path.exists():
        m=HierModel(DONOR); dic=book();docs=data(split,n)
        vals=[];ys=[];names=[];ends=[0]
        for key,txt,gold,_ in docs:
            names.append(key);ys.extend(gold);ends.append(len(ys))
            vals.append(signal([m.tokenize(s) for s in txt],dic))
        temp=path.with_suffix(".part")
        with temp.open("wb") as fp: np.savez(fp,new=np.concatenate(vals),gold=ys,keys=names,ends=ends)
        temp.replace(path)
    return np.load(path)

def combine(old,new):
    return np.column_stack((old[:,:100],new[:,:14],old[:,100:],new[:,14:]))

def bank(split,n):
    f=added(split,n)
    b=np.load(store_state(split,"base",n))
    np.testing.assert_array_equal(f["keys"],b["keys"]);np.testing.assert_array_equal(f["gold"],b["gold"])
    return f,combine(b["states"],f["new"])

def checkpoint(h):
    B=WIDTH; new=np.arange(100,B); source=load_file(str(BASE)); changed={}
    dic=book();changed["sent_encoder.emb.weight"]=np.column_stack((source["sent_encoder.emb.weight"],dic))
    for level,prefix in enumerate(("sent_encoder.lstm","emitter.lstm")):
        for rev,suff in enumerate(("","_reverse")):
            i,f,o,g=GATES[0 if level==0 else 1+rev]
            for suffix in ("weight_ih_l0","weight_hh_l0","bias_ih_l0","bias_hh_l0"):
                key=f"{prefix}.{suffix}{suff}"
                a=source[key]
                if a.ndim==1:
                    v=np.zeros(4*B,np.float32)
                elif a.shape[1]==100:
                    v=np.zeros((4*B,B),np.float32)
                else:
                    v=np.zeros((4*B,2*B),np.float32)
                for gate in range(4):
                    to=slice(gate*B,gate*B+100);old=slice(100*gate,100*gate+100)
                    if v.ndim==1:
                        v[to]=a[old]
                        if suffix=="bias_ih_l0":v[gate*B+new]=(i,f,0.,o)[gate]
                    elif a.shape[1]==100:v[to,:100]=a[old]
                    else:
                        v[to,:100]=a[old,:100];v[to,B:B+100]=a[old,100:]
                if suffix=="weight_ih_l0":
                    if level==0: v[2*B+new,new]=g
                    else:
                        src=new+B*rev; adj=new+B*(1-rev)
                        v[2*B+new,src]=.8*g;v[2*B+new,adj]=.55*g
                changed[key]=v
    changed["crf.transitions"]=source["crf.transitions"]
    p=source["emitter.hidden2tag.weight"]
    w=np.zeros((len(p),2*B),np.float32);w[:,:100]=p[:,:100];w[:,B:B+100]=p[:,100:]
    w[3:]=h[:-1].T;bias=source["emitter.hidden2tag.bias"].copy();bias[3:]=h[-1]
    changed["emitter.hidden2tag.weight"]=w;changed["emitter.hidden2tag.bias"]=bias
    return changed

def main():
    x,states=bank("fit",240); v,dev=bank("select",71);m=HierModel(DONOR)
    old=m.weight[3:].T
    prior=np.row_stack((old[:100],np.zeros((14,7)),old[100:],np.zeros((14,7)),m.bias[3:]))
    y=x["gold"];gt=v["gold"];ends=v["ends"];opts={};fit={}
    for power in (0.,.28,.6):
        for reg in (.03,.3):
            full=fit_head(states.astype(float),y,prior,power,reg)
            for scale in (.5,1.25,2.5):
                h=(prior+scale*(full-prior)).astype(np.float32)
                pred=decode(dev@h[:-1]+h[-1],ends,m.transitions)
                tag=f"{power},{reg},{scale}";opts[tag]=metric(confusion(gt,pred));fit[tag]=h
    chosen=max(opts,key=lambda k:(opts[k]["macro_f1"],opts[k]["accuracy"]))
    h=fit[chosen]; state=checkpoint(h);folder=HERE/"widened-model";folder.mkdir(exist_ok=True)
    for n in ("word2idx.json","tag2idx.json"):
        if not (folder/n).exists(): (folder/n).symlink_to(DONOR/n)
    path=folder/"hier.safetensors";tmp=folder/"hier.tmp";save_file(state,str(tmp));tmp.replace(path)
    proof=data("select",71);model=HierModel(folder)
    selected=[min(range(len(proof)),key=lambda j:len(proof[j][1])),0]
    errors=[]
    for j in selected:
        _,txt,_,_=proof[j]; ref=dev[ends[j]:ends[j+1]]; live=model.contextualize(model.words(txt))
        err=float(np.abs(ref-live).max());errors.append(err)
        zp=ref@h[:-1]+h[-1];zl=model.logits(live)
        np.testing.assert_array_equal(decode(zp,[0,len(zp)],m.transitions),decode(zl,[0,len(zl)],model.transitions))
        assert err<3e-5,err
    mask=np.r_[np.arange(100),np.arange(WIDTH,WIDTH+100)]
    control=load_file(str(HERE/"native-base-240-71.safetensors"))
    frozen=dict(chosen=chosen,scores=opts,train_cases=x["keys"].tolist(),select_cases=v["keys"].tolist(),train_rows=len(y),select_rows=len(gt),
                gradient_steps=0,model_parameters=model.parameters,old_dimensions=100,new_dimensions=WIDTH,
                sources={"donor":file_hash(BASE),"counts":file_hash(HERE/"counts.npz"),"residual":file_hash(HERE/"residual-dictionary.npz")},
                weights_sha256=file_hash(path),normal_inference_parity=dict(cases=[proof[i][0] for i in selected],max_state_difference=errors,decoded_disagreements=0),
                shapes={k:list(a.shape) for k,a in state.items()})
    dump(HERE/"widened-frozen.json",frozen)
    np.savez(HERE/"widened-select-71.npz",keys=v["keys"],gold=gt,ends=ends,logits=dev@h[:-1]+h[-1],pred=decode(dev@h[:-1]+h[-1],ends,m.transitions))
    print("widened",model.parameters,chosen,opts[chosen]["macro_f1"],opts[chosen]["accuracy"],errors,flush=True)

if __name__=="__main__":main()
