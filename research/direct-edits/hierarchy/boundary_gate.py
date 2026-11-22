"""Write train-estimated role-boundary keys directly into native LSTM forget/input gates."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from safetensors.numpy import save_file
from native_memory import HERE,ROOT,DONOR,BASE,HierModel,data,native,confusion_docs,metric,decode,dump,file_hash
from replay_native import read_native
from embedding_write import digest

def train():
    path=HERE/"boundary-keys.npz"
    if path.exists():return np.load(path)
    d=np.load(HERE/"native-base-fit-240.npz");v=np.load(ROOT/"cache/sentence-recurrent-inverse/fit.npz")
    np.testing.assert_array_equal(d["keys"],v["keys"]);np.testing.assert_array_equal(d["ends"],v["ends"])
    h=d["states"];x=v["words"];y=d["gold"];off=d["ends"]
    arrays={}
    for i in range(2):
        prev=np.zeros((len(y),100),np.float32);target=np.ones(len(y),np.float32);valid=np.ones(len(y),bool)
        for a,b in zip(off,off[1:]):
            if i==0:
                prev[a+1:b]=h[a:b-1,:100];target[a+1:b]=y[a+1:b]!=y[a:b-1];valid[a]=False
            else:
                prev[a:b-1]=h[a+1:b,100:];target[a:b-1]=y[a:b-1]!=y[a+1:b];valid[b-1]=False
        z=np.column_stack((x,prev))[valid].astype(np.float64)
        t=target[valid];mu=z.mean(0);sd=z.std(0).clip(.02);A=(z-mu)/sd
        gram=A.T@A;lam=.12*gram.trace()/len(gram)
        score=cho_solve(cho_factor(gram+lam*np.eye(gram.shape[0])),A.T@(t-t.mean()))/sd
        intercept=-mu@score
        scale=np.std(z@score+intercept)
        score/=scale;intercept/=scale
        arrays["weight"+str(i)]=score.astype(np.float32);arrays["bias"+str(i)]=np.float32(intercept)
        arrays["target_rate"+str(i)]=np.float32(t.mean());arrays["train_total"+str(i)]=len(t)
        rng=np.random.default_rng(9162+i)
        shuffle=score.copy();shuffle[:200]=shuffle[rng.permutation(200)];shuffle[200:]=shuffle[200+rng.permutation(100)]
        shuffle_bias=-mu@shuffle;ss=np.std(z@shuffle+shuffle_bias);shuffle/=ss;shuffle_bias/=ss
        arrays["shuf_weight"+str(i)]=shuffle.astype(np.float32);arrays["shuf_bias"+str(i)]=np.float32(shuffle_bias)
    tmp=path.with_suffix(".part")
    with tmp.open("wb") as f:np.savez(f,**arrays)
    tmp.replace(path);return np.load(path)

def apply(m,bank,mode,gain):
    p=native(m);updates={}
    for i,suff in enumerate(("","_reverse")):
        pre="shuf_" if mode=="shuf" else ""
        w=bank[pre+"weight"+str(i)]
        b=bank[pre+"bias"+str(i)]
        for name,vector in (("weight_ih_l0",w[:200]),("weight_hh_l0",w[200:]),("bias_ih_l0",b)):
            k="emitter.lstm."+name+suff;v=p[k].numpy();a=v.copy()
            v[100:200]-=gain*vector
            if mode in ("gated","shuf"): v[:100]+=gain*vector
            updates[k]=dict(shape=list(a.shape),changed=int((v!=a).sum()),old_sha256=digest(a),new_sha256=digest(v),delta_L2=float(np.linalg.norm(v-a)))
    return updates

def evaluate(split):
    n={"select":71,"confirm":70,"shared":16}[split];docs=data(split,n)
    read,_=read_native(HERE/"native-base-240-71.safetensors")
    data_=train();keys=[d[0] for d in docs];y=np.concatenate([d[2] for d in docs]);ends=np.r_[0,np.cumsum([len(d[2]) for d in docs])]
    words=[d[-1] if d[-1] is not None else read.words(d[1]) for d in docs]
    if split=="select":
        arms=[("base",0),*((name,s) for name in ("forget","gated","shuf") for s in (.08,.22,.5,1.)),("gated",-.22)]
    else:
        sel=json.loads((HERE/"boundary-frozen.json").read_text())["arms"]
        arms=[("base",0)]+[tuple(item) for item in sel.values()]
    report={"split":split,"cases":keys,"rows":len(y),"gradient_steps":0,"arms":{}}
    original=None
    for mode,g in arms:
        m=HierModel(DONOR);patch=apply(m,data_,mode,g) if g else {}
        h=np.concatenate([m.contextualize(w) for w in words]);p=decode(read.logits(h),ends,m.transitions)
        if original is None:original=p
        count=confusion_docs(y,p,ends);base=confusion_docs(y,original,ends)
        rng=np.random.default_rng(390);boot=[]
        for j in rng.integers(len(keys),size=(1000,len(keys))):
            a,b=metric(count[j].sum(0)),metric(base[j].sum(0));boot.append([a["macro_f1"]-b["macro_f1"],a["accuracy"]-b["accuracy"]])
        info=dict(**metric(count.sum(0)),changed=int((original!=p).sum()),correct=int(((original!=p)&(p==y)).sum()),broken=int(((original!=p)&(original==y)).sum()),
                  paired95=np.quantile(boot,[.025,.975],axis=0).T.tolist(),tensors=patch)
        report["arms"][str((mode,g))]=info;dump(HERE/f"boundary-{split}.json",report)
        print(split,mode,g,info["macro_f1"],info["accuracy"],flush=True)
    if split=="select":
        chosen={name:max([arm for arm in arms if arm[0]==name and arm[1]>0],key=lambda a:report["arms"][str(a)]["macro_f1"]) for name in ("forget","gated","shuf")}
        dump(HERE/"boundary-frozen.json",dict(arms=chosen,train_rows=int(data_["train_total0"]),train_cases=240,selection_cases=keys,sha256_key=file_hash(HERE/"boundary-keys.npz"),baseline_sha256=file_hash(BASE),readout_sha256=file_hash(HERE/"native-base-240-71.safetensors"),gradient_steps=0))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("split",choices=("select","confirm","shared"));evaluate(p.parse_args().split)
