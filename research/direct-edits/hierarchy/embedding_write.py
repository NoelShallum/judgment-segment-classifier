"""Direct, analytically specified edits to the frozen LegalSeg word embedding bank."""
import argparse
from collections import defaultdict
import csv
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file
import torch

from lex_counts import HERE, ROOT, NAMES, TO_ID, dump, file_hash
sys.path.insert(0,str(ROOT))
from hier_model import HierModel, path_labels

DONOR = ROOT/"references/hier"
BASE = DONOR/"hier.safetensors"
TENSOR = "sent_encoder.emb.weight"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()

def load_docs(split,n):
    source = ROOT/"cache/sentence-recurrent-inverse"/(split+".npz")
    with np.load(source) as f:
        assert f["backbone"].item() == file_hash(BASE)
        ids = f["keys"][:n].tolist()
        ends = f["ends"][:len(ids)+1]
        wt = f["words"][:ends[-1]]
    docs = {k:([],[]) for k in ids}
    with (ROOT/"references/official/val.csv").open(newline="",encoding="utf-8-sig") as fp:
        for row in csv.DictReader(fp):
            if row["Index"] in docs and row["Text"].strip():
                texts,gold = docs[row["Index"]]
                texts.append(row["Text"].strip())
                gold.append(TO_ID[row["Label"]])
    out = []
    for i,k in enumerate(ids):
        txt,g=docs[k]
        words = torch.from_numpy(wt[ends[i]:ends[i+1]])
        assert len(txt)==len(words)
        out.append((k,txt,np.asarray(g),words))
    return out

def probe_directions(model,threshold=.075):
    """Finite-difference 7 by 100 map via natural 3- to 9-token train fragments.
    Probe middle sentence inside a three-sentence document with left/right real adjacent sentences.
    No labels select the sentence or token, and no gradients are computed.
    """
    rng=np.random.default_rng(13394)
    options=[]
    with (ROOT/"references/official/train.csv").open(newline="",encoding="utf-8-sig") as fp:
        previous=None
        for row in csv.DictReader(fp):
            if not row["Text"].strip(): continue
            now=(row["Index"],row["Text"].strip())
            if len(options) and len(options[-1])==2:
                a,b=options[-1]
                if now[0]==rowkey:
                    options[-1]=(a,b,now[1])
                else:
                    options.pop()
            if previous and previous[0]==now[0] and rng.random()<.075:
                options.append((previous[1],now[1]));rowkey=now[0]
            if len(options)>=56 and len(options[-1])==3: break
            previous=now
    points=np.concatenate((np.eye(100),-np.eye(100)),axis=0).astype(np.float32)*threshold
    p=torch.from_numpy(points)
    jac=[]
    with torch.inference_mode():
        for prev,text,nxt in options:
            tok=model.tokenize(text)
            width=min(len(tok),int(rng.integers(3,10)))
            start=int(rng.integers(0,len(tok)-width+1))
            ids=tok[start:start+width]
            idx=int(rng.integers(0,len(ids)))
            x=model.word[ids][None,:,:].repeat(200,1,1)
            x[:,idx]+=p
            _,(h,_)=model.sent(x)
            target=h.transpose(0,1).reshape(200,-1)
            lr=model.words([prev,nxt])
            doc=torch.stack((lr[0].expand(200,-1),target,lr[1].expand(200,-1)),dim=1)
            hh,_=model.doc(doc)
            y=model.logits(hh[:,1].numpy())
            dy=(y[:100]-y[100:])/(2*threshold)
            jac.append(dy-dy.mean(1,keepdims=True))
    return np.mean(jac,axis=0)

def plan(model,kind):
    stat=np.load(HERE/"counts.npz")
    counts=stat["counts"].transpose(0,2,1)
    counts=.65*counts[0]+.2*counts[1]+.15*counts[2]
    freq=counts.sum(1)
    basep=counts.sum(0)/counts.sum()
    eligible=(freq>=12)&(np.arange(len(freq))>=2)&(np.arange(len(freq))<len(model.word))
    idx=np.flatnonzero(eligible)
    p=(counts[idx,:]+20*basep)/(freq[idx,None]+20)
    s=np.clip(np.log(p/basep),-3.5,3.5)
    s-=s.mean(1,keepdims=True)
    vec=model.word.numpy()[idx].astype(float)
    mu=np.average(vec,axis=0,weights=np.sqrt(freq[idx]))
    v=vec-mu
    gram=v.T@(np.sqrt(freq[idx,None])*v)
    target=v.T@(np.sqrt(freq[idx,None])*s)
    ridge=.05*np.trace(gram)/len(gram)
    linear=np.linalg.solve(gram+ridge*np.eye(gram.shape[0]),target)
    if kind.startswith("cov"):
        mapping=linear.T
    else:
        cache=HERE/"probe.npy"
        if not cache.exists():
            arr=probe_directions(model)
            with cache.with_suffix(".part").open("wb") as fp: np.save(fp,arr)
            cache.with_suffix(".part").replace(cache)
        J=np.load(cache)
        G=J.T@J
        mapping=np.linalg.solve(G+.15*np.trace(G)/7*np.eye(7),J.T)
    if "res" in kind: s-=v@linear
    if "shuf" in kind:
        rng=np.random.default_rng(9732)
        for a,b in ((12,30),(30,100),(100,1000),(1000,10000),(10000,np.inf)):
            group=np.flatnonzero((freq[idx]>=a)&(freq[idx]<b))
            s[group]=s[rng.permutation(group)]
    # Weighted public-train evidence sets the length of each edit; 95% lie below one donor-row norm at strength one.
    delta=s@mapping
    dn=np.linalg.norm(delta,axis=1)
    scale=np.linalg.norm(vec,axis=1)/np.maximum(dn,1.e-10)
    norm_scale=np.percentile(scale,5)
    delta*=np.minimum(scale,norm_scale)[:,None]
    return idx,delta.astype(np.float32)

def confusion(gold,pred):
    return np.bincount(gold*7+pred,minlength=49).reshape(7,7)

def metric(c):
    f=2*c.diagonal()/np.maximum(c.sum(0)+c.sum(1),1)
    return dict(macro_f1=float(f.mean()),accuracy=float(c.diagonal().sum()/c.sum()),f1=f.tolist(),support=c.sum(1).tolist())

def evaluate(model,docs,use_cached):
    y=[]; pred=[]; scores=[]
    for key,txt,gt,words in docs:
        if not use_cached:
            words=model.words(txt)
        st=model.contextualize(words); logits=model.logits(st)
        ys=path_labels(logits,model.transitions)
        pred.append(ys); y.append(gt); scores.append(logits)
    return np.concatenate(y),np.concatenate(pred),np.concatenate(scores)

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--n",type=int,default=12)
    parser.add_argument("--split",default="select",choices=("select","confirm","holdout"));parser.add_argument("--arms",default="cov-res:0.28,probe-res:0.28,cov:0.28,probe:0.28");args=parser.parse_args()
    model=HierModel(DONOR); raw=model.word.clone(); docs=load_docs(args.split,args.n)
    name=f"{args.split}-{len(docs)}"
    path=HERE/(name+".json")
    old=__import__("json").loads(path.read_text()) if path.exists() else {}
    raw_y,raw_h,raw_logits=evaluate(model,docs,True)
    np.savez(HERE/(name+"-base.npz"),pred=raw_h,logits=raw_logits,gold=raw_y,keys=[d[0] for d in docs],ends=np.r_[0,np.cumsum([len(d[2]) for d in docs])])
    old.update(baseline=metric(confusion(raw_y,raw_h)),params=model.parameters,tensor=TENSOR,shape=list(raw.shape),input="whole judgment, public sentence boundaries, seven roles, blank rows excluded",cases=[d[0] for d in docs],rows=len(raw_y),weights_source=file_hash(BASE),gradient_steps=0)
    dump(path,old); print(name,"baseline",old["baseline"]["macro_f1"],old["baseline"]["accuracy"],flush=True)
    for arm in args.arms.split(","):
        key,strength=arm.split(":");strength=float(strength)
        if arm in old: continue
        st=time.monotonic();model.word=raw
        idx,delta=plan(model,key)
        model.word=raw.clone()
        model.word[idx]+=torch.from_numpy(delta)*strength
        y,h,logits=evaluate(model,docs,False)
        np.testing.assert_equal(y,raw_y)
        changes=(model.word[idx]-raw[idx]).numpy()
        matrices=np.stack([confusion(d[2],p) for d,p in zip(docs,np.split(h,np.cumsum([len(d[2]) for d in docs])[:-1]))])
        before=np.stack([confusion(d[2],p) for d,p in zip(docs,np.split(raw_h,np.cumsum([len(d[2]) for d in docs])[:-1]))])
        rng=np.random.default_rng(57291)
        boots=[]
        for j in rng.integers(len(docs),size=(1000,len(docs))):
            boots.append(metric(matrices[j].sum(0))["macro_f1"]-metric(before[j].sum(0))["macro_f1"])
        old[arm]=dict(**metric(matrices.sum(0)),changes=int((h!=raw_h).sum()),corrected=int(((h!=raw_h)&(h==y)).sum()),broken=int(((h!=raw_h)&(raw_h==y)).sum()),changed_rows=int((changes!=0).any(1).sum()),changed_entries=int((changes!=0).sum()),delta_L2=float(np.linalg.norm(changes.astype(np.float64))),original_sha256=digest(raw.numpy()),written_sha256=digest(model.word.numpy()),step=strength,first5_indices=idx[:5].tolist(),row_index_sha256=digest(idx.astype(np.int64)),paired_bootstrap95=np.quantile(boots,[.025,.975]).tolist(),time_seconds=time.monotonic()-st)
        np.savez(HERE/(name+"-"+arm+".npz"),pred=h,logits=logits)
        dump(path,old)
        print(name,arm,old[arm]["macro_f1"],old[arm]["accuracy"],old[arm]["changes"],round(old[arm]["time_seconds"],1),flush=True)

if __name__=="__main__": main()
