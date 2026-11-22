"""A physical seven-channel lexical accumulator inside the released 16.8M-parameter pair of LSTMs."""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file
from scipy.special import softmax
import torch

from embedding_write import BASE, DONOR, HierModel, NAMES, TO_ID, HERE, ROOT, confusion, digest, dump, file_hash, metric, path_labels

M = 7

def val_ids():
    return json.loads((HERE.parent/"shared-confirmation.json").read_text())["extended_validation_judgment_ids"]

def data(split,limit=None):
    source=ROOT/"references/official"/("train.csv" if split=="fit" else "val.csv")
    if split=="shared":
        ids=val_ids()[:limit]
        cached={}
    else:
        with np.load(ROOT/"cache/sentence-recurrent-inverse"/(split+".npz")) as f:
            assert f["backbone"].item()==file_hash(BASE)
            ids=f["keys"].tolist()
            end=f["ends"]; mat=f["words"]
            cached={k:torch.from_numpy(mat[a:b]) for k,a,b in zip(ids,end,end[1:])}
        if split!="fit":
            ids=[k for k in ids if k not in val_ids()]
        ids=ids[:limit]
    b={k:([],[]) for k in ids}
    with source.open(newline="",encoding="utf-8-sig") as fp:
        for row in csv.DictReader(fp):
            k=row["Index"]; t=row["Text"].strip()
            if k in b and t:
                b[k][0].append(t);b[k][1].append(TO_ID[row["Label"]])
    return [(k,*b[k],cached.get(k)) for k in ids]

def native(model):
    return {"sent_encoder.emb.weight":model.word,
        **{"sent_encoder.lstm."+k:t.detach() for k,t in model.sent.named_parameters()},
        **{"emitter.lstm."+k:t.detach() for k,t in model.doc.named_parameters()},
        "emitter.hidden2tag.weight":torch.from_numpy(model.weight),"emitter.hidden2tag.bias":torch.from_numpy(model.bias),"crf.transitions":torch.from_numpy(model.transitions)}

def importance(model):
    """Use only stored official-train word representations for activation salience."""
    with np.load(ROOT/"cache/sentence-recurrent-inverse/fit.npz") as f: words=f["words"]
    norms=np.sqrt(np.mean(words*words,axis=0))[:100]
    t=native(model)
    fwd=t["sent_encoder.lstm.weight_hh_l0"].numpy()
    docin=np.concatenate([t["emitter.lstm.weight_ih_l0"+d].numpy()[:,:100] for d in ("","_reverse")])
    sent_score=norms*np.sqrt((fwd**2).sum(0)+(docin**2).sum(0))
    drec=t["emitter.lstm.weight_hh_l0"].numpy(); last=model.weight[3:,:100]
    # End task linear readout is shared by every sequence position.
    doc_score=np.linalg.norm(drec,axis=0)/np.linalg.norm(drec)*np.sqrt(100)+3*np.linalg.norm(last,axis=0)/np.linalg.norm(last)*np.sqrt(100)
    return np.argsort(sent_score)[:M],np.argsort(doc_score)[:M]

def wire(model,kind):
    sent,doc=importance(model); a=100; axes=np.arange(93,100)
    tensors=native(model); w=model.word.numpy()
    original=w.copy()
    mat=np.load(HERE/"counts.npz")["counts"].transpose(0,2,1).astype(np.float64)
    C=.65*mat[0]+.2*mat[1]+.15*mat[2]
    freq=C.sum(1); prior=C.sum(0)/C.sum()
    p=(C+20*prior)/(freq[:,None]+20)
    odds=np.clip(np.log(p/prior),-3,3); odds-=odds.mean(1,keepdims=True)
    odds*=np.minimum(1,freq/12)[:,None]; odds[:2]=0
    if kind.startswith("shuf"):
        rng=np.random.default_rng(655)
        for lo,hi in ((0,12),(12,30),(30,100),(100,1000),(1000,10000),(10000,np.inf)):
            ids=np.flatnonzero((freq>=lo)&(freq<hi)&(np.arange(len(freq))>=2))
            odds[ids]=odds[rng.permutation(ids)]
    if kind.startswith("zero"): odds.fill(0)
    if kind.startswith("residual"):
        with np.load(HERE/"residual-dictionary.npz") as f: odds=f["word_scores"]
    weights=freq.clip(0,3000)
    mean=(weights@original)/weights.sum()
    q=original.astype(np.float64)-mean
    cov=(q.T*weights)@q/weights.sum()+np.eye(100)*1.e-7
    e,v=np.linalg.eigh(cov)
    root=(v*np.sqrt(e))@v.T
    invroot=(v*(1/np.sqrt(e)))@v.T
    oldW=np.concatenate([tensors["sent_encoder.lstm.weight_ih_l0"+d].numpy() for d in ("","_reverse")],axis=0)
    _,_,vh=np.linalg.svd(oldW@root,full_matrices=False)
    rotation=vh.T
    for d in ("","_reverse"):
        ih=tensors["sent_encoder.lstm.weight_ih_l0"+d].numpy()
        tensors["sent_encoder.lstm.bias_ih_l0"+d].numpy()[:]+=ih@mean
        ih[:]=ih@root@rotation
        ih[:,axes]=0
    w[:]=(q@invroot@rotation).astype(np.float32)
    w[:,axes]=odds.astype(np.float32)
    # The other native cells cannot receive spurious evidence from the reused channels.
    def block(prefix,cols):
        for suffix in ("","_reverse"):
            tensors[prefix+suffix].numpy()[:,cols]=0
    tensors["sent_encoder.lstm.weight_hh_l0"].numpy()[:,sent]=0
    block("emitter.lstm.weight_ih_l0",sent)
    tensors["emitter.lstm.weight_hh_l0"].numpy()[:,doc]=0
    model.weight[:,doc]=0
    def program_lstm(prefix,channels,source,ii,ff,oo,candidate):
        ih=tensors[prefix+".weight_ih_l0"].numpy()
        hh=tensors[prefix+".weight_hh_l0"].numpy()
        bi=tensors[prefix+".bias_ih_l0"].numpy()
        bh=tensors[prefix+".bias_hh_l0"].numpy()
        for gate,logit in enumerate((ii,ff,0,oo)):
            rows=gate*a+channels
            ih[rows,:]=0; hh[rows,:]=0; bi[rows]=logit; bh[rows]=0
        ih[2*a+channels,source]=candidate
    program_lstm("sent_encoder.lstm",sent,axes,-.6,2.6,4.,.24)
    program_lstm("emitter.lstm",doc,sent,4.,-6.,4.,1.1)
    return {"embedding_axes":axes.tolist(),"sentence_cells":sent.tolist(),"document_cells":doc.tolist(),"dict":kind}

def store_state(split,kind,limit=None):
    path=HERE/(f"native-{kind}-{split}-{limit}.npz")
    if path.exists(): return path
    docs=data(split,limit)
    model=HierModel(DONOR)
    details=wire(model,kind) if kind!="base" else {}
    arr=[]; y=[]; offsets=[0]
    now=time.monotonic()
    for i,(k,text,g,cache) in enumerate(docs):
        words=cache if kind=="base" and cache is not None else model.words(text)
        states=model.contextualize(words).astype(np.float32)
        arr.append(states); y.extend(g); offsets.append(len(y))
        if (i+1)%48==0: print("encode",kind,split,i+1,len(y),round(time.monotonic()-now),flush=True)
    tmp=path.with_suffix(".part")
    with tmp.open("wb") as fp:
        np.savez(fp,states=np.concatenate(arr),ends=offsets,gold=y,keys=[d[0] for d in docs])
    tmp.replace(path)
    print("cached",path.name,len(y),round(time.monotonic()-now),flush=True)
    if details: dump(HERE/("wiring-"+kind+".json"),details)
    return path

def confusion_docs(g,h,ends):
    return np.stack([confusion(g[a:b],h[a:b]) for a,b in zip(ends,ends[1:])])

def decode(z,ends,tr):
    return np.concatenate([path_labels(z[a:b],tr) for a,b in zip(ends,ends[1:])])

def fit_head(X,y,initial,power,reg):
    # The whole head is a one-shot ridge solution with 7 outcomes. No optimizer or backpropagation.
    x=np.column_stack((X,np.ones(len(y))))
    freq=np.bincount(y,minlength=7)/len(y)
    mass=freq[y]**(-power)
    mass/=mass.mean()
    old=x@initial
    target=np.eye(7)[y]
    posterior=softmax(old/2.2,axis=1)
    desired=(target-posterior)*4.4
    gram=(x.T*mass)@x
    degree=reg*gram.trace()/len(gram)
    pen=np.full(x.shape[1],degree);pen[-1]=.001*degree
    delta=np.linalg.solve(gram+np.diag(pen),(x.T*mass)@desired)
    return initial+delta

def score(kind,select_limit=71,fit_limit=240):
    kwargs={"allow_pickle":False}
    train=np.load(store_state("fit",kind,fit_limit),**kwargs)
    dev=np.load(store_state("select",kind,select_limit),**kwargs)
    old=HierModel(DONOR); model=HierModel(DONOR)
    if kind!="base": wire(model,kind)
    prior=np.concatenate((model.weight[3:].T,model.bias[None,3:]),axis=0)
    x=train["states"].astype(float);y=train["gold"]
    z=dev["states"];gt=dev["gold"];ends=dev["ends"]
    opts={}; fits={}
    for p in (0.,.28,.6):
        for reg in (.03,.3):
            A=fit_head(x,y,prior,p,reg)
            for scale in (.5,1.25,2.5):
                h=(prior+scale*(A-prior)).astype(np.float32)
                logits=z@h[:-1]+h[-1]
                pred=decode(logits,ends,model.transitions)
                metric_=metric(confusion(gt,pred))
                tag=f"{p},{reg},{scale}"
                opts[tag]=metric_;fits[tag]=h
    take=max(opts,key=lambda k:(opts[k]["macro_f1"],opts[k]["accuracy"]))
    spec=dict(kind=kind,train_cases=train["keys"].tolist(),select_cases=dev["keys"].tolist(),
              rows_fit=len(y),rows_selection=len(gt),chosen=take,scores=opts,gradient_steps=0,
              sha256_base=file_hash(BASE),sha256_train_csv=json.loads((HERE/"counts.json").read_text())["train_sha256"])
    prefix=f"native-{kind}-{fit_limit}-{select_limit}"
    dump(HERE/(prefix+".json"),spec)
    head=fits[take]
    model.weight[3:]=head[:-1].T;model.bias[3:]=head[-1]
    src=native(old); dest=native(model)
    changed={}; tensor_dict={}
    for k in src:
        a=src[k].numpy();b=dest[k].numpy()
        if np.array_equal(a,b): continue
        d=b-a
        changed[k]=dict(shape=list(a.shape),changed_entries=int(np.count_nonzero(d)),original_hash=digest(a),new_hash=digest(b),delta_L2=float(np.linalg.norm(d.astype(float))))
        tensor_dict[k]=b.copy()
    temp=HERE/(prefix+".tmp")
    save_file(tensor_dict,str(temp),metadata={"parent":file_hash(BASE),"kind":kind,"gradient_steps":"0"})
    path=HERE/(prefix+".safetensors")
    temp.replace(path); spec.update(weights_sha256=file_hash(path),tensors=changed,model_parameters=model.parameters)
    dump(HERE/(prefix+".json"),spec)
    print("head",kind,take,opts[take]["macro_f1"],opts[take]["accuracy"],flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("method",choices=("cache","score"));ap.add_argument("--kind",default="odds")
    ap.add_argument("--split",default="fit",choices=("fit","select","confirm","shared","holdout"));ap.add_argument("--limit",type=int,default=240)
    ap.add_argument("--select-limit",type=int,default=71);arg=ap.parse_args()
    if arg.method=="cache": store_state(arg.split,arg.kind,arg.limit)
    else: score(arg.kind,arg.select_limit,arg.limit)
