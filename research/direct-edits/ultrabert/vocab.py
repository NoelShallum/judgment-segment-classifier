"""Encode supervised legal token statistics into the actual original UltraBERT input embedding matrix."""
import argparse
import hashlib
import json
import math
import os

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file
from tokenizers import Tokenizer
import torch
from torch import nn

from prepare import ROOT, LABELS, SALT, SPLIT, docs, hash_int, save
from study import metrics, compare, summary, h
from extract import groups
from joint import decode, position, transition

V=49152
SRC="ultrabert.embed_tokens.weight"
SIGN_SEED=22112026
PERM=np.random.default_rng(SIGN_SEED).permutation(V)


def train():
    path=ROOT/"vocab-counts.npz";receipt=ROOT/"vocab-counts.json"
    if path.exists() and receipt.exists():return
    tok=Tokenizer.from_file(str(ROOT/"model/tokenizer.json"))
    matrix=np.zeros((V,7),np.int64);n=np.zeros(7,np.int64)
    pp=np.zeros((24,7),np.int64);ids=[]
    for cid,rows in docs("train.csv"):
        if not rows or hash_int("ultra-vocab-public-v1|"+cid)%8!=0:continue
        whole="\n".join(" ".join(r["text"].casefold().split()) for r in rows)
        digest=hashlib.sha256(whole.encode()).hexdigest()
        if hash_int(SALT+digest)%100>=76:continue
        ids.append(cid)
        toks=tok.encode_batch([r["text"] for r in rows],add_special_tokens=False)
        for i,(r,t) in enumerate(zip(rows,toks)):
            # Sentence document frequency prevents a repeated token dominating sufficient statistics.
            lab=r["y"];n[lab]+=1;pp[min(23,24*i//len(rows)),lab]+=1
            matrix[np.unique(np.asarray(t.ids,np.int32)),lab]+=1
        if len(ids)%100==0:print("lexicon fit cases",len(ids),"rows",n.sum(),flush=True)
    np.savez_compressed(path,token_document_frequency=matrix,class_counts=n,position=pp,
                        log_transition=transition(set(ids)),judgment_ids=np.asarray(ids))
    save(receipt,{"source":"official/train.csv","source_sha256":"60028fc085e3cb018e8d49a8879ba796dd03a0838367891f3312f9e576921794",
         "fit_ids":ids,"fit_rows":int(n.sum()),"role_counts":n.tolist(),"vocabulary":V,
         "case_rule":"SHA256(ultra-vocab-public-v1|case ID) modulo 8 zero, then same official train case hash fit/edit fold <76. Selection and confirmation fold cases excluded.",
         "counts_sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    print("count finished",len(ids),"rows",n.sum(),flush=True)


def raw_data(items, tok):
    out={k:[] for k in ["row","case","ix","ncase","start","end","y","ids"]}
    for item in items:
        enc=tok.encode_batch([r["text"] for r in item["rows"]],add_special_tokens=False)
        for row, t in zip(item["rows"],enc):
            for key in ["row","ix","start","end","y"]:out[key].append(row[key])
            out["case"].append(item["id"]);out["ncase"].append(item["ncase"])
            out["ids"].append(np.unique(t.ids))
    for k in out:
        if k!="ids":out[k]=np.asarray(out[k])
    return out


def score_tokens(d, mat):
    return np.stack([mat[t].sum(0)/math.sqrt(len(t)) for t in d['ids']])


def priors(d, pp):
    bucket=np.minimum(23,24*d["ix"]//d["ncase"])
    return np.log((pp[bucket]+.5)/(pp[bucket].sum(1,keepdims=True)+3.5))


def fit_coeffs(emb,target,df):
    weights=np.sqrt(1.+df/20)
    # Ridge approximation to the same full training dictionary through the unmodified donor.
    # Solve in chunks for lower peak memory on CPU.
    n=emb.shape[1]
    A=np.zeros((n,n));rhs=np.zeros((n,7))
    for i in range(0,V,2048):
        e=emb[i:i+2048].astype(float);ww=weights[i:i+2048,None]
        A+=e.T@(ww*e);rhs+=e.T@(ww*target[i:i+2048])
    A.flat[::n+1]+=.1*A.trace()/n
    return np.linalg.solve(A,rhs)


def target_logs(df,prior,shrink):
    a=shrink
    # Conditional presence probability, centered against the other classes; Laplace regularization
    # toward global prevalence makes unseen token rows contribute exactly zero.
    marginal=(df.sum(1)+.5)/(prior.sum()+1.)
    likelihood=(df+a*marginal[:,None])/(prior[None,:]+a)
    log=np.log(np.maximum(likelihood,1e-14))-np.log(marginal[:,None])
    log-=log.mean(1,keepdims=True)
    log[df.sum(1)==0]=0
    return np.clip(log,-3.,3.)


def solve(stage):
    train()
    path=ROOT/"vocab-counts.npz"
    with np.load(path) as data:
        df=data['token_document_frequency'].astype(float)
        prior=data['class_counts'];pp=data['position'];tx=data['log_transition']
    with safe_open(str(ROOT/"model/model.safetensors"),framework="numpy") as f:emb=f.get_tensor(SRC)
    tok=Tokenizer.from_file(str(ROOT/"model/tokenizer.json"))
    gs=groups(stage)
    ds={n:raw_data(gs[n],tok) for n in ['select','confirm']}
    se=ds['select']
    records=[];candidates={}
    for shrink in [20.,500.,5000.]:
        target=target_logs(df,prior,shrink)
        w=fit_coeffs(emb,target,df.sum(1))
        zero=emb@w
        candidates[shrink]=(target,w,zero)
        sel_zero=score_tokens(se,zero);sel_full=score_tokens(se,target)
        pos=priors(se,pp)
        for gain in [0.,.25,.75,1.]:
            logits=sel_zero+gain*(sel_full-sel_zero)
            for cpos in [0.,.3,1.]:
                for tscale in [0.,.2,.7]:
                    q=decode(se,logits+cpos*pos,tx,tscale)
                    ff=metrics(se['y'],q)['macro_f1_7']
                    records.append([ff,shrink,gain,cpos,tscale])
    # Freeze text and document-structured separately. Baseline reoptimizes the exact same information.
    result={"stage":stage,"role_order":LABELS,"parameter":SRC,"dimensions":list(emb.shape),
            "counts_receipt":"vocab-counts.json","normalization":"sum distinct native token vectors, divided by square root of number of distinct tokens",
            "inference_path":"input embedding pooled directly into linear seven-class readout. The 24 contextual layers are bypassed in this lexical ablation.",
            "fit_statistics":"same training-label token-frequency dictionary for donor baseline, edit, and permutation. Native donor zero projects the dictionary into original embedding span; direct lexical edit writes the residual directions.",
            "all_selection_grid":records,"experiments":{}}
    root=ROOT/f"vocab-{stage}";root.mkdir(exist_ok=True)
    for mode in ['text','structure']:
        select=[v for v in records if ((v[3]==0 and v[4]==0) if mode=='text' else True)]
        best=max([v for v in select if v[2]>0],key=lambda i:i[0])
        baseopt=max([v for v in select if v[2]==0],key=lambda i:i[0])
        _,s,gain,ps,ts=best
        target,w,zero=candidates[s]
        delta_role=gain*(target-zero)
        lift=np.linalg.pinv(w,rcond=1e-6)
        effective=np.empty((V,7),float)
        modified_sha=hashlib.sha256()
        original_sha=hashlib.sha256()
        norm2=est2=0.
        count=0
        # The dense pretrained table is written into a real torch.nn.Embedding, verified and reproduced from two portable low-rank tensors.
        # Only one live full dense copy. The persisted artifact is the reversible low-rank patch.
        module=nn.Embedding(V,768)
        with torch.no_grad():
            module.weight.copy_(torch.from_numpy(emb))
            for i in range(0,V,1024):
                orig=emb[i:i+1024]
                new=(orig+(delta_role[i:i+1024]@lift).astype(np.float32)).astype(np.float32)
                module.weight[i:i+1024].copy_(torch.from_numpy(new))
                physical=module.weight[i:i+1024].detach().numpy()
                effective[i:i+1024]=physical@w
                diff=physical.astype(float)-orig
                norm2+=np.einsum('ij,ij->',diff,diff);est2+=np.einsum('ij,ij->',orig.astype(float),orig.astype(float))
                modified_sha.update(physical.tobytes());original_sha.update(orig.tobytes())
                count+=np.count_nonzero(diff)
        err=float(np.max(abs(effective-(zero+delta_role))))
        blob={"vocab.role_residual":np.asarray(delta_role,dtype=np.float32),"role.to_hidden":np.asarray(lift,dtype=np.float32),
              "head.weight":np.ascontiguousarray(w.T,dtype=np.float32),"token_count_position_logp":np.asarray(np.log((pp+.5)/(pp.sum(1,keepdims=True)+3.5)),dtype=np.float32),
              "transition.logp":np.asarray(tx,dtype=np.float32)}
        # FP64 factors preserve byte-exact FP32 round-trip. Runtime storage remains dominated by donor embeddings.
        blob["vocab.role_residual"]=np.ascontiguousarray(delta_role,dtype=np.float64)
        blob["role.to_hidden"]=np.ascontiguousarray(lift,dtype=np.float64)
        save_file(blob,str(root/f"{mode}-patch.safetensors"),metadata={"source_tensor":SRC,"materialize":"source + float32(vocab.role_residual @ role.to_hidden), fp32 add; apply blockwise"})
        singular=np.linalg.svd(np.linalg.qr(delta_role,mode='r')[...]@np.linalg.qr(lift.T,mode='r').T,compute_uv=False)
        new={"best_selection":best,"independent_baseline_selection":baseopt,
             "config_order":["select seven-class macro-F1","pseudocount","gain","position weight","transition weight"],
             "tensor":{"name":SRC,"shape":list(emb.shape),"dtype":"float32","source_sha256":original_sha.hexdigest(),"materialized_sha256":modified_sha.hexdigest(),
                       "changed_entries":int(count),"frobenius_delta":math.sqrt(norm2),"relative_frobenius":math.sqrt(norm2/est2),
                       "rank_at_1e-6_relative":int((singular>singular[0]*1e-6).sum()),"max_abs_physical_logit_error":err},
             "results":{}}
        for tag,d in ds.items():
            initial=score_tokens(d,zero);pr=priors(d,pp)
            logits=score_tokens(d,effective)+ps*pr
            p=decode(d,logits,tx,ts)
            b=decode(d,initial+ps*pr,tx,ts)
            bopt=decode(d,score_tokens(d,candidates[baseopt[1]][2])+baseopt[3]*pr,tx,baseopt[4])
            shuffled=decode(d,initial+score_tokens(d,delta_role[PERM])+ps*pr,tx,ts)
            z={k:metrics(d['y'],v) for k,v in [('primary',p),('matched_zero',b),('optimized_zero',bopt),('shuffle',shuffled)]}
            z['vs_optimized_zero']=compare(d['y'],bopt,p)
            new['results'][tag]=z
            np.savez_compressed(root/f"{mode}-{tag}.npz",case=d['case'],source_row=d['row'],ix=d['ix'],start=d['start'],end=d['end'],gold=d['y'],
                                primary=p,optimized_zero=bopt,matched_zero=b,shuffled=shuffled,logits=logits.astype(np.float32))
            print(mode,tag,[(k,round(q['macro_f1_7'],5)) for k,q in z.items() if 'macro_f1_7' in q],flush=True)
        # Native nn.Embedding gather and pooling end to end on a sentence.
        ids=torch.tensor(ds['confirm']['ids'][0]);wh=torch.from_numpy(w.astype(np.float32))
        with torch.no_grad():actual=module(ids).sum(0)@wh/math.sqrt(len(ids))
        wanted=effective[ids.numpy()].sum(0)/math.sqrt(len(ids))
        new['actual_native_lookup']={"source_row":int(ds['confirm']['row'][0]),"length":len(ids),"same_argmax":bool(int(actual.argmax())==int(wanted.argmax())),"max_abs":float(np.max(abs(actual.numpy()-wanted)))}
        result['experiments'][mode]=new;save(root/'report.json',result)
        del module

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument("op",choices=["counts","solve"]);p.add_argument("--stage",default="screen");o=p.parse_args()
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','1')));torch.set_num_interop_threads(1)
    if o.op=='counts':train()
    else:solve(o.stage)
