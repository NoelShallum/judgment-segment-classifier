"""Compile previous-role-conditioned lexical residuals into eight original embedding-table edits."""
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np
import scipy.linalg as la
from safetensors import safe_open
from safetensors.numpy import save_file,load_file
from tokenizers import Tokenizer
import torch
from prepare import ROOT,LABELS,docs,save
from extract import groups
from study import metrics,compare
from joint import decode
from vocab import SRC,V,PERM,raw_data,priors,fit_coeffs,target_logs
from sentence_vocab import from_ids
from uncertainty import interval

def accumulate(tok):
    p=ROOT/'conditional-vocab-counts.npz'
    if p.exists():return
    train=set(json.loads((ROOT/'vocab-counts.json').read_text())['fit_ids'])
    sums=np.zeros((8,V,7),np.int32);freq=np.zeros((8,7),np.int32);n=0
    for cid,rs in docs('train.csv'):
        if cid not in train:continue
        enc=tok.encode_batch([r["text"] for r in rs],add_special_tokens=False)
        prev=7
        for r,t in zip(rs,enc):
            y=r['y'];freq[prev,y]+=1
            sums[prev,np.unique(t.ids),y]+=1;prev=y
        n+=1
        if n%150==0:print('conditional count',n,flush=True)
    assert n==len(train)
    np.savez_compressed(p,counts=sums,role_counts=freq)


def conditional(df,f,prev_df,prev_f,beta):
    # Per role: smooth token presence toward a role-specific pooled probability, then subtract its log.
    marginal=(df.sum(1)+.5)/(f.sum()+1.)
    marginal=(df+20*marginal[:,None])/(f[None,:]+20.)
    log=np.log(np.maximum((prev_df+beta*marginal[None,:,:])/(prev_f[:,None,:]+beta),1e-14)/marginal[None,:,:])
    log-=log.mean(2,keepdims=True)
    return np.clip(log,-1.5,1.5)


def sequence(d,base,bonus,tx,pc,ts,eta,pos):
    n=len(base)
    pred=np.zeros(n,int)
    # All input rows are already in case/sentence order. At inference no previous gold class is supplied.
    ends=np.r_[0,np.flatnonzero(d['case'][1:]!=d['case'][:-1])+1,n]
    value=base+pc*pos
    for a,b in zip(ends,ends[1:]):
        scores=value[a]+eta*bonus[a,7]
        back=np.empty((max(0,b-a-1),7),np.int8)
        for i in range(a+1,b):
            states=scores[:,None]+ts*tx+eta*bonus[i,:7]
            back[i-a-1]=np.argmax(states,0)
            scores=np.max(states,0)+value[i]
        cur=int(scores.argmax());pred[b-1]=cur
        for i in range(b-2,a-1,-1):
            cur=int(back[i-a,cur]);pred[i]=cur
    return pred


def make_view(ds,matrix):
    x=from_ids(ds['ids']);return np.stack([x@tab for tab in matrix],axis=1)


def main():
    tok=Tokenizer.from_file(str(ROOT/'model/tokenizer.json'));accumulate(tok)
    root=ROOT/'conditional-vocab';root.mkdir(exist_ok=True)
    with np.load(ROOT/'vocab-counts.npz') as rt:df=rt['token_document_frequency'].astype(float);f=rt['class_counts'];pp=rt['position'];tx=rt['log_transition']
    with np.load(ROOT/'conditional-vocab-counts.npz') as z:rdf=z['counts'];rf=z['role_counts']
    np.testing.assert_array_equal(rf.sum(0),f);np.testing.assert_array_equal(rdf.sum(0),df)
    source=load_file(str(ROOT/'vocab-screen/compiled-token-scores.safetensors'))['structure.token_to_class']
    pools=json.loads((ROOT/'pools.json').read_text())['pools']
    bank={s:pools[s]['mixed'][:12] for s in ['select','confirm']}
    ds={n:raw_data(v,tok) for n,v in bank.items()}
    selected=ds['select'];x=from_ids(selected['ids'])
    base=x@source;pos=priors(selected,pp);grid=[];tables={}
    for beta in [100.,1000.]:
        tables[beta]=conditional(df,f,rdf,rf,beta)
        bonus=make_view(selected,tables[beta])
        for eta in [0.,.4,1.2]:
            for pc in [.3,1.]:
                for ts in [.2,.7]:
                    pred=sequence(selected,base,bonus,tx,pc,ts,eta,pos)
                    grid.append([metrics(selected['y'],pred)['macro_f1_7'],beta,eta,pc,ts])
    best=max([g for g in grid if g[2]>0],key=lambda g:g[0])
    zero=max([g for g in grid if g[2]==0],key=lambda g:g[0])
    pr=ROOT/'model/revision-receipt.json'
    freeze={'fit_cases':495,'fit_rows':int(f.sum()),'train_select_ids':[g['id'] for g in bank['select']],'train_confirmation_ids':[g['id'] for g in bank['confirm']],
       'grid':grid,'best_edit':best,'best_zero':zero,'configuration_order':['selection seven-class macro-F1','conditional smoothing','conditional weight','position weight','transition weight'],
       'validation_only_for_replay':True,'inference':'Viterbi carries every possible previous predicted class. Start is an eighth state. No ground-truth previous role enters inference.'}
    fp=root/'freeze.json'
    if not fp.exists():save(fp,freeze)
    assert json.loads(fp.read_text())==freeze
    _,beta,eta,pc,ts=best;_,_,_,pc0,ts0=zero
    # Install conditional residuals on the physical embedding table for each state. The deployed readout is the seven-column compilation.
    imp=json.loads((ROOT/'vocab-screen/report.json').read_text())['experiments']['structure']['best_selection']
    _,sr,basegain,_,_=imp
    with safe_open(str(ROOT/'model/model.safetensors'),framework='numpy') as ck:emb=ck.get_tensor(SRC)
    W=fit_coeffs(emb,target_logs(df,f,sr),df.sum(1));lift=la.pinv(W,rtol=1e-6,check_finite=False)
    parent=load_file(str(ROOT/'vocab-screen/structure-patch.safetensors'));pkey=parent['vocab.role_residual'];plift=parent['role.to_hidden']
    material=np.empty_like(tables[beta],dtype=np.float32);receipts=[]
    for prev in range(8):
        h=hashlib.sha256();orig_s=hashlib.sha256();d2=e2=0.;changed=0
        raw=eta*tables[beta][prev]
        for i in range(0,V,1024):
            parent_value=(emb[i:i+1024]+(pkey[i:i+1024]@plift).astype(np.float32)).astype(np.float32)
            edited=(parent_value+(raw[i:i+1024]@lift).astype(np.float32)).astype(np.float32)
            h.update(edited.tobytes());orig_s.update(parent_value.tobytes())
            delta=edited.astype(float)-parent_value
            material[prev,i:i+1024]=delta@W
            d2+=np.sum(delta*delta);e2+=np.sum(parent_value.astype(float)**2);changed+=np.count_nonzero(delta)
        singular=la.svdvals(la.qr(raw,mode='economic')[1]@la.qr(lift.T,mode='economic')[1].T)
        receipts.append({'previous_role':"START" if prev==7 else LABELS[prev],'source_tensor':SRC,'shape':[49152,768],
                         'source_sha256':orig_s.hexdigest(),'materialized_sha256':h.hexdigest(),'changed_entries':int(changed),
                         'relative_frobenius':math.sqrt(d2/e2),'rank':int((singular>singular[0]*1e-6).sum())})
    save_file({'previous_role.compiled_residual':np.ascontiguousarray(material),'word.compiled_logits':source,"transition.logp":tx,
               "position.logp":np.log((pp+.5)/(pp.sum(1,keepdims=True)+3.5))},str(root/'conditional-compiled.safetensors'),
              metadata={'source_tensor':SRC,'readout':'word.mean + one of 8 typed residual tables; dynamic program enumerates previous role'})
    s16=groups('common')['common'];ds['shared16']=raw_data(s16,tok)
    with np.load(ROOT/'vocab-common16-predictions.npz') as z:old=z['incumbent'];np.testing.assert_array_equal(ds['shared16']['row'],z['source_csv_row'])
    report={'method':'Eight physical embedding-table edits conditioned on previous candidate role, including START; source native table with its unconditioned legal edit.',
            'freeze_sha256':hashlib.sha256(fp.read_bytes()).hexdigest(),'source_float32_tensor_receipts':receipts,
            'primary_selection_prefers_no_edit':zero[0]>=best[0],'results':{}}
    for n,d in ds.items():
        bean=from_ids(d['ids'])@source;pos=priors(d,pp)
        bonus=make_view(d,material);rbonus=make_view(d,material[:,PERM,:])
        ep=sequence(d,bean,bonus,tx,pc,ts,1.,pos);z=sequence(d,bean,bonus,tx,pc,ts,0.,pos)
        opt=sequence(d,bean,bonus,tx,pc0,ts0,0.,pos);shuffle=sequence(d,bean,rbonus,tx,pc,ts,1.,pos)
        met={k:metrics(d['y'],i) for k,i in [('edited',ep),('matched_zero',z),('reoptimized_zero',opt),('equal_norm_shuffle',shuffle)]}
        met['paired_delta_vs_reoptimized_zero']=interval(d['case'],d['y'],opt,ep)
        if n=='shared16':met['incumbent']=metrics(d['y'],old);met['delta_vs_incumbent']=interval(d['case'],d['y'],old,ep)
        np.savez_compressed(root/f'{n}-predictions.npz',case=d['case'],row=d['row'],ix=d['ix'],start=d['start'],end=d['end'],gold=d['y'],edited=ep,matched_zero=z,optimized_zero=opt,shuffle=shuffle)
        report['results'][n]=met
        print(n,len(np.unique(d['case'])),len(ep),{k:round(met[k]['macro_f1_7'],5) for k in ['edited','matched_zero','reoptimized_zero','equal_norm_shuffle']},flush=True)
    save(root/'report.json',report)
if __name__=='__main__':main()
