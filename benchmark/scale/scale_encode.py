"""Extract compact score/semantic features in judgment-disjoint training shards on exactly one T4."""
import gzip,json,sys,time
from pathlib import Path
import numpy as np
import torch
from safetensors.numpy import load_file
from transformers import BertTokenizerFast
from legal_encode import get,check,MODELS,windows
import legal_solve as legal
H=Path(__file__).resolve().parent;L=legal.H
def compact(kind,h,raw):
    p=legal.get(kind+'-pca');w=legal.get(kind+'-edited')
    # Match the exact trained 192-judgment model's rounding before standardisation.
    x=h.astype(np.float16).astype(np.float32);edited=x@w['weight'].T+w['bias']
    pcs=np.clip((x-p['mu'])@p['proj'],-5,5)
    return np.column_stack((raw,edited,pcs)).astype(np.float32)
def fixed(g):
    out=[]
    for k in MODELS:
        d=legal.read(k,g);out.append(compact(k,d.hidden,d.raw))
    return np.column_stack(out),d.offsets,d.gold,d.case_ids
def panels():
    for g in ('fit','select','confirm','fresh'):
        x,off,y,case=fixed(g);np.savez(H/(g+'-panel.npz'),x=x,offsets=off,y=y,case=case)
@torch.inference_mode()
def one(kind,model,tok,token_docs,offsets):
    packs=[]
    for st,ids in zip(offsets,token_docs):
        end=len(ids)
        for j in range(end):
            ix=(j-2,j-1,j) if kind=='past' else (j-1,j,j+1)
            for v,weight in windows([ids[k] if 0<=k<end else [] for k in ix],tok.sep_token_id,tok.cls_token_id):
                packs.append((len(v),int(st+j),weight,v))
    packs.sort(key=lambda v:v[0],reverse=True)
    hidden=np.zeros((offsets[-1],768),np.float32);cov=np.zeros(len(hidden),np.float32)
    i=0
    while i<len(packs):
        size=max(8,min(160,11264//packs[i][0]));part=packs[i:i+size];i+=len(part)
        x=np.zeros((len(part),(part[0][0]+7)//8*8),dtype=np.int64)
        for j,v in enumerate(part):x[j,:v[0]]=v[3]
        xt=torch.as_tensor(x,device='cuda');h=model.bert(xt,attention_mask=xt!=0).pooler_output.float().cpu().numpy()
        where=np.array([v[1] for v in part]);w=np.array([v[2] for v in part],np.float32)
        np.add.at(hidden,where,h*w[:,None]);np.add.at(cov,where,w)
    np.testing.assert_allclose(cov,1,atol=3e-6)
    w=model.classifier.weight.cpu().numpy();b=model.classifier.bias.cpu().numpy()
    return compact(kind,hidden,hidden@w.T+b)
def generate(limit,callback=None):
    import itertools
    check();models={k:get(k) for k in MODELS};tok=BertTokenizerFast.from_pretrained(L/'tokenizer',local_files_only=True)
    with gzip.open(H/'fit.jsonl.gz','rt') as stream:
        for i,group in itertools.groupby((json.loads(l) for l in stream),key=lambda z:z['shard']):
            if i>=limit:break
            dest=H/(f'train-{i:02d}.npz')
            if dest.exists():
                if callback:callback(i,dest)
                continue
            docs=list(group);tm=time.time()
            token_docs=[tok(d['text'],add_special_tokens=False,return_attention_mask=False,verbose=False)['input_ids'] for d in docs]
            off=np.r_[0,np.cumsum([len(x) for x in token_docs])]
            x=np.column_stack([one(k,m,tok,token_docs,off) for k,m in models.items()])
            with dest.with_suffix('.part').open('wb') as f:
                np.savez(f,x=x,offsets=off,y=np.concatenate([d['gold'] for d in docs]).astype(np.int8),case=np.array([d['case'] for d in docs]))
            dest.with_suffix('.part').replace(dest)
            print('scale shard',i,len(docs),len(x),round(time.time()-tm,1),flush=True)
            if callback:callback(i,dest)
