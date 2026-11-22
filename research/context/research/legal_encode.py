"""Encode every target with the named public InLegalBERT donors and their proper actual neighbor text."""
import argparse,gzip,hashlib,json,os,subprocess,time
from pathlib import Path
from urllib.parse import quote
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import BertForSequenceClassification,BertConfig,BertTokenizerFast
ROOT=Path('/content/reviewer2');H=Path(__file__).resolve().parent
MODELS={'past':('InLegalBERT(i-2, i-1, i)model.safetensors','a81f509d250f98b44872e89e10186d5dcc17d37905b69aa1d9a077fb841a62ac'),
        'both':('InLegalBERT(i-1, i, i+1)model.safetensors','19cd593e269b1a57ffa500014d99b2fd9a76d2f76cf315d239a8fda205fc122f')}
REV='44d7925e5ab422ecf04acd5dcbb0b62a6f8d8f53'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def check():
    n=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).strip()
    assert n in {'T4','Tesla T4','NVIDIA T4'} and torch.cuda.get_device_name() in {'T4','Tesla T4','NVIDIA T4'}
    return n
def get(kind):
    name,digest=MODELS[kind];target=H/(kind+'-donor.safetensors')
    if not target.exists():
        u='https://huggingface.co/L-NLProc/LegalSeg_InLegalBERT/resolve/'+REV+'/InLegalBERT/'+quote(name,safe='')
        subprocess.run(['curl','-fLsS','--retry','3',u,'-o',str(target.with_suffix('.part'))],check=True)
        assert sha(target.with_suffix('.part'))==digest;target.with_suffix('.part').replace(target)
    assert sha(target)==digest
    cfg=BertConfig.from_json_file(str(ROOT/'banks/bert-config.json'));cfg.num_labels=7;cfg._attn_implementation='sdpa'
    m=BertForSequenceClassification(cfg);m.load_state_dict(load_file(str(target)),strict=True,assign=True)
    m.eval().requires_grad_(False).half().cuda()
    # The classification head is replayed in FP32, including after materializing the edit.
    w=load_file(str(target));m.classifier.float()
    m.classifier.weight.copy_(w['classifier.weight'].cuda());m.classifier.bias.copy_(w['classifier.bias'].cuda())
    return m
def records(group):
    src=(H/'official.jsonl.gz') if group=='test' else ROOT/'context/cohorts.jsonl.gz'
    with gzip.open(src,'rt') as f:return [r for l in f if (r:=json.loads(l))['group']==group]
def windows(items,sep,cls):
    seq=[]
    for k,ids in enumerate(items):
        if k:seq.append(sep)
        seq.extend(ids)
    out=[];lo=0
    while True:
        hi=min(lo+510,len(seq));out.append((lo,hi,[cls]+seq[lo:hi]+[sep]))
        if hi==len(seq):break
        lo+=410
    cuts=[0]+[(hi+lo2)//2 for (_,hi,_),(lo2,_,_) in zip(out,out[1:])]+[len(seq)]
    return [(v[2],(b-a)/max(len(seq),1)) for v,a,b in zip(out,cuts,cuts[1:])]
def encode(kind,g,m,tok):
    docs=records(g);n=sum(len(d['text']) for d in docs)
    lens=[len(d['text']) for d in docs];off=np.r_[0,np.cumsum(lens)]
    if g=='test':base=np.concatenate([d['base'] for d in docs])
    else:
        with np.load(ROOT/(g+'.npz'),allow_pickle=False) as b:
            np.testing.assert_array_equal(b['case_ids'][:len(docs)],[d['case'] for d in docs]);np.testing.assert_array_equal(off,b['offsets'][:len(off)])
            base=b['reference'][:n]
    data=[]
    for a,d in zip(off,docs):
        ids=tok(d['text'],add_special_tokens=False,return_attention_mask=False,verbose=False)['input_ids'];last=len(ids)-1
        for j in range(len(ids)):
            ix=(j-2,j-1,j) if kind=='past' else (j-1,j,j+1)
            chunks=windows([ids[k] if 0<=k<=last else [] for k in ix],tok.sep_token_id,tok.cls_token_id)
            for v,weight in chunks:data.append((len(v),int(a+j),weight,v))
    data.sort(key=lambda v:v[0],reverse=True)
    hidd=np.zeros((n,m.config.hidden_size),np.float32);coverage=np.zeros(n,np.float32)
    start=time.time();i=0;count=0
    with torch.inference_mode():
        while i<len(data):
            size=max(8,min(160,11264//data[i][0]));part=data[i:i+size];i+=len(part);count+=1
            ix=np.array([v[1] for v in part]);weights=np.array([v[2] for v in part],np.float32)
            width=(part[0][0]+7)//8*8
            x=np.zeros((len(part),width),dtype=np.int64)
            for j,p in enumerate(part):x[j,:p[0]]=p[3]
            xt=torch.as_tensor(x,device='cuda')
            rep=m.bert(xt,attention_mask=(xt!=0)).pooler_output.float().cpu().numpy()
            np.add.at(hidd,ix,rep*weights[:,None]);np.add.at(coverage,ix,weights)
            if count%180==1:print(kind,g,i,len(data),round(time.time()-start),flush=True)
    np.testing.assert_allclose(coverage,1,atol=3e-6)
    w=m.classifier.weight.detach().cpu().numpy();b=m.classifier.bias.detach().cpu().numpy()
    np.savez(H/(kind+'-source.npz'),w=w,b=b)
    dest=H/(kind+'-'+g+'.npz')
    with dest.with_suffix('.part').open('wb') as f:
        np.savez(f,hidden=hidd.astype(np.float16),raw=hidd@w.T+b,gold=np.concatenate([d['gold'] for d in docs]).astype(np.int8),base=base,case_ids=np.array([d['case'] for d in docs]),offsets=off)
    dest.with_suffix('.part').replace(dest)
    rec=dict(rows=int(n),windows=len(data),max_tokens=int(max(x[0] for x in data)),additional_windows=len(data)-int(n),seconds=time.time()-start,sha256=sha(dest),model_sha256=MODELS[kind][1],device=check())
    (H/(kind+'-'+g+'.json')).write_text(json.dumps(rec,indent=2))
    print('finished',kind,g,rec,flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('groups',nargs='*');p=parser.parse_args();check()
    tok=BertTokenizerFast.from_pretrained(H/'tokenizer',local_files_only=True)
    for kind in MODELS:
        model=get(kind)
        for g in p.groups or ('fit','select','confirm','fresh'):
            encode(kind,g,model,tok)
        del model;torch.cuda.empty_cache()
