"""Analyze three pooling methods with identical direct BF16 writes and data."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file,save_file
from tokenizers import Tokenizer
from cohort import case_rows,LABEL_ID,incumbent_dir
from fetch import ROOT,MODEL,sha
from weights import arr,ridge,source,savej,wtproof
from compare import stats,scores
DIR=ROOT/'readout-study';DIR.mkdir(exist_ok=True)
def raw_mask(tokenizer,text):
 full='\n'+text;e=tokenizer.encode(full,add_special_tokens=False)
 keep=np.array([any(not c.isspace() for c in full[a:b]) for a,b in e.offsets],bool)
 if not keep.any():raise ValueError('nonblank input has no encoded content token')
 return np.asarray(e.ids,np.int32),keep
def banks(split,content=False):
 p=json.loads((ROOT/'packed-full-plan.json').read_text());ids=p['fixed_fit_ids'] if split=='train' else p['fixed_validation_ids']
 data=case_rows(split,set(ids));tok=Tokenizer.from_file(str(MODEL/'tokenizer.json'))
 vectors=[];cues=[];groups=[];labels=[];cases=[];positions=[];context=[]
 ftr=ROOT/'features/ov-GPU-f32/packed-ordered'
 with safe_open(str(MODEL/'model-00001-of-00001.safetensors'),'pt',device='cpu') as f:
  T=f.get_tensor('model.embeddings.tok_embeddings.weight')
  g=arr(f.get_tensor('model.embeddings.norm.weight'))
  cache={}
  for cid in ids:
   dg=json.loads((ftr/'layouts'/(split+'-'+cid+'.json')).read_text());layout=dg['index_to_slot_group_size_and_index']
   with np.load(ftr/(split+'-'+cid+'.npz'),allow_pickle=False) as z:
    assert z['idx'].tolist()==list(range(len(data[cid])))
    cues.extend(z['hidden'])
   if content:
    with np.load(Path(os.environ.get('GLIDE_CONTENT_ROOT',ROOT/'content-hidden'))/(split+'-'+cid+'.npz'),allow_pickle=False) as z:
     assert str(z['input_hash'])==dg['sha256_input_ids_cues_targets'] and z['positions'].tolist()==list(range(len(data[cid])))
     context.extend(z['content_mean'])
   for i,(text,l) in enumerate(data[cid]):
    token,selected=raw_mask(tok,text);token=token[selected]
    missing=[j for j in set(token.tolist()) if j not in cache]
    if missing:
     b=arr(T[missing]);b-=b.mean(1,keepdims=True);b=b/np.sqrt((b*b).mean(1,keepdims=True)+1.e-5)*g
     cache.update(zip(missing,b))
    vectors.append(np.mean([cache[int(j)] for j in token],axis=0))
    groups.append(f'{cid}-{layout[str(i)][2]}');labels.append(LABEL_ID[l]);cases.append(cid);positions.append(i)
 return {'lexical':np.stack(vectors),'cue':np.stack(cues),'groups':np.array(groups),'y':np.array(labels),'cases':np.array(cases),'positions':np.array(positions),'context':np.stack(context) if context else None}
def remove_block(x,groups,mu):
 out=x.copy()
 for k in dict.fromkeys(groups):
  sel=groups==k;n=sel.sum()
  if n>1:out[sel]=x[sel]-(x[sel].sum(0)-x[sel])/(n-1)+mu
 return out
def swapped(x,groups):
 out=x.copy()
 for k in dict.fromkeys(groups):
  ids=np.flatnonzero(groups==k);out[ids]=x[np.roll(ids,1)]
 return out
def transfer(train,val,kind):
 if kind=='cue-minus-block':return [remove_block(a['cue'],a['groups'],train['cue'].mean(0)) for a in (train,val)]
 name={'static-tokens':'lexical','contextual-tokens':'context'}[kind]
 return train[name],val[name]
def solve(train,tw,y):
 W=arr(tw)[:7];raw=train@W.T;z=raw-raw.mean(1,keepdims=True)
 delta=ridge(train,4*(np.eye(7)[y]-1/7)-z,y)
 new=tw.clone();new[:7]=torch.from_numpy(W+delta).to(new.dtype)
 assert torch.equal(new[7:],tw[7:]);return new
def run(context=False):
 plan=json.loads((ROOT/'content-plan.json').read_text());train=banks('train',context);val=banks('val',context)
 t=source()[0];y=val['y'];c=val['cases'];source_info=sha(ROOT/'content-plan.json')
 established={}
 file=ROOT/'weights/ov-GPU-f32/packed-ordered/whole16/predictions.json';cue=np.array(json.loads(file.read_text())['predictions']['roles-readout'])
 inbank={}
 for cid in set(c):inbank[cid]=json.loads((incumbent_dir()/f'{cid}.json').read_text())['predictions']['unchanged']
 inc=np.array([inbank[cid][int(i)] for cid,i in zip(c,val['positions'])]);cm0,base=stats(y,inc,c);cmc,central=stats(y,cue,c)
 rng=np.random.default_rng(760230);bs=rng.integers(0,16,(4000,16));sb=scores(cm0[bs].sum(1));sc=scores(cmc[bs].sum(1))
 for kind in (['contextual-tokens'] if context else ['static-tokens','cue-minus-block']):
  tr,v=transfer(train,val,kind);path=DIR/kind;path.mkdir(exist_ok=True)
  # Fit and write before scoring this candidate on its separate judgments.
  files={}
  for name,labels in [('roles',train['y']),('shuffle',np.random.default_rng(221126).permutation(train['y']))]:
   new=solve(tr,t,labels);f=path/(name+'-readout.safetensors');save_file({'weight':new},str(f));files[name]={'sha256':sha(f),'change':wtproof(t,new)}
  savej(path/'fit.json',{'pre_registration':source_info,'train_rows':len(tr),'fit_cases':list(dict.fromkeys(train['cases'])),'files':files})
  if kind=='cue-minus-block':np.save(path/'train-anchor.npy',train['cue'].mean(0),allow_pickle=False)
  methods={'unchanged':(v,arr(t)[:7])}
  for name in files:
   f=path/(name+'-readout.safetensors');assert sha(f)==files[name]['sha256'];W=arr(load_file(str(f))['weight'])[:7];methods[name]=(v,W)
   if name=='roles':methods['swapped_targets_same_weights']=(swapped(v,val['groups']),W)
  results={}
  for name,(x,w) in methods.items():
   p=(x@w.T).argmax(1);cm,m=stats(y,p,c);bo=scores(cm[bs].sum(1))
   results[name]={'metrics':m,'corrected_vs_cue_head':int(((p==y)&(cue!=y)).sum()),'broken_vs_cue_head':int(((p!=y)&(cue==y)).sum()),
     'paired_ci_vs_cue_head':np.quantile(bo-sc,[.025,.975]).tolist(),'paired_ci_vs_incumbent':np.quantile(bo-sb,[.025,.975]).tolist()}
  savej(path/'validation.json',{'pre_registration':source_info,'human_support':np.bincount(y,minlength=7).tolist(),'fixed_cue_head':central,'incumbent':base,'methods':results})
  print(kind,[(k,round(q['metrics']['macro_f1'],6)) for k,q in results.items()],flush=True)
if __name__=='__main__':
 q=argparse.ArgumentParser();q.add_argument('--content',action='store_true');a=q.parse_args();run(a.content)
