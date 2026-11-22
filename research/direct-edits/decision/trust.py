"""Directly cap a previously written role edit's Frobenius norm per target matrix."""
import argparse
import json
import numpy as np
import torch
from safetensors.torch import load_file,save_file
from fetch import ROOT,sha
from weights import INTERNAL,READOUT,arr,metrics,normed,read_rows,savej,source,wtproof
def run(mode,backend,arm,tier):
 cfg=json.loads((ROOT/'trust-plan.json').read_text());wd=ROOT/'weights'/backend/arm/tier
 full_head,g,full_internal=source()
 if mode=='fit':
  result={'plan_sha256':sha(ROOT/'trust-plan.json'),'parents':{},'changes':{}}
  for name,key,old in [('readout',READOUT,full_head),('internal',INTERNAL,full_internal)]:
   parent=wd/('roles-'+name+'.safetensors');child=wd/('trust-'+name+'.safetensors')
   orig=old.float();prior=load_file(str(parent))[key].float();d=prior-orig
   part=orig[:7] if name=='readout' else orig
   scale=min(1.,cfg['limit']*float(torch.linalg.norm(part))/float(torch.linalg.norm(d)))
   new=(orig+scale*d).to(old.dtype)
   save_file({key:new},str(child))
   result['parents'][name]=sha(parent)
   result['changes'][name]={'factor':scale,'sha256':sha(child),'target':key,**wtproof(old,new)}
  savej(wd/'trust-frozen.json',result)
 else:
  x=read_rows(backend,arm,'val',tier);h=x['hidden'];r=x['before_normalization'];u=x['mlp_activation'];y=x['y']
  proof=json.loads((wd/'trust-frozen.json').read_text());result={}
  for which,key,old in [('readout',READOUT,full_head),('internal',INTERNAL,full_internal)]:
   path=wd/('trust-'+which+'.safetensors');assert sha(path)==proof['changes'][which]['sha256']
   new=arr(load_file(str(path))[key]);old=arr(old)
   if which=='readout':z=h@new[:7].T
   else:z=normed(r+u@(new-old).T,arr(g))@arr(full_head)[:7].T
   p=z.argmax(1);native=x['logit'].argmax(1)
   result[which]={'metrics':metrics(y,p),'fixed_vs_native':int(((p==y)&(native!=y)).sum()),'broken_vs_native':int(((p!=y)&(native==y)).sum()),'predictions':p.tolist()}
  savej(wd/'trust-validation.json',result)
if __name__=='__main__':
 q=argparse.ArgumentParser();q.add_argument('mode',choices=['fit','score']);q.add_argument('--backend',default='ov-GPU-f32');q.add_argument('--arm',default='alone');q.add_argument('--tier',default='pilot');a=q.parse_args();run(a.mode,a.backend,a.arm,a.tier)
