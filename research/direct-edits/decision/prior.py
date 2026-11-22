"""A second native readout write to correct the case-coverage pilot's role frequencies."""
import argparse
import json
import numpy as np
import torch
from safetensors.torch import load_file,save_file
from fetch import ROOT,sha
from weights import arr,metrics,read_rows,savej,wtproof
def run(mode,backend,arm,tier):
 where=ROOT/'weights'/backend/arm/tier; prior=json.loads((ROOT/'prior_plan.json').read_text())
 before=where/'roles-readout.safetensors';after=where/'natural-prior-readout.safetensors'
 if mode=='fit':
  train=read_rows(backend,arm,'train',tier);x=train['hidden'].astype(float);y=train['y']
  mu=x.mean(0);direction=mu/(mu@mu)
  fit=np.bincount(y,minlength=7);full=np.asarray(prior['train_counts'])
  shift=prior['coefficient']*np.log((full/full.sum())/(fit/fit.sum()));shift-=shift.mean()
  old=load_file(str(before))['weight'];new=old.clone();new[:7]=(old[:7].float()+torch.from_numpy(shift[:,None]*direction).float()).to(old.dtype)
  assert torch.equal(old[7:],new[7:])
  save_file({'weight':new},str(after))
  savej(where/'natural-prior-frozen.json',{'specification':sha(ROOT/'prior_plan.json'),'parent':sha(before),'patch':sha(after),'logit_direction':shift.tolist(),'mean_approximate_constant':float((x@direction).mean()),'sd_approximate_constant':float((x@direction).std()),'change_from_parent':wtproof(old,new)})
 else:
  spec=json.loads((where/'natural-prior-frozen.json').read_text())
  assert sha(before)==spec['parent'] and sha(after)==spec['patch']
  val=read_rows(backend,arm,'val',tier);h=val['hidden'];y=val['y']
  p=(h@arr(load_file(str(after))['weight'])[:7].T).argmax(1);base=(h@arr(load_file(str(before))['weight'])[:7].T).argmax(1)
  savej(where/'natural-prior-validation.json',{'metrics':metrics(y,p),'fixed_vs_parent':int(((base!=y)&(p==y)).sum()),'broken_vs_parent':int(((base==y)&(p!=y)).sum()),'predictions':p.tolist()})
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['fit','score']);a.add_argument('--backend',default='ov-GPU-f32');a.add_argument('--arm',default='alone');a.add_argument('--tier',default='pilot');q=a.parse_args();run(q.mode,q.backend,q.arm,q.tier)
