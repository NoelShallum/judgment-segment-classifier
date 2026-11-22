"""Run both matched arms at a single precision and verify physically written tensors."""
import argparse
import json
import time
import numpy as np
import torch
from safetensors.torch import load_file
from cohort import case_rows
from extract import modules,extract,states_for,encode
from fetch import ROOT
from weights import READOUT,INTERNAL,freeze,score,savej,source,normed,arr
def verify_physical(backend,arm,tier,ready):
 model,tok,temp=ready
 wd=ROOT/'weights'/backend/arm/tier
 plan=json.loads((ROOT/'plan.json').read_text())
 case=plan['val_judgment_ids'][1]
 ix=plan['validation_indices'][case][0]
 st=states_for(case_rows('val',{case})[case],[ix],arm)[0]
 base,native,n,r,u=encode(model,tok,temp,st)
 original,g0,internal=source();W,g,I=arr(original)[:7],arr(g0),arr(internal)
 details={'case':case,'position':ix,'native_label':int(native.argmax()),'tokens':n,'checks':{}}
 with torch.inference_mode():
  for n in ('roles-readout','roles-internal'):
   f=wd/(n+'.safetensors')
   patch=load_file(str(f))
   if READOUT in patch:
    tensor=model.readout.weight
    before=tensor.detach().clone()
    tensor.copy_(patch[READOUT]);expected=base@arr(patch[READOUT])[:7].T
   else:
    tensor=model.model.layers[-1].mlp.Wo.weight
    before=tensor.detach().clone()
    tensor.copy_(patch[INTERNAL])
    expected=(normed((r+u@(arr(patch[INTERNAL])-I).T)[None,:],g)@W.T)[0]
   try:
    _h,z,*_rest=encode(model,tok,temp,st)
    difference=float(np.max(np.abs(z-expected)))
    details['checks'][n]={'max_abs_logit_error':difference,'same_argmax':bool(z.argmax()==expected.argmax()),'edited_label':int(z.argmax())}
    if difference>.05 or z.argmax()!=expected.argmax():raise ValueError(details)
   finally: tensor.copy_(before)
 savej(wd/'runtime-proof.json',details)
 print('physical check',arm,details['checks'],flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--backend',default='fp32');a.add_argument('--tier',default='pilot');a.add_argument('arms',nargs='*');args=a.parse_args()
 t=time.monotonic();ready=modules(args.backend);print('model-ready',args.backend,round(time.monotonic()-t,1),flush=True)
 for arm in args.arms or ['alone','local']:
  extract(args.backend,arm,['train-'+args.tier],ready)
  freeze(args.backend,arm,args.tier)
  extract(args.backend,arm,['val-'+args.tier],ready)
  score(args.backend,arm,args.tier)
  verify_physical(args.backend,arm,args.tier,ready)
