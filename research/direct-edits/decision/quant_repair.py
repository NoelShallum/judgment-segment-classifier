"""Fit native decision tensors from unlabeled FP32/INT8 paired activations."""
import argparse
import json
import numpy as np
import torch
from safetensors.torch import load_file,save_file
from fetch import ROOT,sha
from weights import arr,metrics,normed,source,savej,wtproof
from compare import stats,scores
STATES=('unchanged','same_written_readout','same_written_internal')
CHOICES=('offset',.02,.1,.5,2.)
def unseen(split,arm):
 p=json.loads((ROOT/'plan.json').read_text())
 cases=p['fit_judgment_ids'][:3] if split=='train' else p['val_judgment_ids']
 result={}
 for name in ('ov-GPU-f32','int8'):
  parts={k:[] for k in ('hidden','before_normalization','mlp_activation','case')}
  for c in cases:
   with np.load(ROOT/'features'/name/arm/(split+'-'+c+'.npz'),allow_pickle=False) as f:
    ids=f['idx'].tolist()
    lookup={k:j for j,k in enumerate(ids)}
    keep=range(p['fit_lengths'][c]) if split=='train' else p['validation_indices'][c][:4]
    sel=[lookup[i] for i in keep]
    for k in ('hidden','before_normalization','mlp_activation'):parts[k].append(f[k][sel])
    parts['case'].append([c]*len(sel))
  result[name]={k:np.concatenate(v) for k,v in parts.items()}
 assert np.array_equal(result['int8']['case'],result['ov-GPU-f32']['case'])
 return result
def bases(data,arm):
 native,g,old=source();g=arr(g);old=arr(old);path=ROOT/'weights/ov-GPU-f32'/arm/'pilot';readout=load_file(str(path/'roles-readout.safetensors'))['weight']
 di=arr(load_file(str(path/'roles-internal.safetensors'))['model.layers.27.mlp.Wo.weight'])-old
 rows={}
 for name,x in data.items():
  rows[name]={'unchanged':(x['hidden'],native),'same_written_readout':(x['hidden'],readout),
              'same_written_internal':(normed(x['before_normalization']+x['mlp_activation']@di.T,g),native)}
 return rows
def center(z):return z-z.mean(1,keepdims=True)
def corrected(x,w,teacher,rad):
 x=x.astype(float);w=w.astype(float);r=center(teacher-x@w.T)
 mean=x.mean(0);d=mean/(mean@mean);bias=r.mean(0)
 biasw=np.outer(bias,d)
 if rad=='offset':return w+biasw
 residual=r-x@biasw.T;xm=x-mean
 scale=np.mean((xm*xm).sum(1))
 gram=xm@xm.T;coef=np.linalg.solve(gram+np.eye(len(xm))*float(rad)*scale,residual)
 wd=(xm.T@coef).T
 # Centering the keys is represented exactly up to the positive projective scale of each sample.
 return w+biasw+wd-np.outer(wd@mean,d)
def frozen(arm):
 data=unseen('train',arm);by=bases(data,arm);rows=data['int8']['case'];report={'no_gold_labels_accessed':True,'fit_cases':list(dict.fromkeys(rows)),'rows':len(rows),'plan_sha256':sha(ROOT/'quant-repair-plan.json'),'states':{}}
 wd=ROOT/'weights/int8'/arm/'pilot/quant-repair';wd.mkdir(parents=True,exist_ok=True)
 for state in STATES:
  f,sw=by['ov-GPU-f32'][state];x,tw=by['int8'][state];w=arr(sw)[:7];raw=center(f@w.T)
  result={}
  for category,target in [('paired',raw),('shuffled',np.random.default_rng(221126).permutation(raw))]:
   grid=[]
   for v in CHOICES:
    q=np.zeros_like(target)
    for c in set(rows):
     keep=rows!=c
     par=corrected(x[keep],w,target[keep],v)
     q[~keep]=center(x[~keep]@par.T)
    grid.append(dict(choice=v,fidelity=float((q.argmax(1)==target.argmax(1)).mean()),centered_mae=float(abs(q-target).mean())))
   best=max(grid,key=lambda g:(g['fidelity'],-g['centered_mae'],100 if g['choice']=='offset' else g['choice']))
   new=tw.clone();new[:7]=torch.tensor(corrected(x,w,target,best['choice'])).to(tw.dtype)
   fn=wd/(state+'-'+category+'.safetensors');save_file({'weight':new},str(fn))
   result[category]={'cross_validated_selection':best,'grid':grid,'weights_sha256':sha(fn),'physical_edit':wtproof(tw,new)}
  report['states'][state]=result
 savej(wd/'frozen.json',report);print('repair frozen',arm,flush=True)
def evaluate(arm):
 data=unseen('val',arm);by=bases(data,arm);wd=ROOT/'weights/int8'/arm/'pilot/quant-repair';rep=json.loads((wd/'frozen.json').read_text())
 pos=json.loads((ROOT/'weights/int8'/arm/'pilot/predictions.json').read_text());y=np.array(pos['gold']);c=data['int8']['case'];assert np.array_equal(c,pos['cases'])
 out={'validation_rows':len(y),'judgments':len(set(c)),'frozen_sha256':sha(wd/'frozen.json'),'states':{}}
 for state in STATES:
  f,sw=by['ov-GPU-f32'][state];x,tw=by['int8'][state];truth=center(f@arr(sw)[:7].T);p=truth.argmax(1);raw=center(x@arr(tw)[:7].T)
  def measure(z):
   choice=z.argmax(1)
   return {'agreement_with_fp32':float((p==choice).mean()),'centered_logit_mae_vs_fp32':float(abs(z-truth).mean()),'human_metrics':metrics(y,choice)}
  result={'unrepaired':measure(raw),'fp32':measure(truth)}
  for typ in ('paired','shuffled'):
   fn=wd/(state+'-'+typ+'.safetensors');assert sha(fn)==rep['states'][state][typ]['weights_sha256']
   val=center(x@arr(load_file(str(fn))['weight'])[:7].T)
   result[typ]=measure(val)
  out['states'][state]=result
 savej(wd/'validation.json',out)
 print('CPU precision repair',arm,{s:{k:(round(v['agreement_with_fp32'],4),round(v['human_metrics']['macro_f1'],4)) for k,v in q.items()} for s,q in out['states'].items()},flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('arm',choices=['alone','local']);q=a.parse_args();frozen(q.arm);evaluate(q.arm)
