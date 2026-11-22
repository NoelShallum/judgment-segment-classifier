"""Actual gradient-free writes into Fastino's native option readout and last feed-forward projection."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file,load_file
from cohort import LABELS,LABEL_ID,case_rows
from fetch import ROOT,MODEL,sha,REV

torch.set_num_threads(1)
READOUT='weight'
INTERNAL='model.layers.27.mlp.Wo.weight'
GAIN='model.final_norm.weight'
ENC=MODEL/'model-00001-of-00001.safetensors'
HEAD=MODEL/'readout.safetensors'
EPS=1.e-5
def savej(path,data):
 tmp=path.with_suffix('.part')
 tmp.write_text(json.dumps(data,indent=2))
 tmp.replace(path)
def source():
 with safe_open(str(ENC),'pt',device='cpu') as f:
  g=f.get_tensor(GAIN)
  internal=f.get_tensor(INTERNAL)
 return load_file(str(HEAD))['weight'],g,internal
def arr(t): return t.to(torch.float32).cpu().numpy()
def read_rows(backend,arm,split,tier):
 plan=json.loads((ROOT/'plan.json').read_text())
 group={'train_cases':12,'validation_rows_per_shared_judgment':4} if tier in ('whole16','match64') else plan[tier]
 cases=plan['fit_judgment_ids'][:group['train_cases']] if split=='train' else plan['val_judgment_ids']
 docs=case_rows(split,set(cases))
 arrays={k:[] for k in ('hidden','logit','ntok','before_normalization','mlp_activation','y','sample_case','position')}
 for case in cases:
  path=ROOT/'features'/backend/arm/f'{split}-{case}.npz'
  with np.load(path,allow_pickle=False) as d:
   wanted=list(range(len(docs[case]))) if split=='train' or tier=='whole16' else plan['validation_indices'][case][:group['validation_rows_per_shared_judgment']]
   indexes={int(i):j for j,i in enumerate(d['idx'])}
   take=[indexes[i] for i in wanted]
   for name in ('hidden','logit','ntok','before_normalization','mlp_activation'): arrays[name].append(d[name][take])
   arrays['y'].append([LABEL_ID[docs[case][i][1]] for i in wanted])
   arrays['sample_case'].append([case]*len(take))
   arrays['position'].append(wanted)
 return {k:np.concatenate(v,axis=0) for k,v in arrays.items()}
def normed(x,g):
 x=x-x.mean(1,keepdims=True)
 return x/np.sqrt((x*x).mean(1,keepdims=True)+EPS)*g
def ridge(keys,resid,y):
 keys=keys.astype(np.float64)
 scale=float(np.sqrt(np.mean(np.sum(keys*keys,axis=1))))
 key=keys/scale
 w=1/np.sqrt(np.maximum(np.bincount(y,minlength=7),1)[y]);w=w/w.mean()
 gram=key@key.T
 damping=.15*np.diag(gram).mean()
 coeff=np.linalg.solve(gram+np.diag(damping/w),resid.astype(np.float64))
 return ((key.T@coeff)/scale).T
def wtproof(orig,modified):
 a,b=arr(orig),arr(modified)
 diff=b-a
 return dict(shape=list(a.shape),dtype=str(orig.dtype),changed_values=int(np.count_nonzero(diff)),total_values=a.size,
             frobenius_delta=float(np.linalg.norm(diff)),frobenius_original=float(np.linalg.norm(a)),maximum_abs_delta=float(np.abs(diff).max()),
             original_tensor_sha256=hashlib.sha256(orig.view(torch.uint8).numpy().tobytes()).hexdigest(),
             edited_tensor_sha256=hashlib.sha256(modified.view(torch.uint8).numpy().tobytes()).hexdigest())
def freeze(backend,arm,tier):
 train=read_rows(backend,arm,'train',tier)
 h,z,r,u,y=(train[k] for k in ('hidden','logit','before_normalization','mlp_activation','y'))
 tW,tg,tI=source();W,g=arr(tW)[:7],arr(tg)
 recreated=normed(r,g)@W.T
 max_abs=float(np.max(abs(recreated-z)))
 if max_abs>.05:raise ValueError(('native logits do not regenerate',max_abs))
 where=ROOT/'weights'/backend/arm/tier
 where.mkdir(parents=True,exist_ok=True)
 info=dict(donor_revision=REV,encoder_sha256=sha(ENC),readout_sha256=sha(HEAD),plan_sha256=sha(ROOT/'plan.json'),
           backend=backend,arm=arm,training=tier,train_sentences=len(y),training_judgments=list(dict.fromkeys(train['sample_case'])),
           support=np.bincount(y,minlength=7).tolist(),maximum_replay_native_error=max_abs,gradient_steps=0,files={})
 rng=np.random.default_rng(221126)
 # After LayerNorm, the class ordering is exactly the ordering under the
 # centered readout before the common positive normalization factor.
 directions=W*g
 directions-=directions.mean(1,keepdims=True)
 directions-=directions.mean(0,keepdims=True)
 lift=np.linalg.pinv(directions.astype(np.float64),rcond=1e-6)
 assert np.linalg.matrix_rank(directions,tol=1e-5)==6
 std=np.sqrt(r.var(1,keepdims=True)+EPS)
 for control,labels in (('roles',y),('shuffled',rng.permutation(y))):
  desired=4*(np.eye(7)[labels]-1/7)
  centered=z-z.mean(1,keepdims=True)
  d_head=ridge(h,desired-centered,labels)
  head=tW.clone()
  head[:7]=torch.from_numpy(W+d_head).to(tW.dtype)
  assert torch.equal(head[7:],tW[7:])
  dest=where/(control+'-readout.safetensors');save_file({READOUT:head.contiguous()},str(dest))
  info['files'][dest.name]=dict(sha256=sha(dest),tensor=READOUT,native_name='readout.safetensors:weight',edited_rows='0-6 inclusive',**wtproof(tW,head))
  residual=(desired-centered)*std
  d_keys=ridge(u,residual,labels)
  d_ffn=lift@d_keys
  new=(tI.float()+torch.from_numpy(d_ffn).to(torch.float32)).to(tI.dtype)
  dest=where/(control+'-internal.safetensors');save_file({INTERNAL:new.contiguous()},str(dest))
  info['files'][dest.name]=dict(sha256=sha(dest),tensor=INTERNAL,calculated_rank_before_rounding=6,**wtproof(tI,new))
 savej(where/'receipt.json',info)
 print('frozen',backend,arm,tier,'train',len(y),'classes',info['support'],'replay',max_abs,flush=True)
def metrics(y,p):
 m=np.bincount(7*y+p,minlength=49).reshape(7,7)
 f=2*m.diagonal()/np.maximum(m.sum(0)+m.sum(1),1)
 return {'macro_f1':float(f.mean()),'accuracy':float((p==y).mean()),'per_role_f1':f.tolist(),'support':m.sum(1).tolist(),'confusion':m.tolist()}
def score(backend,arm,tier):
 wd=ROOT/'weights'/backend/arm/tier
 info=json.loads((wd/'receipt.json').read_text());assert info['plan_sha256']==sha(ROOT/'plan.json')
 val=read_rows(backend,arm,'val',tier);h,z,r,u,y=(val[k] for k in ('hidden','logit','before_normalization','mlp_activation','y'))
 tW,tg,tI=source();W,g,I=arr(tW)[:7],arr(tg),arr(tI)
 pred={'original':z.argmax(1)}
 for name,result in info['files'].items():
  patch=wd/name
  assert sha(patch)==result['sha256']
  par=arr(load_file(str(patch))[result['tensor']])
  logits=h@par[:7].T if result['tensor']==READOUT else normed(r+u@(par-I).T,g)@W.T
  pred[name.split('.')[0]]=logits.argmax(1)
 predmeta={k:dict(metrics=metrics(y,p),fixed_vs_original=int(((p==y)&(pred['original']!=y)).sum()),broken_vs_original=int(((p!=y)&(pred['original']==y)).sum())) for k,p in pred.items()}
 result=dict(weights_receipt=sha(wd/'receipt.json'),val_sentences=len(y),val_judgments=len(set(val['sample_case'])),label_order=LABELS,arm=arm,
             context={'alone':'target only','local':'one preceding and one following sentence with target marked','packed-alone':'one target with shared option preamble','packed-ordered':'up to 8 ordered sentences with one cue per sentence','packed-shuffled':'same block after a deterministic sentence shuffle','packed-rotated':'same block rotated by half its sentence count, fixed ordered weights'}[arm],backend=backend,
             max_tokens=int(val['ntok'].max()),median_tokens=float(np.median(val['ntok'])),method=predmeta)
 savej(wd/'validation.json',result)
 savej(wd/'predictions.json',{'cases':val['sample_case'].tolist(),'positions':val['position'].tolist(),'gold':y.tolist(),'predictions':{k:p.tolist() for k,p in pred.items()}})
 print(backend,arm,tier,'VALIDATION',len(y),{k:round(v['metrics']['macro_f1'],5) for k,v in predmeta.items()},flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','score']);ap.add_argument('--backend',default='fp32');ap.add_argument('--arm',default='alone');ap.add_argument('--tier',default='pilot')
 args=ap.parse_args()
 (freeze if args.mode=='freeze' else score)(args.backend,args.arm,args.tier)
