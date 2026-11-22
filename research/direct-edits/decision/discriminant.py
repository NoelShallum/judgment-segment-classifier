"""Closed-form discriminant write with a physically bias-free seven-role output tensor."""
import argparse
import itertools
import json
import numpy as np
import torch
from safetensors.torch import load_file,save_file
from fetch import ROOT,sha
from weights import source,arr,read_rows,metrics,savej,wtproof
GRID=list(itertools.product((.1,1.,10.),(0.,.5,1.),(0.,.35)))
def build(x,roles,orig,config,params):
 x=x.astype(np.float64);origin=orig.astype(np.float64)
 d=x.mean(0);d/=d@d
 positive=x@d
 if not (positive>0).all():raise ValueError('training constant coordinate changed sign')
 nx=x/positive[:,None];mean=nx.mean(0);z=nx-mean
 mu=np.stack([z[roles==k].mean(0) for k in range(7)])
 residual=z-mu[roles]
 eigen,v=np.linalg.eigh(residual@residual.T)
 eigen=np.maximum(eigen,0.)
 scale=eigen.sum()/min(x.shape[0]-7,x.shape[1])
 projected=residual@mu.T
 projected=v.T@projected
 target_scale=float(np.mean(np.std(x@origin.T,axis=1)))
 priors=np.array(config['full_training_prior_counts']);logp=np.log(priors/priors.sum())
 solved={}
 for lam,beta,native_frac in params:
  ridge=lam*scale
  # Shared inverse within-class covariance via Woodbury. Scale by sample count
  # to match covariance rather than the residual scatter matrix.
  W=((mu.T-residual.T@(v@(projected/(eigen[:,None]+ridge))))*(len(x)/ridge)).T
  offset=-.5*(W*mu).sum(1)+beta*logp-W@mean
  coeff=W+np.outer(offset,d)
  coeff-=coeff.mean(0,keepdims=True)
  znew=x@coeff.T
  coeff*=target_scale/max(float(np.mean(np.std(znew,axis=1))),1e-8)
  solved[(lam,beta,native_frac)]=(1.-native_frac)*coeff+native_frac*(origin-origin.mean(0,keepdims=True))
 return solved,d
def freeze(arm='packed-ordered',tier='whole16',backend='ov-GPU-f32'):
 cfg=json.loads((ROOT/'discriminant-plan.json').read_text());wd=ROOT/'weights'/backend/arm/tier;data=read_rows(backend,arm,'train',tier)
 x,y,cases=(data[k] for k in ('hidden','y','sample_case'));t=source()[0];orig=arr(t)[:7]
 registry=json.loads((ROOT/'packed-full-plan.json').read_text());groups=registry['fixed_fit_ids']
 group_of={cid:j%3 for j,cid in enumerate(groups)}
 fold=np.array([group_of[c] for c in cases]);pred={g:np.zeros(len(y),int) for g in GRID}
 for i in range(3):
  keep=fold!=i;assert np.bincount(y[keep],minlength=7).min()>0
  heads,d=build(x[keep],y[keep],orig,cfg,GRID)
  for g,w in heads.items():pred[g][~keep]=(x[~keep]@w.T).argmax(1)
 score={g:metrics(y,p) for g,p in pred.items()};best=max(GRID,key=lambda g:(score[g]['macro_f1'],score[g]['accuracy'],g[0],-g[1],g[2]))
 report={'configuration':sha(ROOT/'discriminant-plan.json'),'train_support':np.bincount(y,minlength=7).tolist(),'train_count':len(y),'group_ids':group_of,'chosen':best,
         'cross_validated_macro_f1':score[best]['macro_f1'],'cross_validated_accuracy':score[best]['accuracy'],
         'grid':[dict(parameters=g,out_of_fold_macro_f1=s['macro_f1'],out_of_fold_accuracy=s['accuracy']) for g,s in score.items()],'files':{}}
 for name,labels in [('roles',y),('shuffle',np.random.default_rng(221126).permutation(y))]:
  heads,d=build(x,labels,orig,cfg,[best]);changed=t.clone();changed[:7]=torch.tensor(heads[best]).to(t.dtype)
  assert torch.equal(changed[7:],t[7:])
  fn=wd/(name+'-discriminant.safetensors');save_file({'weight':changed},str(fn))
  report['files'][name]=dict(sha256=sha(fn),**wtproof(t,changed))
 np.save(wd/'discriminant-bias-coordinate.npy',d,allow_pickle=False)
 savej(wd/'discriminant-frozen.json',report);print('discriminant frozen',best,score[best]['macro_f1'],flush=True)
def evaluate(arm='packed-ordered',tier='whole16',backend='ov-GPU-f32'):
 from compare import stats,scores
 wd=ROOT/'weights'/backend/arm/tier;info=json.loads((wd/'discriminant-frozen.json').read_text());x=read_rows(backend,arm,'val',tier)
 h=x['hidden'];y=x['y'];c=x['sample_case'];d=np.load(wd/'discriminant-bias-coordinate.npy')
 pp=json.loads((wd/'predictions.json').read_text());base=np.array(pp['predictions']['original']);cm_base,_=stats(y,base,c)
 base_inc=[]
 for_case={}
 for cid in set(c):
  q=json.loads((ROOT.parent/'pooler/shared-cases'/(str(cid)+'.json')).read_text());for_case[cid]=q['predictions']['unchanged']
 for cid,idx in zip(c,x['position']):base_inc.append(for_case[cid][idx])
 inc=np.array(base_inc);cm_inc,_=stats(y,inc,c)
 rng=np.random.default_rng(760230);boots=rng.integers(0,len(cm_base),size=(4000,len(cm_base)))
 n0=scores(cm_base[boots].sum(1));n1=scores(cm_inc[boots].sum(1));out={'nonpositive_coordinate_rows':int((h@d<=0).sum()),'minimum_coordinate':float((h@d).min()),'methods':{}}
 for name,metadata in info['files'].items():
  fn=wd/(name+'-discriminant.safetensors');assert sha(fn)==metadata['sha256']
  W=arr(load_file(str(fn))['weight'])[:7];pred=(h@W.T).argmax(1);cm,m=stats(y,pred,c);delta=scores(cm[boots].sum(1))
  out['methods'][name]={'metrics':m,'fixes_vs_native':int(((base!=y)&(pred==y)).sum()),'breaks_vs_native':int(((base==y)&(pred!=y)).sum()),
    'change_ci_vs_native':np.quantile(delta-n0,[.025,.975]).tolist(),'change_ci_vs_incumbent':np.quantile(delta-n1,[.025,.975]).tolist()}
 savej(wd/'discriminant-validation.json',out);print('discriminant',[(k,v['metrics']['macro_f1']) for k,v in out['methods'].items()],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','score']);a=p.parse_args();(freeze if a.mode=='freeze' else evaluate)()
