"""Stratified and paired assessments of block position, separate from the full-corpus comparison."""
import json
import numpy as np
from fetch import ROOT,sha
from compare import stats,scores
def load(arm,tier):
 p=ROOT/'weights/ov-GPU-f32'/arm/tier/'predictions.json'
 d=json.loads(p.read_text())
 cs=np.array(d['cases']);ix=np.array(d['positions'])
 return p,np.array(d['gold']),cs,ix,{k:np.array(v) for k,v in d['predictions'].items()}
def generate():
 where=ROOT/'weights/ov-GPU-f32'
 plan=json.loads((ROOT/'plan.json').read_text())
 f,y,c,i,p=load('packed-ordered','whole16')
 assert len(y)==3558
 slot=[];size=[];layouts={}
 for cid in np.unique(c):
  path=ROOT/'features/ov-GPU-f32/packed-ordered/layouts'/('val-'+cid+'.json');j=json.loads(path.read_text())
  assert j['invalid_label_counterfactual_identical']
  layouts[cid]=j['index_to_slot_group_size_and_index']
 for cid,idx in zip(c,i):s,n,_=layouts[cid][str(idx)];slot.append(s);size.append(n)
 slot=np.array(slot);size=np.array(size);prog={}
 for name,pred in p.items():
  strata={}
  for key,k in [('slot'+str(s),slot==s) for s in range(8)]+[('block length 1-3',size<=3),('block length 4-7',(size>=4)&(size<=7)),('block length 8',size==8)]:
   if not np.any(k):continue
   _,d=stats(y[k],pred[k],c[k]);d['n']=int(k.sum());strata[key]=d
  prog[name]=strata
 out={'full_ordered_reference':sha(f),'slot_note':'slots are zero-indexed. Stratum scores are descriptive: they contain different sentences and class mixes. Matched rotation and singleton comparisons are reported separately.','n':3558,'frozen_design':sha(ROOT/'packed-full-plan.json'),'slot_strata':prog,'matched_64':{}}
 pair_index={(cid,int(idx)):j for j,(cid,idx) in enumerate(zip(c,i))}
 keys=[(cid,int(idx)) for cid in plan['val_judgment_ids'] for idx in plan['validation_indices'][cid][:4]]
 take=np.array([pair_index[k] for k in keys]);assert len(take)==64
 rng=np.random.default_rng(760230);boots=rng.integers(0,16,(4000,16))
 for arm in ('packed-rotated','packed-alone'):
  basepath=where/arm/'match64'/'predictions.json'
  if not basepath.exists():continue
  otherfile,yy,cc,ii,pp=load(arm,'match64')
  m={(cid,int(idx)):j for j,(cid,idx) in enumerate(zip(cc,ii))}
  ot=np.array([m[k] for k in keys]);assert np.array_equal(y[take],yy[ot])
  bymethod={}
  for method in p:
   a,sa=stats(y[take],p[method][take],c[take]);b,sb=stats(yy[ot],pp[method][ot],cc[ot])
   boot=(scores(a[boots].sum(1))-scores(b[boots].sum(1)))
   bymethod[method]={'ordered':sa,'comparison':sb,'ordered_minus_comparison':sa['macro_f1']-sb['macro_f1'],
      'paired_group_boot_95':np.quantile(boot,[.025,.975]).tolist(),'changed_predictions':int((p[method][take]!=pp[method][ot]).sum()),
      'ordered_fixes':int(((p[method][take]==y[take])&(pp[method][ot]!=yy[ot])).sum()),'ordered_breaks':int(((p[method][take]!=y[take])&(pp[method][ot]==yy[ot])).sum())}
  out['matched_64'][arm]={'source':sha(otherfile),'by_method':bymethod}
 dest=ROOT/'packed-position-effects.json';dest.write_text(json.dumps(out,indent=2))
 return dest
if __name__=='__main__': print(generate())
