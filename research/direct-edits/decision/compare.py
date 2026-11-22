"""Compare each fixed GLiDE arm against the exact same incumbent sentence positions."""
import argparse
import json
from pathlib import Path
import numpy as np
from fetch import ROOT,sha
def confusion(y,p):
 return np.bincount(7*y+p,minlength=49).reshape(7,7)
def scores(m):
 f=2*m.diagonal(axis1=-2,axis2=-1)/np.maximum(m.sum(-1)+m.sum(-2),1)
 return f.mean(-1)
def stats(y,p,case):
 cm=np.stack([confusion(y[case==k],p[case==k]) for k in dict.fromkeys(case)])
 m=cm.sum(0)
 return cm,{'macro_f1':float(scores(m)),'accuracy':float(m.trace()/m.sum()),'per_class_f1':(2*m.diagonal()/np.maximum(m.sum(0)+m.sum(1),1)).tolist(),'support':m.sum(1).tolist()}
def evaluate(backend,arm,tier):
 plan=json.loads((ROOT/'plan.json').read_text());wd=ROOT/'weights'/backend/arm/tier
 r=json.loads((wd/'predictions.json').read_text())
 gold=np.array(r['gold']);case=np.array(r['cases']);position=np.array(r['positions'])
 predictions={n:np.asarray(p) for n,p in r['predictions'].items()}
 root=ROOT.parent/'pooler/shared-cases';inc=[];hashes={}
 for gid,i in zip(case,position):
  p=root/(str(gid)+'.json')
  j=json.loads(p.read_text());assert j['id']==gid and j['manifest_sha256']==plan['source']['shared_manifest']
  inc.append(j['predictions']['unchanged'][int(i)]);hashes[str(gid)]=sha(p)
 inc=np.asarray(inc);baseline,base=stats(gold,inc,case)
 orig=predictions['original'];orig_cm,_=stats(gold,orig,case)
 rng=np.random.default_rng(760230);idx=rng.integers(0,len(baseline),(4000,len(baseline)))
 inc_b=scores(baseline[idx].sum(1));orig_b=scores(orig_cm[idx].sum(1))
 out=dict(incumbent=base,j=case.size,distinct_judgments=len(baseline),sources=hashes,models={})
 for n,p in predictions.items():
  cm,res=stats(gold,p,case);delta=scores(cm[idx].sum(1))
  res['paired_macro_f1_diff_vs_native_95']=np.quantile(delta-orig_b,[.025,.975]).tolist()
  res['paired_macro_f1_diff_vs_incumbent_95']=np.quantile(delta-inc_b,[.025,.975]).tolist()
  res['fixed_vs_incumbent']=int(np.sum((inc!=gold)&(p==gold)))
  res['broken_vs_incumbent']=int(np.sum((inc==gold)&(p!=gold)))
  res['fixed_vs_native']=int(np.sum((orig!=gold)&(p==gold)))
  res['broken_vs_native']=int(np.sum((orig==gold)&(p!=gold)))
  out['models'][n]=res
 (wd/'paired.json').write_text(json.dumps(out,indent=2))
 print(tier,arm,'matched incumbent',out['incumbent']['macro_f1'])
 print({k:[round(v['macro_f1'],5),v['fixed_vs_incumbent'],v['broken_vs_incumbent']] for k,v in out['models'].items()})
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--backend',default='int8');ap.add_argument('--tier',default='pilot');ap.add_argument('arm')
 args=ap.parse_args();evaluate(args.backend,args.arm,args.tier)
