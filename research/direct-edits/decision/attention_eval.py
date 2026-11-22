"""Evaluate the physically frozen final-Q projection with the identical fixed role head. No GPU needed."""
import json
import hashlib
import numpy as np
from safetensors.torch import load_file
from attention_write import params,block_files,read_file,replay,check_reference,diagnostics,own_minus_other_logits,add_slot_diagnostics,summarize_slots,DIR,FUSED,D
from cohort import case_rows,LABEL_ID,incumbent_dir
from compare import stats,scores
from fetch import ROOT,sha
from weights import savej
def evaluate():
 recipe=json.loads((DIR/'frozen.json').read_text());w=params();rules={'base':None}
 for name,spec in recipe['controls'].items():
  f=DIR/(name+'-qkv.safetensors');assert sha(f)==spec['file']
  rules[name]=load_file(str(f))[FUSED][:D].float().numpy()-w['qkv'][:D]
 plan=json.loads((ROOT/'packed-full-plan.json').read_text());ds=case_rows('val',set(plan['fixed_validation_ids']))
 preds={k:[] for k in rules};attn={k:[] for k in rules};logit_pairs={k:[] for k in rules};slots={k:{} for k in rules};other={k:[] for k in rules if k!='base'};y=[];c=[];positions=[];baseline_error=0.
 for case,fn in block_files('val'):
  rows,b=read_file(fn,w);y.extend(LABEL_ID[ds[case][int(i)][1]] for i in rows);c.extend([case]*len(rows));positions.extend(map(int,rows))
  eligible=np.flatnonzero(~np.isin(np.arange(len(b['noncue'])),b['pos']))
  take=sorted(eligible,key=lambda j:hashlib.sha256(f'{case}|{fn.stem}|{int(j)}|non-cue'.encode()).digest())[:8]
  protected=b['noncue'][take]
  denom=np.maximum(np.linalg.norm(protected@w['qkv'][:D].T,axis=1),1e-9)
  for name,delta in rules.items():
   h,p,attention_logits,_=replay(b,w,delta)
   if name=='base':baseline_error=max(baseline_error,check_reference(case,rows,h,w,'val'))
   else:other[name].extend((np.linalg.norm(protected@delta.T,axis=1)/denom).tolist())
   preds[name].extend((h@w['readout'].T).argmax(1));observed=diagnostics(b,p);attn[name].extend(observed);add_slot_diagnostics(slots[name],observed);logit_pairs[name].extend(own_minus_other_logits(b,attention_logits))
 y=np.array(y);c=np.array(c);pp={k:np.array(v) for k,v in preds.items()}
 reference=json.loads((ROOT/'weights/ov-GPU-f32/packed-ordered/whole16/predictions.json').read_text())
 assert np.array_equal(y,reference['gold']) and np.array_equal(c,reference['cases']) and np.array_equal(positions,reference['positions'])
 assert np.array_equal(pp['base'],reference['predictions']['roles-readout'])
 inc_cases={cid:json.loads((incumbent_dir()/f'{cid}.json').read_text())['predictions']['unchanged'] for cid in set(c)}
 inc=np.asarray([inc_cases[cid][i] for cid,i in zip(c,positions)]);cm0,m0=stats(y,pp['base'],c);cmi,mi=stats(y,inc,c)
 rng=np.random.default_rng(760230);bs=rng.integers(0,16,(4000,16));base=scores(cm0[bs].sum(1));inci=scores(cmi[bs].sum(1))
 report={'rows':len(y),'judgments':16,'pre_registration':sha(ROOT/'attention-plan.json'),'frozen':sha(DIR/'frozen.json'),'reference_logit_error':baseline_error,'incumbent':mi,'methods':{}}
 cms={}
 for name,p in pp.items():
  cm,m=stats(y,p,c);cms[name]=scores(cm[bs].sum(1));aa=np.asarray(attn[name])
  q={'score':m,'attention_mean_own_mass_other_mass_log_per_token_rate_ratio_fraction_heads_preferring_other':aa.mean(0).tolist(),'mean_own_minus_other_attention_logit':float(np.mean(logit_pairs[name])),'position_in_packed_block':summarize_slots(slots[name]),
     'changed':int((p!=pp['base']).sum()),'corrected_vs_frozen_head':int(((p==y)&(pp['base']!=y)).sum()),'broken_vs_frozen_head':int(((p!=y)&(pp['base']==y)).sum()),
     'paired_95_delta_vs_frozen_head':np.quantile(cms[name]-base,[.025,.975]).tolist(),'paired_95_delta_vs_incumbent':np.quantile(cms[name]-inci,[.025,.975]).tolist()}
  if name in other:q['noncue_relative_query_norm_shift_median_p95_max']=np.quantile(other[name],[.5,.95,1.]).tolist()
  report['methods'][name]=q
 report['targeted_minus_matched_random_95']=np.quantile(cms['target']-cms['norm_matched'],[.025,.975]).tolist()
 savej(DIR/'validation.json',report);savej(DIR/'predictions.json',{'case':c.tolist(),'position':positions,'gold':y.tolist(),'predictions':{k:v.tolist() for k,v in pp.items()}})
 print('Q study',{k:v['score']['macro_f1'] for k,v in report['methods'].items()},flush=True)
if __name__=='__main__':evaluate()
