"""Describe how each native terminal attention head distributes cue probability over current vs other sentence tokens."""
import argparse
import numpy as np
from attention_write import params,read_file,replay,block_files,DIR
from weights import savej
def evaluate(mode):
 w=params()
 files=(list((DIR/'cpu-probes').glob('*.npz')) if mode=='probe' else [f for _,f in block_files('train')])
 totals=[];on=[];others=[];ent=[]
 for f in files:
  rows,b=read_file(f,w);_,p,_,_=replay(b,w)
  for i in range(len(rows)):
   own=b['owner']==i;other=(b['owner']>=0)&(~own)
   if not other.any():continue
   a=p[i][:,own].sum(1);c=p[i][:,other].sum(1)
   totals.append(np.log((a/own.sum()+1e-10)/(c/other.sum()+1e-10)));on.append(a);others.append(c)
   ent.append(-(p[i]*np.log(np.maximum(p[i],1e-20))).sum(1)/np.log(p.shape[-1]))
 ratios=np.stack(totals);on=np.stack(on);others=np.stack(others);ent=np.stack(ent)
 result={'cohort':mode,'blocks':len(files),'cues_with_other_sentences':len(ratios),'measure':'log(mean probability per own token / mean probability per other token), calculated separately for each cue and head',
  'heads':[{'head':i,'mean_log_ratio':float(ratios[:,i].mean()),'median_log_ratio':float(np.median(ratios[:,i])),'fraction_preferring_others_per_token':float((ratios[:,i]<0).mean()),'own_mass':float(on[:,i].mean()),'other_mass':float(others[:,i].mean()),'mean_attention_entropy_normalized':float(ent[:,i].mean())} for i in range(16)]}
 savej(DIR/(mode+'-per-head.json'),result)
 print(mode,'ordered mean log ratio',[(z['head'],round(z['mean_log_ratio'],3)) for z in sorted(result['heads'],key=lambda z:z['mean_log_ratio'])],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['probe','train']);a=p.parse_args();evaluate(a.mode)
