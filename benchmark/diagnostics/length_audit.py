"""Describe the released system across judgment lengths using train-only cutpoints."""
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
from audit import cm,stats

h=Path(__file__).resolve().parent;train=h.parents[1]/'references/official/train.csv'
sizes=Counter()
with train.open(encoding='utf-8-sig',newline='') as f:
    for r in csv.DictReader(f):
        if r['Text'].strip():sizes[r['Index']]+=1
cuts=np.quantile(list(sizes.values()),[.25,.5,.75],method='inverted_cdf')
with np.load(h/'official-predictions.npz',allow_pickle=False) as z:
    ids=z['case_ids'];off=z['offsets'];n=np.diff(off);y=z['gold']
    matrices={k:np.stack([cm(y[a:b],z[k][a:b]) for a,b in zip(off,off[1:])]) for k in ('original','legal_context','anchored_scale','semantic_neighbors')}
bins=np.searchsorted(cuts,n,side='left');bybin=[]
for i in range(4):
    include=bins==i
    met={k:stats(v[include].sum(0)) for k,v in matrices.items()}
    bybin.append(dict(judgments=int(include.sum()),sentences=int(n[include].sum()),
                      observed_length_range=[int(n[include].min()),int(n[include].max())],
                      metrics={k:{s:v[s] for s in ('accuracy','macro_f1','pair_errors')} for k,v in met.items()}))
ci={};rng=np.random.default_rng(222603);values=[]
for _ in range(40):
    ix=rng.integers(0,len(ids),size=(100,len(ids)))
    mats=matrices['semantic_neighbors'][ix].sum(1)
    hits=np.diagonal(mats,axis1=1,axis2=2)
    f=(2.*hits/np.maximum(1,mats.sum(1)+mats.sum(2))).mean(1)
    values.extend(np.column_stack([f,hits.sum(1)/mats.sum((1,2))]))
interval=np.quantile(values,[.025,.975],axis=0)
for j,k in enumerate(('macro_f1','accuracy')):ci[k]=interval[:,j].tolist()
out=dict(official_train_judgments=len(sizes),train_length_quartile_cutpoints=cuts.tolist(),method='Sentence-pooled metrics within train-defined length bands. Empirical absolute 95% intervals for the semantic neighbor model resample entire official test judgments 4,000 times.',length_bands=bybin,absolute_intervals=ci)
(h/'judgment-length-audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(cuts,ci,[(b['judgments'],[round(v['macro_f1']*100,2) for v in b['metrics'].values()]) for b in bybin])
