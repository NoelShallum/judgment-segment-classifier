"""Describe indistinguishable input strings without using the diagnostic to fit a model."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit import NAMES

h=Path(__file__).resolve().parent
csv_path=h.parents[1]/'references/official/test.csv'
with np.load(h/'official-predictions.npz',allow_pickle=False) as z:
    gold=z['gold'];ids=z['case_ids'];off=z['offsets'];old=z['original'];current=z['legal_context']
docs=defaultdict(list)
with csv_path.open(encoding='utf-8-sig',newline='') as file:
    for position,row in enumerate(csv.DictReader(file)):
        if row['Text'].strip():docs[row['Index']].append((position,row['Text'],NAMES.index(row['Label'])))

keys={width:defaultdict(lambda:np.zeros(7,dtype=np.int32)) for width in (1,3,5)}
four=defaultdict(list)
for doc,lo,hi in zip(ids,off,off[1:]):
    rows=docs[str(doc)]
    np.testing.assert_array_equal([x[2] for x in rows],gold[lo:hi])
    texts=[x[1] for x in rows]
    for j,(n,text,y) in enumerate(rows):
        for width,d in keys.items():
            half=width//2
            key=tuple(texts[k] if 0<=k<len(texts) else None for k in range(j-half,j+half+1))
            d[key][y]+=1
        a=int(old[lo+j]);b=int(current[lo+j])
        if {a,b}=={0,5} and y in {0,5} and 5<=len(text.split())<=15:
            four[(a,b,y)].append({'case':str(doc),'csv_data_row_zero_based':n,'text':text.strip(),
                                   'previous':NAMES[a],'context':NAMES[b],'gold':NAMES[y]})
out={}
for w,d in keys.items():
    x=np.stack(list(d.values()));amb=(x>0).sum(1)>1;err=int((x.sum(1)-x.max(1)).sum())
    out[w]=dict(identical_byte_string_window=w,distinct_inputs=len(d),conflicting_inputs=int(amb.sum()),
                rows_in_conflicting_inputs=int(x[amb].sum()),observed_minimum_errors_for_deterministic_classifier=err,
                maximum_empirical_accuracy=1-err/len(gold))
(h/'input-ambiguity.json').write_text(json.dumps({'nonblank_rows':len(gold),'cohort':'official test, published human labels',
    'rule':'Group by the exact Unicode strings as provided, including whitespace and punctuation, with distinctive judgment-boundary markers. Optimistic accuracy ceiling for those grouped inputs only. Each unique input receives its modal observed HUMAN label; this is an oracle and supplies no training evidence.',
    'windows':out},indent=2)+'\n')
print(out)
for k,items in four.items():print(k,len(items),[(v['case'],v['text']) for v in items[:3]])
