"""Compute paired uncertainty for the completed official experiments at the pause."""
import json
from pathlib import Path

import numpy as np
from audit import cm, intervals, stats

h=Path(__file__).resolve().parent
with np.load(h/'official-predictions.npz',allow_pickle=False) as f:rows={k:f[k] for k in f.files}
y,off=rows['gold'],rows['offsets'];result={}
for new,old in [('semantic_neighbors','legal_context'),('anchored_scale','legal_context'),('decoder_pair','legal_context'),('semantic_neighbors','anchored_scale')]:
    a,b=rows[new],rows[old];first,second=stats(cm(y,a)),stats(cm(y,b))
    result[new+'_vs_'+old]=dict(
        delta_points=100*(first['macro_f1']-second['macro_f1']),accuracy_points=100*(first['accuracy']-second['accuracy']),
        pair_delta=first['pair_errors']-second['pair_errors'],ci=intervals(y,a,b,off),
        changed=int((a!=b).sum()),fixed=int(((a==y)&(b!=y)).sum()),harmed=int(((a!=y)&(b==y)).sum()))
(h/'comparisons-at-pause.json').write_text(json.dumps(result,indent=2)+'\n')
for name,item in result.items():
    print(name,round(item['delta_points'],3),[round(val*100,3) for val in item['ci']['macro_delta']],item['pair_delta'],item['ci']['pair_delta'])
