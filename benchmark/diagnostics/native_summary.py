"""Pool both native feed-forward confirmation cohorts at the judgment level."""
import io
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from audit import intervals,stats,cm

h=Path(__file__).resolve().parent
w=h.parent/'context-mlp'
with ZipFile(w/'context-mlp-dev.zip') as z:
    arrays={}
    for g in ('confirm','fresh'):
        with np.load(io.BytesIO(z.read(g+'-pred.npz')),allow_pickle=False) as f:arrays[g]={k:f[k] for k in f.files}
    freeze=json.loads(z.read('freeze.json'))
a,b=arrays.values()
assert not set(a['case_ids'])&set(b['case_ids'])
off=np.r_[a['offsets'],a['offsets'][-1]+b['offsets'][1:]]
y=np.r_[a['gold'],b['gold']];base=np.r_[a['context'],b['context']]
result={}
for k in ('context','edit','shuffled','standalone-original','standalone-edit','standalone-shuffled'):
    pred=np.r_[a[k],b[k]]
    result[k]=dict(scores=stats(cm(y,pred)),paired=intervals(y,pred,base,off),
                   fixed=int(np.sum((pred==y)&(base!=y))),harmed=int(np.sum((pred!=y)&(base==y))))
(w/'pooled-confirmation.json').write_text(json.dumps(dict(cases=len(off)-1,rows=len(y),source=freeze['source'],scores=result),indent=2)+'\n')
for k,v in result.items():print(k,round(100*v['scores']['macro_f1'],4),v['scores']['pair_errors'])
