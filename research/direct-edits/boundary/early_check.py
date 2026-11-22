"""Predetermined training-only diagnostic; all five documents belong to the 12 fit documents."""
from score import PR,readcases,build,native,get_pred,evaluate,NAMES,KEYS,OUT,hsh,np,json
from safetensors.numpy import save_file
from safetensors import safe_open
trainids=PR['fit'][:3]; checkids=PR['fit'][3:5];train=readcases('train',trainids);check=readcases('train',checkids)
r=0.3;d=build(train)[r];s=build(train,True)[r];s=[c*(np.linalg.norm(a)/np.linalg.norm(c)) for a,c in zip(d,s)];pred={'native':{k:native(a).argmax(1) for k,a in check.items()},'edit':get_pred(check,d),'shuffled':get_pred(check,s)}
with safe_open(str(OUT.parent/'model/model.safetensors'),framework='np') as f:
    ws={k:f.get_tensor(k) for k in KEYS}
for name,delta in [('edit',d),('shuffled',s)]:save_file({k:ws[k]+v for k,v in zip(KEYS,delta)},str(OUT/('early-'+name+'-replacement.safetensors')))
report=dict(purpose='Pipeline diagnostic on five of the preassigned fit judgments; no shared validation, no selector and no official test. Data later reused for main fit.',fit=trainids,check=checkids,fit_rows=sum(len(x['y']) for x in train.values()),method='three native marginal state-projection writes, reg=0.3 and gain=1 fixed',results={name:evaluate(check,p)[0] for name,p in pred.items()},norms={k:[float(np.linalg.norm(a)),float(np.linalg.norm(b))] for k,a,b in zip(KEYS,d,s)});(OUT/'early-training-only.json').write_text(json.dumps(report,indent=2));print({k:{z:round(v[z],5) for z in ('macro_f1','accuracy','exact_span_micro_f1','rows')} for k,v in report['results'].items()})
