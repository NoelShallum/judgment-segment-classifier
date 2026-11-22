"""Measure the prespecified two-encoder deployment ablation at three fixed sample sizes."""
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from safetensors.numpy import load_file
import run_scale as scale
from legal_encode import sha,check

h=scale.H
stages=(8,24,72)
check()
assert (h/'selection.json').exists()
policy={'stages':list(stages),'primary':72,'rule':'Use each stage’s frozen coefficient tensor, both original contextual donors and their compact features. No old document model, no reviewer anchor and no additional transition correction. The all-4600 stage is the prespecified deployment comparison, independent of its test performance.',
        'frozen_before': 'evaluation of the expansion families on the official test',
        'basis_sha256':sha(h/'basis.safetensors'),'code_sha256':sha(Path(__file__)),
        'development_receipts':{n:sha(h/f'scale-{n}-freeze.json') for n in stages}}
path=h/'standalone-lock.json';scale.dump(path,policy)
params=load_file(str(h/'basis.safetensors'))
coefs={}
for n in stages:
    p=h/f'scale-{n}.safetensors';lock=json.loads((h/f'scale-{n}-freeze.json').read_text());assert sha(p)==lock['weights_sha256']
    coefs[n]=load_file(str(p))['coef']
result={};files=[path,Path(__file__)]
for g in ('select','confirm','fresh','test'):
    if g=='test':x,off,y,case=scale.fixed(g)
    else:
        d=scale.readbank(h/(g+'-panel.npz'));x,off,y,case=d.x,d.offsets,d.y,d.case
    preds={n:[] for n in stages}
    for lo in range(0,len(case),12):
        upper=min(lo+12,len(case));a,b=off[lo],off[upper]
        design=scale.lib.docbasis(scale.design(x[a:b],off[lo:upper+1]-a),params)
        for n in stages:preds[n].append((design@coefs[n]).argmax(1).astype(np.int8))
    preds={f'stage_{n}':np.concatenate(v) for n,v in preds.items()}
    for n in stages:
        if g!='test':
            with np.load(h/f'scale-{n}-{g}-predictions.npz') as existing:
                np.testing.assert_array_equal(preds[f'stage_{n}'],existing['scale_only'])
    result[g]={k:scale.lib.metric(y,p) for k,p in preds.items()}
    if g=='test':
        f=h/'standalone-official-predictions.npz';np.savez_compressed(f,case_ids=case,offsets=off,gold=y,**preds);files.append(f)
f=h/'standalone-results.json';scale.dump(f,result);files.append(f)
dest=h.parent/'scale-standalone.zip'
with ZipFile(dest.with_suffix('.part'),'w',ZIP_DEFLATED) as z:
    for p in files:z.write(p,p.name)
    z.writestr('SHA256SUMS.json',json.dumps({p.name:sha(p) for p in files}))
dest.with_suffix('.part').replace(dest)
print({g:{k:round(v['macro_f1']*100,3) for k,v in vals.items()} for g,vals in result.items()})
