"""Audit the fixed fast models on the 32 additional validation judgments and the complete nested 48."""
import argparse
import hashlib
import json
import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file
from tokenizers import Tokenizer
from prepare import ROOT,LABELS,docs,save
from study import metrics,compare
from uncertainty import interval
from vocab import SRC,raw_data,priors
from joint import decode
from portable_vocab import classify

def main():
    parse=argparse.ArgumentParser();parse.add_argument('method',choices=['vocabulary','sentence']);op=parse.parse_args().method
    casefile=ROOT.parent/'shared-confirmation.json';manifest=json.loads(casefile.read_text())
    ids=manifest['extended_validation_judgment_ids']
    doc={cid:{'id':cid,'ncase':len(rs),'rows':rs} for cid,rs in docs('val.csv') if cid in ids}
    assert set(doc)==set(ids)
    tok=Tokenizer.from_file(str(ROOT/'model/tokenizer.json'))
    allp={}
    if op=='vocabulary':
        ckpt=ROOT/'vocab-screen/compiled-token-scores.safetensors';w=load_file(str(ckpt))
        for name in ['text','structure']:
            allp[name]=np.concatenate([classify([r['text'] for r in doc[c]['rows']],w,tok,name=='structure') for c in ids])
        with np.load(ROOT/'vocab-common16-predictions.npz') as data:
            for name in ['text','structure']:assert np.array_equal(allp[name][:len(data[name])],data[name])
        root=ROOT/'vocab-screen';primary='structure'
    else:
        root=ROOT/'sentence-vocab';freeze=json.loads((root/'freeze.json').read_text());primary=freeze['primary_name']
        ds=raw_data([doc[c] for c in ids],tok);y=ds['y']
        from sentence_vocab import from_ids,head
        X=from_ids(ds['ids'])
        with np.load(ROOT/'vocab-counts.npz') as z:pp=z['position'];tx=z['log_transition']
        with np.load(ROOT/'sentence-vocab-gram.npz') as z:gr=z['gram'];sx=z['sumx'];freq=z['freq']
        with safe_open(str(ROOT/'model/model.safetensors'),framework='numpy') as f:base=f.get_tensor(SRC)
        pos=priors(ds,pp)
        for name,params in freeze['plans'].items():
            _,powr,reg,shrink,g,pc,tc=params
            if 'edited' in name:
                ckpt=root/f'{name}.safetensors';patch=load_file(str(ckpt))
                W=patch['head.weight'].T;b=patch['head.bias'];a=patch['embedding.role_residual'];lift=patch['role.to_hidden']
                table=np.empty((len(base),7))
                for i in range(0,len(base),1024):
                    mat=(base[i:i+1024]+(a[i:i+1024]@lift).astype(np.float32)).astype(np.float32)
                    table[i:i+1024]=mat@W
            else:
                W,b=head(gr,sx,freq,powr,reg);table=base@W
            allp[name]=decode(ds,X@table+b+pc*pos,tx,tc)
            ck=root/'best_edited_structured-shared16.npz'
            if name=='best_edited_structured':
                with np.load(ck) as p:assert np.array_equal(allp[name][:len(p['edited'])],p['edited'])
    gold=np.asarray([r['y'] for c in ids for r in doc[c]['rows']]);cids=np.concatenate([np.repeat(c,len(doc[c]['rows'])) for c in ids])
    prior=np.concatenate([json.loads((ROOT.parent/'pooler/shared-cases'/(c+'.json')).read_text())['predictions']['unchanged'] for c in ids])
    assert len(prior)==len(gold)
    out={'method':op,'protocol_sha256':hashlib.sha256((ROOT/'extra32-protocol.json').read_bytes()).hexdigest(),
         'role_order':LABELS,'case_manifest_sha256':hashlib.sha256(casefile.read_bytes()).hexdigest(),'primary':primary,'sets':{}}
    for tag,cohort in [('additional32',ids[16:]),('all48',ids)]:
        mask=np.isin(cids,cohort);g=gold[mask];d=cids[mask];old=prior[mask]
        r={'cases':len(cohort),'rows':len(g),'ids':cohort,'incumbent':metrics(g,old),'models':{}}
        for k,p in allp.items():
            pred=p[mask];r['models'][k]={**metrics(g,pred),'versus_incumbent':compare(g,old,pred),'paired_interval_vs_incumbent':interval(d,g,old,pred)}
            if k=='best_edited_structured':r['models'][k]['paired_interval_vs_sentence_optimized_zero']=interval(d,g,allp['best_zero_structured'][mask],pred)
        out['sets'][tag]=r
        print(op,tag,len(cohort),len(g),r['models'][primary]['macro_f1_7'],r['incumbent']['macro_f1_7'],flush=True)
    np.savez_compressed(root/'extended-predictions.npz',case=cids,gold=gold,incumbent=prior,csv_row=np.asarray([r['row'] for c in ids for r in doc[c]['rows']]),**allp)
    save(root/'extended-report.json',out)
if __name__=='__main__':main()
