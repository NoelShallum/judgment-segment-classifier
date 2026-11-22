"""Predeclared train-only gate on the input embedding write, prospective incumbent comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file
from tokenizers import Tokenizer
from prepare import ROOT,LABELS,docs,hash_int,save
from extract import groups
from study import compare,metrics
from joint import decode
from vocab import SRC,PERM,raw_data,score_tokens,priors,fit_coeffs,target_logs
from uncertainty import interval

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def options_bank(d,emb,tab,w,p,tx,ts):
    score=score_tokens(d,tab)+p
    cand=decode(d,score,tx,ts)
    neighbor=np.zeros(len(cand),bool)
    same=(d['case'][:-1]==d['case'][1:])&(d['ix'][:-1]+1==d['ix'][1:])&(cand[:-1]==cand[1:])
    neighbor[:-1]|=same;neighbor[1:]|=same
    old=d['old'];row=np.arange(len(old))
    return cand,score[row,cand]-score[row,old],neighbor


def gate(d,opt,margin,near):
    if margin is None:return d['old'].copy()
    c,v,n=opt
    return np.where((v>=margin)&(n if near else True),c,d['old'])


def train_cases():
    with np.load(ROOT/'historical-incumbent-crossfit.npz') as a:
        ends=a['ends'];gold=a['gold'];pred=a['predicted']
        rows={str(cid):(gold[s:e],pred[s:e]) for cid,s,e in zip(a['cases'],ends,ends[1:])}
    seen=set(json.loads((ROOT/'vocab-counts.json').read_text())['fit_ids'])
    pools=json.loads((ROOT/'pools.json').read_text())['pools']
    for n in ['select','confirm']:
        for vals in pools[n].values():seen.update(i['id'] for i in vals)
    eligible=sorted(set(rows)-seen,key=lambda k: hash_int("ultra-hybrid-protocol-v1|"+k))
    num=2*len(eligible)//3
    actual={cid:{'id':cid,'ncase':len(rs),'rows':rs} for cid,rs in docs('train.csv') if cid in eligible}
    assert set(actual)==set(eligible)
    return rows,{n:[actual[i] for i in ids] for n,ids in [('select',eligible[:num]),('confirm',eligible[num:])]}


def main():
    root=ROOT/'hybrid-vocab';root.mkdir(exist_ok=True)
    protocol=ROOT/'hybrid-protocol.json'
    src=json.loads((ROOT/'vocab-screen/report.json').read_text())['experiments']['structure']
    _,s,g,ps,ts=src['best_selection']
    patch=load_file(str(ROOT/'vocab-screen/structure-patch.safetensors'))
    with np.load(ROOT/'vocab-counts.npz') as z:
        df=z['token_document_frequency'].astype(float);prior=z['class_counts'];pp=z['position'];tx=z['log_transition']
    with safe_open(str(ROOT/'model/model.safetensors'),framework='numpy') as f:emb=f.get_tensor(SRC)
    w=fit_coeffs(emb,target_logs(df,prior,s),df.sum(1));zero=emb@w
    delta=patch['vocab.role_residual']
    assert np.array_equal(w.T.astype(np.float32),patch['head.weight'])
    allm={'edited':zero+delta,'paired_zero':zero,'shuffled':zero+delta[PERM]}
    tok=Tokenizer.from_file(str(ROOT/'model/tokenizer.json'))
    old,gs=train_cases()
    pools={}
    for n,items in gs.items():
        d=raw_data(items,tok)
        d['old']=np.concatenate([old[v['id']][1] for v in items])
        np.testing.assert_array_equal(d['y'],np.concatenate([old[v['id']][0] for v in items]))
        pools[n]=d
    plans={}
    refs={}
    for n in ['select','confirm']:
        d=pools[n];pr=ps*priors(d,pp)
        refs[n]={m:options_bank(d,emb,tab,w,pr,tx,ts) for m,tab in allm.items()}
    for k in ['edited','paired_zero']:
        d=pools['select'];best=None;grid=[]
        for threshold,near in [(None,False),*[(j,near) for j in [1.,2.,4.,8.,16.] for near in [False,True]]]:
            pred=gate(d,refs['select'][k],threshold,near)
            m=metrics(d['y'],pred);c=int(np.sum(pred!=d['old']))
            row={"threshold":threshold,"adjacent_agreement":near,"selected_macro_f1_7":m['macro_f1_7'],"changes":c}
            grid.append(row)
            if best is None or (m['macro_f1_7'],-c)>(best['selected_macro_f1_7'],-best['changes']):best=row
        plans[k]={"choice":best,"all_trials":grid}
    freeze={"protocol_sha256":sha(protocol),"patch_sha256":sha(ROOT/'vocab-screen/structure-patch.safetensors'),
            "crossfitted_training_predictions_sha256":sha(ROOT/'historical-incumbent-crossfit.npz'),
            "selection_ids":[v['id'] for v in gs['select']],"train_confirmation_ids":[v['id'] for v in gs['confirm']],
            "plans":plans,"historical_set":"Original 144 official train judgments; archived BERT cross-fit reviewer on its historical referred subset is a gate-selection proxy. Common replay compares the complete frozen 20% route.",
            "role_order":LABELS}
    fp=root/'freeze.json'
    if not fp.exists():save(fp,freeze)
    assert freeze==json.loads(fp.read_text())
    items=groups('common')['common']
    d=raw_data(items,tok)
    d['old']=np.concatenate([json.loads((ROOT.parent/'pooler/shared-cases'/(v['id']+'.json')).read_text())['predictions']['unchanged'] for v in items])
    pools['common16']=d
    pr=ps*priors(d,pp)
    refs['common16']={m:options_bank(d,emb,tab,w,pr,tx,ts) for m,tab in allm.items()}
    rec={"method":"If an edited lexical prediction exceeds the incumbent class score by the preselected margin, use it, optionally requiring an identical adjacent prediction.",
         "freeze_sha256":sha(fp),"stage":"separate prospectively registered fusion","splits":{}}
    ep=plans['edited']['choice'];zp=plans['paired_zero']['choice']
    for n,d in pools.items():
        out={"incumbent":metrics(d['y'],d['old']),"cases":len(np.unique(d['case'])),"rows":len(d['y'])}
        blobs={"case":d['case'],"csv_row":d['row'],"ix":d['ix'],"char_start":d['start'],"char_end":d['end'],"gold":d['y'],"incumbent":d['old']}
        for name,key,cfg in [('edited','edited',ep),('matched_zero','paired_zero',ep),('reoptimized_zero','paired_zero',zp),('shuffle','shuffled',ep)]:
            pred=gate(d,refs[n][key],cfg['threshold'],cfg['adjacent_agreement'])
            out[name]={**metrics(d['y'],pred),"relative_to_incumbent":compare(d['y'],d['old'],pred),"paired_delta_vs_incumbent":interval(d['case'],d['y'],d['old'],pred)}
            blobs[name]=pred
        np.savez_compressed(root/f'{n}-predictions.npz',**blobs)
        rec['splits'][n]=out
        print(n,out['cases'],out['rows'],{k:round(out[k]['macro_f1_7'],6) for k in ['incumbent','edited','matched_zero','reoptimized_zero','shuffle']},out['edited']['relative_to_incumbent'],flush=True)
    save(root/'report.json',rec)
if __name__=='__main__':main()
