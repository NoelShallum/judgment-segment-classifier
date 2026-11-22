"""Aligned, predeclared shared replay of the fast lexical path on all 16 official validation cases."""
import hashlib
import json
import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file
from tokenizers import Tokenizer
from vocab import SRC, PERM, raw_data, score_tokens, priors, fit_coeffs, target_logs
from joint import decode
from extract import groups
from prepare import ROOT, save, LABELS
from study import metrics, compare
from uncertainty import interval

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    protocol=ROOT/'vocab-common-protocol.json';reg=json.loads(protocol.read_text())
    assert digest(ROOT/'vocab.py')==reg['source_sha256']
    loc=ROOT/'vocab-screen';r=json.loads((loc/'report.json').read_text())
    p0=max(r['experiments'],key=lambda key:r['experiments'][key]['best_selection'][0])
    frozen={"primary":p0,"selection_scores":{k:v['best_selection'][0] for k,v in r['experiments'].items()},
            "checkpoints":{k:digest(loc/f'{k}-patch.safetensors') for k in r['experiments']},
            "prospective_protocol_sha256":digest(protocol),"fit_receipt_sha256":digest(ROOT/'vocab-counts.json')}
    fp=ROOT/'vocab-common-freeze.json'
    if not fp.exists():save(fp,frozen)
    assert json.loads(fp.read_text())==frozen
    bank=groups('common')['common']
    tok=Tokenizer.from_file(str(ROOT/'model/tokenizer.json'))
    ds=raw_data(bank,tok);y=ds['y']
    with safe_open(str(ROOT/'model/model.safetensors'),framework='numpy') as f:emb=f.get_tensor(SRC)
    with np.load(ROOT/'vocab-counts.npz') as z:
        df=z['token_document_frequency'].astype(float);prior=z['class_counts'];pp=z['position'];tx=z['log_transition']
    prior_score=priors(ds,pp);basemap={}
    result={"status":"complete","scope":"prospectively registered direct edits to the pretrained input table. This lexical skip readout bypasses the 24 contextual layers.",
            "protocol":protocol.name,"freeze":fp.name,"freeze_sha256":digest(fp),"role_order":LABELS,"cases":len(bank),"rows":len(y),
            "ids":[q['id'] for q in bank],"train_fit_cases":len(json.loads((ROOT/'vocab-counts.json').read_text())['fit_ids']),
            "train_fit_rows":int(prior.sum()),"primary_name":p0,"methods":{}}
    truth=[];old=[]
    for g in bank:
        v=json.loads((ROOT.parent/'pooler'/'shared-cases'/(g['id']+'.json')).read_text())
        assert v['rows']==len(g['rows']) and v['manifest_sha256']==reg['manifest_sha256']
        old.extend(v['predictions']['unchanged'])
    old=np.asarray(old)
    result['incumbent_identity']='frozen Attoflow 20-percent InLegalBERT reviewer route with unchanged pooler'
    result['incumbent']=metrics(y,old)
    pred={"case":ds['case'],"source_csv_row":ds['row'],"sentence_index":ds['ix'],"char_start":ds['start'],"char_end":ds['end'],
          'gold':y,'incumbent':old}
    for name,ex in r['experiments'].items():
        _,s,gain,ps,ts=ex['best_selection'];_,sb,_,pb,tb=ex['independent_baseline_selection']
        patch=load_file(str(loc/f'{name}-patch.safetensors'));key=patch['vocab.role_residual'];lift=patch['role.to_hidden']
        # Recover the FP64 original W from the same dictionary/regularizer used in the fitting receipt.
        if s not in basemap:basemap[s]=fit_coeffs(emb,target_logs(df,prior,s),df.sum(1))
        if sb not in basemap:basemap[sb]=fit_coeffs(emb,target_logs(df,prior,sb),df.sum(1))
        W=basemap[s]
        assert np.array_equal(W.T.astype(np.float32),patch['head.weight'])
        orig=emb@W
        physical=np.empty_like(orig);hasher=hashlib.sha256()
        for i in range(0,len(emb),1024):
            actual=(emb[i:i+1024]+(key[i:i+1024]@lift).astype(np.float32)).astype(np.float32)
            hasher.update(actual.tobytes());physical[i:i+1024]=actual@W
        assert hasher.hexdigest()==ex['tensor']['materialized_sha256']
        scores=score_tokens(ds,physical)+ps*prior_score
        p=decode(ds,scores,tx,ts)
        zero=decode(ds,score_tokens(ds,orig)+ps*prior_score,tx,ts)
        optimum=decode(ds,score_tokens(ds,emb@basemap[sb])+pb*prior_score,tx,tb)
        shuffled=decode(ds,score_tokens(ds,orig)+score_tokens(ds,key[PERM])+ps*prior_score,tx,ts)
        result['methods'][name]={k:metrics(y,v) for k,v in [('primary',p),('paired_zero',zero),('reoptimized_zero',optimum),('equal_norm_shuffle',shuffled)]}
        result['methods'][name]['compared_to_incumbent']=compare(y,old,p)
        result['methods'][name]['paired_bootstrap_vs_incumbent']=interval(ds['case'],y,old,p)
        result['methods'][name]['paired_bootstrap_vs_optimized_head_only']=interval(ds['case'],y,optimum,p)
        result['methods'][name]['compared_to_reoptimized_zero']=compare(y,optimum,p)
        result['methods'][name]['tensor']=ex['tensor'];result['methods'][name]['checkpoint']=str((loc/f'{name}-patch.safetensors').relative_to(ROOT))
        pred[name]=p;pred[name+'_logits']=scores.astype(np.float32);pred[name+'_optimized_head']=optimum
        print(name,[(k,round(v['macro_f1_7'],6)) for k,v in result['methods'][name].items() if isinstance(v,dict) and 'macro_f1_7' in v],flush=True)
    result['primary']=result['methods'][p0];save(ROOT/'vocab-common16-report.json',result)
    np.savez_compressed(ROOT/'vocab-common16-predictions.npz',**pred)
if __name__=='__main__':main()
