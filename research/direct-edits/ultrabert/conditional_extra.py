"""Fresh additional-32 replay for previously frozen role-conditioned embedding writes."""
import json
import numpy as np
from tokenizers import Tokenizer
from safetensors.numpy import load_file
from prepare import ROOT,docs,save
from vocab import raw_data
from sentence_vocab import from_ids
from conditional_vocab import make_view,sequence
from study import metrics
from uncertainty import interval

def main():
    root=ROOT/'conditional-vocab'
    plan=json.loads((root/'freeze.json').read_text());patch=load_file(str(root/'conditional-compiled.safetensors'))
    manifest=json.loads((ROOT.parent/'shared-confirmation.json').read_text());ids=manifest['extended_validation_judgment_ids']
    by={k:{'id':k,'ncase':len(rs),'rows':rs} for k,rs in docs('val.csv') if k in ids}
    tok=Tokenizer.from_file(str(ROOT/'model/tokenizer.json'));d=raw_data([by[k] for k in ids],tok)
    source=from_ids(d['ids'])@patch['word.compiled_logits']
    bonus=make_view(d,patch['previous_role.compiled_residual'])
    pos=patch['position.logp'][np.minimum(23,24*d['ix']//d['ncase'])]
    _,_,_,pc,ts=plan['best_edit'];_,_,_,pc0,ts0=plan['best_zero'];tx=patch['transition.logp']
    pred=sequence(d,source,bonus,tx,pc,ts,1.,pos);zero=sequence(d,source,bonus,tx,pc0,ts0,0.,pos)
    with np.load(root/'shared16-predictions.npz') as z:
        assert np.array_equal(pred[:len(z['edited'])],z['edited'])
    old=np.concatenate([json.loads((ROOT.parent/'pooler/shared-cases'/(k+'.json')).read_text())['predictions']['unchanged'] for k in ids])
    rec={"weights_fixed_from":"conditional-vocab/freeze.json","sets":{}}
    for tag,subset in [('additional32',ids[16:]),('all48',ids)]:
        mask=np.isin(d['case'],subset);yy=d['y'][mask];cc=d['case'][mask]
        q={'cases':len(subset),'rows':len(yy),'edited':metrics(yy,pred[mask]),'optimized_zero':metrics(yy,zero[mask]),'incumbent':metrics(yy,old[mask]),
           'delta_vs_optimized_zero':interval(cc,yy,zero[mask],pred[mask]),'delta_vs_incumbent':interval(cc,yy,old[mask],pred[mask])}
        rec['sets'][tag]=q
        print(tag,q['rows'],q['edited']['macro_f1_7'],q['optimized_zero']['macro_f1_7'],q['delta_vs_optimized_zero'],flush=True)
    np.savez_compressed(root/'extended-predictions.npz',case=d['case'],ix=d['ix'],row=d['row'],start=d['start'],end=d['end'],gold=d['y'],edited=pred,optimized_zero=zero,incumbent=old)
    save(root/'extended-report.json',rec)
if __name__=='__main__':main()
