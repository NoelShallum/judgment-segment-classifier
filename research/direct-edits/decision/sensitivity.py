"""Replay scores after removing any validation sentence matching a fit sentence verbatim after case-folding/whitespace normalization."""
import json
import numpy as np
from compare import stats
from fetch import ROOT,sha
def evaluate():
 audit=json.loads((ROOT/'text-overlap.json').read_text())['packed12']
 rows={(s['validation_case'],s['position']) for s in audit['matching_positions']}
 f=ROOT/'weights/ov-GPU-f32/packed-ordered/whole16/predictions.json'
 d=json.loads(f.read_text());y=np.array(d['gold']);case=np.array(d['cases']);idx=d['positions']
 keep=np.array([(c,i) not in rows for c,i in zip(case,idx)])
 assert int((~keep).sum())==audit['validation_rows_matching_any_training_text']
 maps={c:json.loads((ROOT.parent/'pooler/shared-cases'/(c+'.json')).read_text())['predictions']['unchanged'] for c in set(case)}
 inc=[maps[c][i] for c,i in zip(case,idx)]
 models={'matched_incumbent':inc,**d['predictions']};scores={}
 for name,seq in models.items():_,scores[name]=stats(y[keep],np.array(seq)[keep],case[keep])
 dst=ROOT/'packed-overlap-sensitivity.json'
 dst.write_text(json.dumps({'full_prediction_receipt':sha(f),'excluded_rows':int((~keep).sum()),'scored_rows':int(keep.sum()),'models':scores},indent=2))
 return dst
