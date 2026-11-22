"""Audit exact normalized text reuse across cases, with no training or rule selection."""
import hashlib
import json
import numpy as np
from pathlib import Path
from prepare import ROOT,docs,save
from study import metrics

def normalized(s):return " ".join(s.casefold().split())
def digest(s):return hashlib.sha256(s.encode()).digest()

def main():
    manifest=json.loads((ROOT.parent/'shared-confirmation.json').read_text());extended=manifest['extended_validation_judgment_ids']
    ids=set(extended);fit=set(json.loads((ROOT/'vocab-counts.json').read_text())['fit_ids'])
    shared={cid:[normalized(r['text']) for r in rs] for cid,rs in docs('val.csv') if cid in ids}
    targets={txt for lines in shared.values() for txt in lines}
    target_docs={digest('\n'.join(lines)):cid for cid,lines in shared.items()}
    matched_all=set();matched_train=set();dupd=[]
    for cid,rs in docs('train.csv'):
        norm=[normalized(r['text']) for r in rs]
        inter=set(norm)&targets;matched_all.update(inter)
        if cid in fit:matched_train.update(inter)
        hh=digest('\n'.join(norm))
        if hh in target_docs:dupd.append({'official_train_case':cid,'shared_validation_case':target_docs[hh],'lexical_fit_case':cid in fit})
    with np.load(ROOT/'vocab-screen/extended-predictions.npz') as f:
        case=f['case'];gold=f['gold'];inc=f['incumbent'];lex=f['structure']
    assert case.tolist()==[k for k in extended for _ in shared[k]]
    concat=[t for k in extended for t in shared[k]]
    train_seen=np.asarray([t in matched_train for t in concat]);all_seen=np.asarray([t in matched_all for t in concat])
    r={'normalization':'Unicode casefold plus whitespace collapsing only. Compared full sentence text and ordered complete nonblank judgment, without labels.',
       'source':'official train.csv against frozen 48 validation cases, plus flags restricting to the actual 495 lexical-fit cases','exact_case_overlaps':dupd,'sets':{}}
    for name,ls in [('initial16',extended[:16]),('additional32',extended[16:]),('all48',extended)]:
        m=np.isin(case,ls)
        subset={'cases':len(ls),'rows':int(m.sum()),'exact_sentence_previously_seen_in_any_training_case':int((m&all_seen).sum()),
                'exact_sentence_seen_in_actual_495_fit_cases':int((m&train_seen).sum())}
        for tag,rows in [('seen_in_actual_fit',m&train_seen),('unseen_in_actual_fit',m&~train_seen)]:
            subset[tag]={'rows':int(rows.sum()),'lexical':metrics(gold[rows],lex[rows]),'incumbent':metrics(gold[rows],inc[rows])}
        r['sets'][name]=subset
    save(ROOT/'literal-overlap-audit.json',r)
    for k,q in r['sets'].items():print(k,q['rows'],q['exact_sentence_previously_seen_in_any_training_case'],q['exact_sentence_seen_in_actual_495_fit_cases'],round(q['unseen_in_actual_fit']['lexical']['macro_f1_7'],5),flush=True)
if __name__=='__main__':main()
