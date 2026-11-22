"""Audit whether reconstructed judgments are exactly shared across splits or evaluation cohorts."""
from collections import defaultdict
import csv
import hashlib
from itertools import groupby, combinations
import json
from pathlib import Path
import random

HERE=Path(__file__).resolve().parent


def signatures(split):
    rows_by_digest=defaultdict(list);by_id={};sizes={}
    with (HERE.parent/'references/official'/f'{split}.csv').open(newline='',encoding='utf-8-sig') as f:
        for k,rr in groupby(csv.DictReader(f),key=lambda r:r['Index']):
            texts=[r['Text'].strip() for r in rr]
            digest=hashlib.sha256(' '.join(t for t in texts if t).encode()).hexdigest()
            if k in by_id:raise ValueError(f'Case {k} appears noncontiguously')
            rows_by_digest[digest].append(k);by_id[k]=digest;sizes[k]=len(texts)
    return rows_by_digest,by_id,sizes


def main():
    groups={}
    for task in ('train','val','test'):
        groups[task],by_id,lengths=signatures(task)
        if task=='val':val,rows=by_id,lengths
    cross={}
    for a,b in combinations(groups,2):
        both=sorted(groups[a].keys() & groups[b].keys())
        cross[a+'_'+b]=[{'first':groups[a][i], 'second':groups[b][i]} for i in both]
    def fold(k):return int(hashlib.sha256(('attoflow-role-2026-11/'+k).encode()).hexdigest()[:8],16)%3
    cohorts={}
    for name,confirm,seed in [('select',False,6520),('confirm',True,6521)]:
        pool=sorted(k for k in rows if (fold(k)==0)==confirm and 40<=rows[k]<=750)
        cohorts[name]=random.Random(seed).sample(pool,72)
    prior=json.loads((HERE.parent/'astra-run-penalty/frozen-selection.json').read_text())
    cohorts['independent']=list(map(str,prior['selection_case_ids']+prior['confirmation_case_ids']))
    sig={k:{val[c] for c in collection} for k,collection in cohorts.items()}
    seen=sig['select']|sig['confirm']; excluded=[]
    for k in cohorts['independent']:
        if val[k] in seen:excluded.append(k)
        seen.add(val[k])
    meta={'method':'SHA256 of case text with original row whitespace trimmed and one joining space, exactly matching raw inference text.',
          'case_counts':{k:sum(map(len,v.values())) for k,v in groups.items()},'cross_split_identical_text':cross,
          'within_split_repeated_cases':{k:sum(len(i)-1 for i in v.values()) for k,v in groups.items()},
          'validation_cohorts':{k:{'cases':len(v),'distinct_case_texts':len(sig[k])} for k,v in cohorts.items()},
          'identical_text_across_validation_cohorts':{a+'_'+b:len(sig[a]&sig[b]) for a,b in combinations(sig,2)},
          'independent_exclusions_for_text_disjoint_check':excluded}
    (HERE/'overlap.json').write_text(json.dumps(meta,indent=2)+'\n')

if __name__=='__main__':main()
