"""Freeze a seven-class, case-disjoint prospective provider comparison from public LegalSeg."""
from collections import defaultdict,Counter
import csv,hashlib,json,random,re
from pathlib import Path
from seven_paths import ROOT,LABELS,folder_for
from source_filter import allow,SENSITIVE
NAMES=('None','Facts','Issue','Arguments of Petitioner','Arguments of Respondent','Reasoning','Decision')
P=ROOT/'prospective-seven'
def save(p,data):
    p.parent.mkdir(exist_ok=True,parents=True);tmp=p.with_suffix('.part');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def main():
    exclude=set();previous=set()
    for g in ('pilot','confirmation','paragraphs','homogeneous'):
        for a in json.loads((folder_for(g)/'approved.json').read_text()):
            exclude.add(a['doc']);previous.add(a['sha256'])
    h=json.loads((ROOT/'results/hierarchical-validation.json').read_text())['document']['groups']
    for g in ('test','cloud'):exclude.update(h[g]['case_ids'])
    exclude.update(k for k,_ in json.loads((ROOT/'results/fragment-validation.json').read_text())['groups']['test']['case_rows'])
    candidates=defaultdict(list)
    with (ROOT/'references/official/test.csv').open(encoding='utf-8-sig',newline='') as f:
        for i,x in enumerate(csv.DictReader(f)):
            case=x['Index']
            if case in exclude:continue
            t=allow(x['Text'])
            if not t or SENSITIVE.search(t):continue
            if not t.isascii() or re.search(r'\b(?:he|she|her|his|my|mine|address|surname|clinic|claimant|young|kill\w*|dead|dispute between|name\w*|drug\w*|corrupt\w*|rob\w*|sectioned|confin\w*)\b',t,re.I):continue
            digest=hashlib.sha256(t.encode()).hexdigest()
            if digest in previous:continue
            y=LABELS[NAMES.index(x['Label'])]
            candidates[y].append({'id':'sentence-'+str(i),'doc':case,'text':t,'gold':y,'sha256':digest,'source':'sentence','transformation':'whitespace normalized; leading enumerator removed when present'})
    rng=random.Random(95720113);picked=[];counts=Counter();seen=set(previous)
    for y,rs in sorted(candidates.items(),key=lambda pair:len(pair[1])):
        rng.shuffle(rs);ds=set()
        for r in rs:
            if r['doc'] in ds or counts[r['doc']]>=3 or r['sha256'] in seen:continue
            ds.add(r['doc']);seen.add(r['sha256']);counts[r['doc']]+=1;picked.append(r)
            if len(ds)==45:break
    rng.shuffle(picked);save(P/'pool.json',picked)
    print('Eligible and screened selections',dict(Counter(r['gold'] for r in picked)),'cases',len(counts),'excluded',len(exclude))
    (P/'blind-review.txt').write_text('\n'.join(str(i)+' '+r['text'] for i,r in enumerate(picked)))
if __name__=='__main__':main()
