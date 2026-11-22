"""Reserve disjoint additional official training judgments before testing corpus scaling."""
import csv,gzip,hashlib,itertools,json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[1]
old=json.loads((R/'live-nov22/reviewer-round2/round2-plan.json').read_text());base=set(old['groups']['fit'])
roles=['None','Facts','Issue','Arguments of Petitioner','Arguments of Respondent','Reasoning','Decision']
def docs(split):
 with (R/'references/official'/(split+'.csv')).open(encoding='utf-8-sig',newline='') as f:
  for k,group in itertools.groupby(csv.DictReader(f),lambda row:row['Index']):
   s=[];y=[];h=hashlib.sha256()
   for r in group:
    t=r['Text']
    if t.strip():
     h.update(' '.join(t.casefold().split()).encode()+b'\0');s.append(t)
     if split=='train':y.append(roles.index(r['Label']))
   yield k,s,y,h.hexdigest()
seen={h for _,_,_,h in docs('val')}; vals={}
for k,s,y,h in docs('train'):
 if k not in base and h not in seen:vals[k]=(s,y,h)
 seen.add(h)
ids=sorted(vals,key=lambda k:hashlib.sha256(('atto-flow-legal-context-scale-v1|'+k).encode()).digest())
size=64
with gzip.open(H/'fit.jsonl.gz','wt',compresslevel=3) as f:
 for j,k in enumerate(ids):
  s,y,h=vals[k];f.write(json.dumps(dict(case=k,shard=j//size,text=s,gold=y),ensure_ascii=False)+'\n')
proto=dict(eligible_cases=len(ids),sentences=sum(len(s) for s,y,h in vals.values()),shard_size=size,case_ids=ids,
   train_only=True,disjoint_from_old_fit=sorted(base),hash_order='atto-flow-legal-context-scale-v1',
   method='Frozen two legal encoders over all rows. Program case-context readout using sums of feature outer products and class residuals. Choose on the original 48 selection judgments, prioritizing unseen text as well as full-set macro-F1. Compare two confirmation panels.',
   planned_expansions=[512,1536,len(ids)],evaluation_derivation='Public train excludes original 384 fit judgments, exact duplicates and case fingerprints in validation. Validation has prior exposure; official historically inspected prior to this family. No official labels used to write weights.')
(H/'protocol.json').write_text(json.dumps(proto,indent=2)+'\n')
print(len(ids),proto['sentences'],(H/'fit.jsonl.gz').stat().st_size)
