"""Make matched, case-disjoint cohorts, with no use of validation or test labels in selection."""
import argparse
import csv
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path
from fetch import ROOT,sha
OFF=ROOT.parent.parent/'references/official'
LABELS=('None','Facts','Issue','Arguments of Petitioner','Arguments of Respondent','Reasoning','Decision')
LABEL_ID={name:i for i,name in enumerate(LABELS)}
def incumbent_dir():
 portable=ROOT/'portable-incumbent'
 return portable if portable.is_dir() else ROOT.parent/'pooler/shared-cases'
SEED='glide-decision-nov22-short-training-v1'
def digest(*values):
 return hashlib.sha256('|'.join(map(str,(SEED,*values))).encode()).hexdigest()
def case_rows(split,ids=None):
 cache=ROOT/'sampled-texts.json'
 if ids is not None and cache.is_file():
  package=json.loads(cache.read_text())
  plan=json.loads((ROOT/'plan.json').read_text())
  if package['source']!=plan['source']:raise ValueError('Portable sample source mismatch')
  records=package[split]
  if ids<=records.keys():return {i:rows for i,rows in records.items() if i in ids}
  if not (OFF/(split+'.csv')).is_file():raise ValueError('Missing judgment from portable sample')
 out=defaultdict(list)
 with (OFF/(split+'.csv')).open(newline='',encoding='utf-8-sig') as fh:
  for r in csv.DictReader(fh):
   if (ids is None or r['Index'] in ids) and (txt:=r['Text'].strip()):
    out[r['Index']].append((txt,r['Label']))
 return out
def plan():
 freq=defaultdict(Counter)
 with (OFF/'train.csv').open(newline='',encoding='utf-8-sig') as fh:
  for row in csv.DictReader(fh):
   if row['Text'].strip(): freq[row['Index']][row['Label']]+=1
 candidates=[i for i,c in freq.items() if 15<=c.total()<=32]
 chosen=[]; counts=Counter()
 for _ in range(12):
  def ordering(i):
   c=freq[i]
   # Concave coverage favors rare roles; every chosen case is included in full.
   import math
   gain=sum(math.log1p(counts[k]+c[k])-math.log1p(counts[k]) for k in LABELS)
   return (-gain/(c.total()**.6),digest('fit',i))
  nxt=min((i for i in candidates if i not in chosen),key=ordering)
  chosen.append(nxt);counts.update(freq[nxt])
 vh=ROOT.parent/'shared-confirmation.json'
 manifest=json.loads(vh.read_text())
 vids=manifest['small_validation_judgment_ids']
 val_count=Counter()
 with (OFF/'val.csv').open(newline='',encoding='utf-8-sig') as fh:
  for row in csv.DictReader(fh):
   if row['Index'] in vids and row['Text'].strip(): val_count[row['Index']]+=1
 vp={i:sorted(range(val_count[i]),key=lambda k:digest('validation-sentence',i,k))[:12] for i in vids}
 recipe={
  'source':{'train':sha(OFF/'train.csv'),'val':sha(OFF/'val.csv'),'shared_manifest':sha(vh)},
  'selection':SEED,'labels':LABELS,'fit_judgment_ids':chosen,
  'fit_lengths':{i:freq[i].total() for i in chosen},
  'pilot':{'train_cases':3,'validation_rows_per_shared_judgment':4},
  'extended':{'train_cases':12,'validation_rows_per_shared_judgment':12},
  'val_judgment_ids':vids,'validation_indices':vp,
  'metric':'seven-class unweighted macro-F1 including unsupported classes with zero_division=0; pooled over chosen sentences; support and per-case paired estimates retained',
  'algorithm':'Fixed decision-row ridge residual; centered native seven logits; positive class margin 4; lambda 0.15 times mean row squared norm; inverse-square-root class weights scaled to mean 1; all 248 non-role code rows untouched; float32 solve rounded to checkpoint bf16.',
  'internal_edit':'Independent last feed-forward weight edit model.layers.27.mlp.Wo.weight. Fit an additive rank-at-most-six map from the final MLP activation to residual class-logit errors using closed-form ridge and the pseudoinverse of the centered original readout through final LayerNorm. Native head stays fixed. Same ridge strength, weighting, and target logits as the head edit. Prespecified shuffle control.',
  'controls':['original readout','label-shuffled readout with seed 221126','separate target-only and adjacent-context input arms; each uses its own training features and the same recipe.'],
  'prompt_policy':'Publisher prompt template, fixed seven option order; all supplied sentences use full text up to native cap. context=min(1 neighbor each side), with TARGET mark and no labels; no fit from official val or test.'
 }
 (ROOT/'plan.json').write_text(json.dumps(recipe,indent=2))
 print('pilot train IDs',chosen[:3],'. count',sum(freq[i].total() for i in chosen[:3]),'role support',sum((freq[i] for i in chosen[:3]),Counter()))
 print('val pilot',sum(len(v[:4]) for v in vp.values()), 'extended train',sum(freq[i].total() for i in chosen),flush=True)
if __name__=='__main__': plan()
