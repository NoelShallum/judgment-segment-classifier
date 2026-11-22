import csv, hashlib, json
from collections import Counter
from pathlib import Path
p=Path(__file__).resolve().parent
src=p.parents[1]/'references/official/train.csv'
with src.open() as f: counts=Counter(r['Index'] for r in csv.DictReader(f) if r['Text'].strip())
order=sorted((k for k,v in counts.items() if 35<=v<=135),key=lambda k:hashlib.sha256(('atto-gliner-boundary-nov22-v1:'+k).encode()).digest())
share=json.loads((p.parent/'shared-confirmation.json').read_text())['small_validation_judgment_ids']
obj=dict(checkpoint='fastino/gliner2.5-small-v1',revision='b5d42bf7dc7d01c40304ae4ed43c2f13c325a75b',schema=['other judgment material','facts','issue','petitioner argument','respondent argument','court reasoning','decision'],roles=['None','Facts','Issue','Arguments of Petitioner','Arguments of Respondent','Reasoning','Decision'],selection='35-135 sentence range inclusive (35 <= nonblank rows <= 135), SHA256 of salt plus judgment ID only; first 12 fit, next 5 select, next 5 internal check',fit=order[:12],select=order[12:17],check=order[17:22],rows={k:counts[k] for k in order[:22]},validation=share,interim=share[:4],primary='Sentence boundary constrained seven-way classification, same native start/end/inside logits before and after named analytic writes; merge adjacent equal predicted roles within judgment for exact labeled spans, including None. Split overly long sentences into pieces, aggregate by native word counts; every nonblank row scored.',edit='At most three existing state projection weight matrices of native boundary query head. Closed-form ridge on empirical query directions and target marginals. Choose among channels and regularization on select only, include unchanged. Equal-norm shuffled-label control with identical analytic recipe. Confirm unchanged, edited, shuffled, incumbent against identical positions, no test use.',metric='macro F1 over all seven classes, accuracy, exact labeled run micro F1, case-level rows and scores')
(p/'protocol.json').write_text(json.dumps(obj,indent=2))
print(obj['fit'],sum(counts[k] for k in obj['fit']),obj['select'],sum(counts[k] for k in obj['select']),obj['check'],sum(counts[k] for k in obj['check']))
