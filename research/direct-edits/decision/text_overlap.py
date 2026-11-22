"""Count literal text reuse between the chosen case-disjoint public training and validation folds."""
import hashlib,json
from collections import defaultdict
from fetch import ROOT
from cohort import case_rows
def norm(s):return ' '.join(s.casefold().split())
def fh(s):return hashlib.sha256(s.encode()).hexdigest()
p=json.loads((ROOT/'plan.json').read_text());v=case_rows('val',set(p['val_judgment_ids']));out={}
for name,tr_ids in [('pilot3',p['fit_judgment_ids'][:3]),('packed12',p['fit_judgment_ids'])]:
 t=case_rows('train',set(tr_ids));known=defaultdict(list);doc_hash={}
 for cid,doc in t.items():
  for i,(s,l) in enumerate(doc):known[norm(s)].append((cid,i))
  doc_hash[fh('\n'.join(norm(s) for s,_ in doc))]=cid
 repeats=[]
 for cid,doc in v.items():
  for i,(s,_) in enumerate(doc):
   if norm(s) in known:repeats.append({'validation_case':cid,'position':i,'text_sha256':fh(norm(s)),'training_positions':known[norm(s)],'in_sampled64':i in p['validation_indices'][cid][:4]})
 out[name]={'train_case_count':len(t),'validation_case_count':len(v),'case_id_overlap':len(set(t)&set(v)),'whole_case_text_reuse':sum(fh('\n'.join(norm(s) for s,_ in d)) in doc_hash for d in v.values()),
    'validation_rows_matching_any_training_text':len(repeats),'sampled_64_matching':sum(z['in_sampled64'] for z in repeats),'validation_rows':sum(len(d) for d in v.values()),'matching_positions':repeats}
(ROOT/'text-overlap.json').write_text(json.dumps(out,indent=2))
print([(k,c['validation_rows_matching_any_training_text'],c['sampled_64_matching'],c['whole_case_text_reuse']) for k,c in out.items()])
