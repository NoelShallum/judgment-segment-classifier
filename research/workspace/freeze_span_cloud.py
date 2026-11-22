from collections import Counter
from pathlib import Path
import hashlib,json
from seven_paths import ROOT,LABELS
from seven_paragraphs import dump
s=ROOT/"seven-spans";pool=json.loads((s/"review-pool.json").read_text());dest=ROOT/"gemini-seven-joined";dest.mkdir(exist_ok=True)
indices={k:[0,1,2,3] for k in LABELS}
indices.update(facts=[0,2,3,4],none=[0,1,3,4],respondent_argument=[0,1,3,4])
rows=[]
for k,nums in indices.items():
 for n in nums:
  b=pool[k][n];assert all(r["gold"]==k for r in b["rows"])
  t=" ".join(r["text"] for r in b["rows"])
  ids=[r["row"] for r in b["rows"]];assert all(j==i+1 for i,j in zip(ids,ids[1:]))
  rows.append({"id":"joined-"+str(ids[0])+"-"+str(ids[-1]),"source":"homogeneous","doc":b["doc"],"gold":k,"source_rows":ids,
               "text":t,"sha256":hashlib.sha256(t.encode()).hexdigest(),"transformation":"joined adjacent unreordered rows from the same judgment; every source label agrees"})
dump(dest/"approved.json",rows)
mixed=[pool["mixed"][i] for i in (0,2,3,4,5,6,7,9,10,11,12,15)]
assert len({b["doc"] for b in mixed})==len(mixed)
dump(s/"cloud.json",mixed)
print("Seven-role joined paragraphs",len(rows),"unmarked public passages",len(mixed),"constituent rows",sum(len(d["rows"]) for d in mixed))
