"""Apply a blind privacy review to a previously sampled list. Input selection ignores gold."""
import json
from prepare_matched import DEST
from program_trial import save_json
from weight_edit import ROOT
REJECTED={33,41,62,65,114,131,141,152,161,170,204}
folder=ROOT/"gemini-confirmation"
data=json.loads((folder/"candidates.json").read_text())
rows=[{**r,"transformation":"whitespace normalized; leading enumerator removed when present"} for i,r in enumerate(data["candidates"]) if i not in REJECTED][:196]
assert len(rows)==len(set(r["doc"] for r in rows))==196
old=json.loads((DEST/"approved.json").read_text())
assert not {r["doc"] for r in rows}&{r["doc"] for r in old}
assert not {r["sha256"] for r in rows}&{r["sha256"] for r in old}
dest=folder/"approved.json"
if dest.exists() and json.loads(dest.read_text())!=rows:raise ValueError("Confirmation already frozen")
save_json(dest,rows)
print("Frozen",len(rows),"distinct judgments; previously excluded",len(REJECTED),"candidate passages during blind privacy review",flush=True)
