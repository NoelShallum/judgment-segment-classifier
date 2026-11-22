"""Prepare a separate, predeclared semantic-verbalizer head diagnostic without teacher labels."""
import hashlib,json
from pathlib import Path
from seven_paths import ROOT,LABELS
from seven_paragraphs import dump
from synthetic_review import DEFINITIONS
base=ROOT/"gemma-projection";out=ROOT/"gemma-semantic";out.mkdir(exist_ok=True);(out/"texts").mkdir(exist_ok=True)
prefix=("<|turn>system\nClassify the supplied English Indian judicial excerpt. Return one JSON object with the field role. "
    "Use the class that explains the excerpt's dominant function. Distinguish a party's submissions from the court's own reasoning. "
    "Choose one exact role name from the following dictionary.\n"+
    "\n".join(f"{k} — {DEFINITIONS[k]}" for k in LABELS)+"<turn|>\n<|turn>user\n")
suffix='<turn|>\n<|turn>model\n{"role":"'
for name,value in (("prefix.txt",prefix),("suffix.txt",suffix),("choices.txt","\n".join(LABELS)+"\n")):
 p=out/name
 if p.exists() and p.read_text()!=value:raise ValueError("Changed frozen semantic prompt")
 p.write_text(value)
rows=[]
for r in json.loads((base/"rows.json").read_text()):
 if r["group"] not in ("neutral","keys","select","confirm"):continue
 t=(base/"texts"/(r["hash"]+".txt")).read_text();key=hashlib.sha256((prefix+"\0"+t+"\0"+suffix).encode()).hexdigest()
 (out/"texts"/(key+".txt")).write_text(t);rows.append({**r,"hash":key})
dump(out/"rows.json",rows);(out/"manifest.txt").write_text("\n".join(dict.fromkeys(r["hash"] for r in rows))+"\n")
spec={"method":"Replace the natural vocabulary head with seven role-name token rows at a fixed JSON answer slot. Evaluate raw projection, content-free bias shifts and analytic authored-key corrections. Fixed candidates match the option-letter trial. Criteria and gold remain unchanged.",
 "gate":"Before accessing additional test results, compare the best actual edited head on 56 balanced development rows to the best observed A–G head (10/56). Advance the semantic variant on the same frozen matched inputs only if it improves development accuracy and has macro-F1 above 0.1171.",
 "data":"Only neutral prompts and authored keys specify weight changes. LegalSeg select/confirm provide independent development/confirmation; provider answers never enter a weight edit."}
dump(out/"design.json",spec)
