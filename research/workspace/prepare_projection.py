"""Create local-only inference inputs in a deterministic order, with the role rubric frozen."""
from pathlib import Path
import hashlib
import json
from seven_paths import PILOT as DEST
from program_trial import official_sample, save_json
from role_fixtures import ANCHORS
from synthetic_review import DEFINITIONS
from weight_edit import LABELS, ROOT

TARGET=ROOT/"gemma-projection"
SYSTEM=("Assign one dominant rhetorical role to the English excerpt from an Indian court judgment. "
        "Choose the single best letter. Identify whose voice is speaking. A party's prayer is an argument. "
        "A lower court's earlier order can be procedural history. The current court's operative order is a decision.\n"+
        "\n".join(f"{chr(65+j)}. {label.replace('_',' ')} — {DEFINITIONS[label]}" for j,label in enumerate(LABELS)))
PREFIX=f"<|turn>system\n{SYSTEM}<turn|>\n<|turn>user\n"
SUFFIX="<turn|>\n<|turn>model\n"

def main():
    TARGET.mkdir(exist_ok=True);(TARGET/"texts").mkdir(exist_ok=True)
    def put(name,value):
        p=TARGET/name
        if p.is_file() and p.read_text()!=value:raise ValueError("Frozen input changed: "+name)
        p.write_text(value)
    put("prefix.txt",PREFIX);put("suffix.txt",SUFFIX)
    rows=[]
    def add(g,i,t,label=None,doc=None):
        h=hashlib.sha256((PREFIX+"\0"+t+"\0"+SUFFIX).encode()).hexdigest()
        put("texts/"+h+".txt",t)
        rows.append({"group":g,"id":i,"hash":h,"label":label,"doc":doc})
    for j,s in enumerate(("","N/A","…","[excerpt omitted]","Text.")):add("neutral",str(j),s)
    for role in LABELS:
        for j,s in enumerate(ANCHORS[role][:3]):add("keys",role+str(j),s,role)
    for split in ("select","confirm"):
        items,_,_=official_sample(split,8)
        for i,doc,t,y in items:add(split,str(i),t,LABELS[y],doc)
    for r in json.loads((DEST/"approved.json").read_text()):add("matched",r["id"],r["text"],r["gold"],r["doc"])
    put("manifest.txt","\n".join(dict.fromkeys(r["hash"] for r in rows))+"\n")
    save_json(TARGET/"rows.json",rows)
    print("Frozen",len(rows),"rows, including disjoint development and confirmation groups",flush=True)
if __name__=="__main__":main()
