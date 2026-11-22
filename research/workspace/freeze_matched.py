"""Materialize the prescreened anonymous inputs locally. No raw judgment text is shipped."""
from collections import Counter
import json
from prepare_matched import DEST, item
from paragraph_transfer import read
from program_trial import save_json
from weight_edit import LABELS

# Screen for personal identification or sensitive case detail before making any model calls.
EXCLUDED={105870,113738,116711,4780,146605,136201,134531,54582}
PARAGRAPHS={
    "Analysis":["D11-4","D21-17","D24-9","D2-9","D8-15"],
    "Fact":["D7-6","D44-2","D44-1","D26-1","D54-4"],
    "Verdict":["D23-10","D20-13","D24-10","D1-14","D26-10"],
    "Argument":["D46-25","D8-6","D10-2","D12-26","D43-5"],
}
REDACTIONS={
 "D10-2":[("Mr Debabrata Mukherjee","counsel")],
 "D12-26":[("Emperor vs Naubat(4)","[precedent]"),("Ramlakhan Chaudhury vs Emperor(1)","[precedent]"),("Emperor vs Atta Muhammad(1)","[precedent]")],
 "D43-5":[("British American Tobacco Company Ltd vs Commissioners of Inland Revenue(3)","[precedent]"),("Mr LG Bash","the proxy holder"),("the Aluminium Ltd","the company"),("(1) [1946] 14 ITR (Suppl.) 7; [1945] I All ER 667; Tax Cas. 167 (2) AIR 1947 Bombay 45; 14 ITR 344.","[citations]")],
 "D8-6":[("(1)[1952] SCR 193.","[citation].")],
}

def main():
    source=json.loads((DEST/"candidates.json").read_text())["candidates"]
    rows=[]
    for role in LABELS:
        chosen=[r for r in source if r["source"]=="sentence" and r["gold"]==role and int(r["id"].split("-")[1]) not in EXCLUDED][:16]
        assert len({r["doc"] for r in chosen})==16
        for r in chosen:rows.append({**r,"transformation":"whitespace normalized; leading enumerator removed when present"})
    originals={f"{doc}-{i}":(doc,t,l) for doc,i,t,l in read("PRR.zip")}
    for label, ids in PARAGRAPHS.items():
        for code in ids:
            doc,t,l=originals[code]
            assert l==label
            for old,new in REDACTIONS.get(code,[]):
                assert old in t,(code,old)
                t=t.replace(old,new)
            t=" ".join(t.split())
            r=item("paragraph",code,doc,t,l)
            rows.append({**r,"transformation":"identifiers or citations replaced" if code in REDACTIONS else "whitespace normalized"})
    assert len({r["id"] for r in rows})==len({r["sha256"] for r in rows})==132
    target=DEST/"approved.json"
    if target.exists() and json.loads(target.read_text())!=rows:raise ValueError("Benchmark already frozen with other inputs")
    save_json(target,rows)
    print("Frozen",len(rows),Counter(r["source"] for r in rows),len({r["doc"] for r in rows}),"judgments")
if __name__=="__main__":main()
