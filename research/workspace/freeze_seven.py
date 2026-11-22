import csv,hashlib,json,re
from pathlib import Path
from seven_paths import ROOT,PILOT
from seven_paragraphs import DEST,dump
def main():
    # Preserve already completed identical seven-role sentence calls; the previous external paragraph corpus is excluded.
    old=ROOT/"gemini-benchmark";PILOT.mkdir(exist_ok=True)
    rows=[r for r in json.loads((old/"approved.json").read_text()) if r["source"]=="sentence"]
    ids={r["id"] for r in rows};assert len(ids)==112
    dump(PILOT/"approved.json",rows)
    dump(PILOT/"answers.json",{k:v for k,v in json.loads((old/"answers.json").read_text()).items() if k in ids})
    (PILOT/"request-schema.json").write_bytes((old/"request-schema.json").read_bytes())
    local=json.loads((old/"local.json").read_text())
    for m in local.values():m["predictions"]={k:v for k,v in m["predictions"].items() if k in ids}
    dump(PILOT/"local.json",local)
    indices={"none":[0,3,4],"facts":[1,2,5],"issue":[0,4,5],"petitioner_argument":[5,6,8],
             "respondent_argument":[0,1,2],"reasoning":[0,3,6],"decision":[0,1,2]}
    data=json.loads((DEST/"candidates.json").read_text())["ranked"];chosen=[]
    for role,ii in indices.items():
        for i in ii:
            r=data[role][i];text=r["text"];original=hashlib.sha256(text.encode()).hexdigest()
            # These transformations apply identically to the API and to all local readers.
            text=re.sub(r"\bRs\.?\s*[\d,]+(?:\.\d+)?/?", "[amount]",text,flags=re.I)
            text=re.sub(r"\b\d{1,2}[/.]\d{1,2}[/.]\d{2,4}\b|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}\b","[date]",text)
            chosen.append({"source":"multi_sentence","id":r["id"],"doc":r["doc"],"text":text,"gold":role,
              "sha256":hashlib.sha256(text.encode()).hexdigest(),"original_sha256":original,
              "transformation":"whitespace normalized; monetary amounts and dates masked when present"})
    assert len({r["id"] for r in chosen})==21
    if (DEST/"approved.json").exists() and json.loads((DEST/"approved.json").read_text())!=chosen:raise ValueError("Frozen inputs changed")
    dump(DEST/"approved.json",chosen)
    print('Matched seven-role sets',len(rows),len(chosen))
if __name__=="__main__":main()
