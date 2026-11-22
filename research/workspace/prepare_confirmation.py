"""Choose one anonymous excerpt per newly sampled judgment without looking at its label."""
import csv
import json
import random
import re
from collections import Counter,defaultdict
from prepare_matched import DEST,allow,item
from program_trial import TO_ID,save_json
from weight_edit import LABELS,ROOT

SENSITIVE=re.compile(r"\b(?:wife|husband|marriage|married|divorce|children|child|father|mother|brother|sister|pregnan\w*|sex\w*|illness|injur\w*|assault\w*|deat\w*|kill\w*|stab\w*|weapon\w*|victim\w*|handicap\w*|patient\w*|caste\w*|religio\w*|body|bodies|blood|wound\w*|victim\w*|convict\w*|imprison\w*|arrest\w*|accused|prosecutrix|widow\w*|confess\w*|disab\w*|rap\w*|violence|narcotic\w*|bribe\w*|died|punish\w*|testament\w*|famili\w*|family|illtreatment|accident\w*|residence|apprehended|kidnap\w*|suffer\w*|kidney|dementia|pensioner|suicid\w*)\b",re.I)
def main():
    pilot=json.loads((DEST/"approved.json").read_text())
    docs={r["doc"] for r in pilot if r["source"]=="sentence"}
    texts={r["sha256"] for r in pilot}
    pool=defaultdict(list)
    with (ROOT/"references/official/test.csv").open(encoding="utf-8-sig",newline="") as f:
        for i,r in enumerate(csv.DictReader(f)):
            if r["Index"] in docs:continue
            t=allow(r["Text"])
            if not t or SENSITIVE.search(t) or re.search(r"\b[a-z]+[A-Z]",t):continue
            v=item("sentence",i,r["Index"],t,LABELS[TO_ID[r["Label"]]])
            if v["sha256"] not in texts:pool[r["Index"]].append(v)
    rng=random.Random(118220971);population=list(sorted(pool));rng.shuffle(population)
    candidates=[]
    for doc in population:
        rng.shuffle(pool[doc])
        r=next((r for r in pool[doc] if r["sha256"] not in texts),None)
        if r:
            texts.add(r["sha256"]);candidates.append(r)
        if len(candidates)==210:break
    target=ROOT/"gemini-confirmation";target.mkdir(exist_ok=True)
    save_json(target/"candidates.json",{"eligible_cases":len(pool),"eligible_rows":sum(map(len,pool.values())),
        "seed":118220971,"candidates":candidates})
    print(len(candidates),"privacy review candidates in",len(pool),"eligible unseen cases",flush=True)
if __name__=="__main__":main()
