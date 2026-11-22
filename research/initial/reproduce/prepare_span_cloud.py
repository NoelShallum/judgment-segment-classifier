"""Prescreen contiguous held-out LegalSeg rows for paragraph and document-boundary evaluation."""
from collections import Counter,defaultdict
import csv,hashlib,json,random,re
from seven_paths import PILOT,ROOT
from seven_paragraphs import SENSITIVE,SAFE_CAPS,ROLES,dump
def clean(t):
    t=' '.join(t.split());words=t.split()
    if not 5<=len(words)<=62 or SENSITIVE.search(t):return None
    if re.search(r"[0-9@/\\]|\b(?:Mr|Mrs|Shri|Smt|Justice|village|email|resid\w*|named|address|sexual|hospital|suicide|deceased|substance abuse|abuse|prosecutrix|burn\w*|prosecuted|caste|religio\w*|alcohol|illtreat\w*)\b|\b[a-z]+[A-Z]",t,re.I if False else 0):return None
    if any(w not in SAFE_CAPS for w in re.findall(r"\b[A-Z][A-Za-z'-]*",t)):return None
    return t
def main():
    old={r["doc"] for q in (PILOT,ROOT/"gemini-confirmation",ROOT/"gemini-seven-paragraphs") for r in json.loads((q/"approved.json").read_text())}
    doclist=defaultdict(list)
    with (ROOT/"references/official/test.csv").open(encoding="utf-8-sig",newline="") as f:
        for i,r in enumerate(csv.DictReader(f)):
            if r["Index"] not in old:doclist[r["Index"]].append((i,clean(r["Text"] or ""),ROLES[r["Label"]]))
    pools=defaultdict(list);mixed=[]
    for doc,rows in doclist.items():
        for j in range(len(rows)-1):
            for count in (2,3,5,6):
                group=rows[j:j+count]
                if len(group)!=count or not all(t for _,t,_ in group) or len({i for i,_,_ in group})!=count:continue
                roles=[y for _,_,y in group];nr=len(set(roles));length=sum(len(t.split()) for _,t,_ in group)
                if length<20 or length>230:continue
                block={"doc":doc,"rows":[{"row":i,"text":t,"gold":role} for i,t,role in group]}
                if count in (2,3) and nr==1:pools[roles[0]].append(block)
                if count>=5 and nr>=2:mixed.append(block)
    rng=random.Random(796202);seen=set();out={}
    def pick(pool,quota):
        rng.shuffle(pool);chosen=[]
        for b in pool:
            if b["doc"] in seen:continue
            seen.add(b["doc"]);chosen.append(b)
            if len(chosen)==quota:break
        return chosen
    # Scarce uniform-role sections are chosen first to maintain coverage.
    for role,g in sorted(pools.items(),key=lambda kv:len(kv[1])):out[role]=pick(g,5)
    out["mixed"]=pick(mixed,22)
    f=ROOT/"seven-spans";f.mkdir(exist_ok=True)
    dump(f/"review-pool.json",out)
    print("Available",[(k,len(g)) for k,g in pools.items()],"mixed",len(mixed),"draw",[(k,len(g)) for k,g in out.items()])
if __name__=="__main__":main()
