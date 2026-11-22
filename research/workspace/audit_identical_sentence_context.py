"""Measure unavoidable ambiguity for equal inputs under fixed LegalSeg context budgets."""
from collections import Counter,defaultdict
import csv
from hashlib import file_digest,sha256
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/"references/official/test.csv"


def digest_file(path):
    with path.open("rb") as stream:
        return file_digest(stream,"sha256").hexdigest()


def buckets(docs,radius,fold_case=False):
    data=defaultdict(Counter)
    for rows in docs.values():
        labels=[r[1] for r in rows]
        parts=[r[0].casefold() for r in rows] if fold_case else [r[0] for r in rows]
        sig=[sha256(text.encode()).digest() for text in parts]
        whole=sha256(b"".join(sig)).digest() if radius is None else None
        for i,label in enumerate(labels):
            if radius is None:
                key=(whole,i)
            else:
                lhs=tuple(sig[max(0,i-radius):i])
                rhs=tuple(sig[i+1:i+radius+1])
                key=(lhs,sig[i],rhs)
            data[key][label]+=1
    rows=sum(sum(c.values()) for c in data.values())
    conflict=[c for c in data.values() if len(c)>1]
    ambiguous=sum(sum(c.values()) for c in conflict)
    errors=sum(sum(c.values())-max(c.values()) for c in conflict)
    pairs=Counter()
    for c in conflict:
        roles=sorted(c)
        for i,a in enumerate(roles):
            for b in roles[i+1:]:
                pairs[(a,b)]+=1
    return {"rows":rows,"distinct_inputs":len(data),"conflicting_inputs":len(conflict),
            "rows_in_conflicting_inputs":ambiguous,"minimum_errors":errors,"accuracy_ceiling":1-errors/rows,
            "most_frequent_role_conflicts":[{"roles":pair,"inputs":n} for pair,n in pairs.most_common(7)]}


def main():
    docs=defaultdict(list)
    with SOURCE.open(encoding="utf-8-sig",newline="") as f:
        for row in csv.DictReader(f):
            if text:=row["Text"].strip():
                docs[row["Index"]].append((text,row["Label"]))
    n=sum(len(rows) for rows in docs.values())
    assert len(docs)==712 and n==149868
    out={"source_sha256":digest_file(SOURCE),"judgments":len(docs),"rows":n,
         "definition":"Exact stripped input text, with the target and neighbor positions preserved. Radius uses available sentences in the same judgment.",
         "modes":{}}
    for key,radius,fold in (("isolated sentence, exact",0,False),("isolated sentence, casefolded",0,True),
                            ("one neighbor per side",1,False),("two neighbors per side",2,False),
                            ("full judgment with target position",None,False)):
        out["modes"][key]=buckets(docs,radius,fold)
        print(key,out["modes"][key],flush=True)
    (ROOT/"results/sentence-identical-input-audit.json").write_text(json.dumps(out,indent=2)+"\n")


if __name__=="__main__":
    main()
