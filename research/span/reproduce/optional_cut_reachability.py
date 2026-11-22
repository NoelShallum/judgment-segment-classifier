"""Audit exact-span reachability of the ranked optional cut pool."""
from collections import Counter
import json

from safetensors.numpy import load_file

from optional_boundary_study import PATCH,RESULT,document,gold_cases,save_json
from optional_boundary_runtime import select_cuts


def cover(d,points):
    hit=Counter()
    missing=0
    end=0
    for s in d["gold_spans"]:
        if {s["start"],s["end"]}<=points:
            hit[s["label"]]+=1
            missing+=int(end<s["start"])
            end=s["end"]
    missing+=int(end<len(d["text"]))
    return hit,missing


def main():
    p=load_file(str(PATCH/"prior.safetensors"))
    vocab=json.loads((PATCH/"vocab.json").read_text())
    all_roles=Counter()
    totals={k:Counter() for k in ("original","optional")}
    gaps=Counter()
    n=0
    for case,rows in gold_cases().items():
        d=document(case,rows)
        text=d["text"]
        cuts={a for a,_ in d["units"]}|{len(text)}
        extra={c for _,c,_ in select_cuts(text,d["units"],vocab,p)}
        all_roles.update(s["label"] for s in d["gold_spans"])
        for name,offsets in (("original",cuts),("optional",cuts|extra)):
            hits,missing=cover(d,offsets)
            totals[name].update(hits)
            gaps[name]+=missing
        n+=1
        if n%180==0:print("Coverage",n,flush=True)
    result={"cases":n,"gold_roles":dict(all_roles),"sets":{}}
    for key,v in totals.items():
        hits=v.total()
        result["sets"][key]={"reachable":hits,"reachable_by_role":dict(v),"extra_gap_runs":gaps[key],
                             "oracle_exact_span_f1":2*hits/(all_roles.total()+hits+gaps[key])}
    save_json(RESULT/"optional-boundary-reachability.json",result)
    print(result,flush=True)


if __name__=="__main__":
    main()
