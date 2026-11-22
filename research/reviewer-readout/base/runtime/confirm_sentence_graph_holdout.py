"""Compare frozen sentence readouts on validation judgments held out from earlier checks."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from hashlib import sha256
import json

import numpy as np

from compare_sentence_document_graph import run
from sentence_document_graph_write import (
    EDGE,OUT,PARENT,ROOT,STEM,document,file_hash,paired,row_summary,save_json,
)
from program_trial import TO_ID

TAG="sentence-graph-holdout"
SIZE=384


def gather_validation(path):
    records=defaultdict(list)
    with path.open(encoding="utf-8-sig",newline="") as f:
        for index,row in enumerate(csv.DictReader(f)):
            records[row["Index"]].append((index,row["Text"],TO_ID[row["Label"]]))
    result={}
    for key,rows in records.items():
        parts=[text.strip() for _,text,_ in rows if text.strip()]
        result[key]=(rows,len(parts),sha256(" ".join(parts).encode()).hexdigest())
    return result


def previous_ids(value,val_ids,field=""):
    context=("case" in field or "doc" in field or field in ("id","ids","judgment","judgments"))
    if isinstance(value,dict):
        for k,v in value.items():
            if k in val_ids:
                yield k
            yield from previous_ids(v,val_ids,k.lower())
    elif isinstance(value,list) and (context or any(isinstance(q,(list,dict)) for q in value)):
        for item in value:
            yield from previous_ids(item,val_ids,field)
    elif context and isinstance(value,(int,str)) and str(value) in val_ids:
        yield str(value)


def choose(records):
    known=set()
    for name in ("known-sentence-pair-edit","hier-emitter-edits-v1","whole-joint-states"):
        path=ROOT/"cache"/name
        if path.exists():
            known|={p.stem.rsplit("-",1)[-1] for p in path.glob("*.npy")}
    paths=sorted(set((ROOT/"results").glob("*.json"))|set(EDGE.glob("*.json")))
    for p in paths:
        if p.name.startswith(TAG):
            continue
        with p.open() as f:
            known.update(previous_ids(json.load(f),records))
    known&=set(records)
    duplicate_check={}
    for k,(_,n,h) in records.items():
        duplicate_check[h]=duplicate_check.get(h,0)+1
    usable=[(n,k) for k,(_,n,h) in records.items() if k not in known and duplicate_check[h]==1 and n>0]
    usable.sort()
    # Select equal samples from four size quartiles of the remaining validation corpus.
    groups=np.array_split(np.asarray([k for n,k in usable]),4)
    selected=[]
    for group in groups:
        prioritized=sorted(group,key=lambda key:sha256((TAG+"-v1|"+key).encode()).digest())
        selected.extend(prioritized[:SIZE//4])
    return [str(k) for k in selected],sorted(known),len(usable),[[int(records[k][1]) for k in (g[0],g[-1])]
                                                              for g in groups if len(g)]


def main():
    folder=ROOT/"results"
    data=ROOT/"references/official/val.csv"
    graph=folder/(STEM+"-frozen.json")
    comparison=folder/(STEM+"-comparison.json")
    f=json.loads(graph.read_text())
    c=json.loads(comparison.read_text())
    assert c["freeze_sha256"]==file_hash(graph)
    assert f["parent_sha256"]==file_hash(PARENT)
    hashes={"local":f["weights_sha256"],**c["comparison_weights_sha256"]}
    for name,digest in hashes.items():
        k="" if name=="local" else "-"+name
        assert file_hash(OUT/(STEM+k+".safetensors"))==digest
    records=gather_validation(data)
    path=folder/(TAG+"-frozen.json")
    invariant={"graph_freeze_sha256":file_hash(graph),"val_csv_sha256":file_hash(data),
               "weights_sha256":hashes,"parent_sha256":file_hash(PARENT),
               "choices":f["choices_by_arm"]}
    if path.is_file():
        plan=json.loads(path.read_text())
        assert all(plan[k]==v for k,v in invariant.items())
    else:
        picked,excluded,eligible,bands=choose(records)
        assert len(picked)==SIZE and set(picked).isdisjoint(excluded)
        plan={**invariant,"cases":picked,"excluded_previous":excluded,"eligible":eligible,
              "case_length_quartiles":bands,"sentences":sum(records[k][1] for k in picked),
              "policy":"Sample 96 judgments from each size quartile. Exclude validation IDs referenced in existing experiment records or hierarchy feature caches and exclude exact duplicate judgments within validation. Existing split audit found zero cross-split duplicates. No labels are used to choose cases."}
        save_json(path,plan)
    assert len(set(plan["cases"]))==len(set(records[k][2] for k in plan["cases"]))==SIZE
    docs=[document(k,records[k][0]) for k in plan["cases"]]
    print("Frozen fresh holdout",len(docs),plan["sentences"],plan["eligible"],flush=True)
    with ProcessPoolExecutor(4) as pool:
        parts=list(pool.map(run,[(docs[i::4],"fresh-holdout",plan["choices"]) for i in range(4)]))
    indices={key:i for i,key in enumerate(k for ids,p,n in parts for k in ids)}
    sequence=[indices[k] for k in plan["cases"]]
    data={k:np.concatenate([p[k] for ids,p,n in parts])[sequence] for k in ("baseline",*plan["choices"])}
    result={"freeze_sha256":file_hash(path),"cases":len(docs),"baseline":row_summary(data["baseline"].sum(0)),"arms":{}}
    for k in plan["choices"]:
        a=data[k]
        q=result["arms"][k]={"scores":row_summary(a.sum(0)),"changed":sum(n[k] for ids,p,n in parts),
            "paired_to_baseline":paired(a,data["baseline"]),"paired_to_local":paired(a,data["local"])}
        assert q["scores"]["rows"]==plan["sentences"]
        print(k,q["scores"]["macro_f1"],q["scores"]["accuracy"],q["paired_to_local"]["macro_f1"],flush=True)
    save_json(folder/(TAG+".json"),result)


if __name__=="__main__":
    main()
