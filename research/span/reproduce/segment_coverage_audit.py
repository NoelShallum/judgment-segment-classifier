"""Attribute missing gold runs to raw units, proposed offsets, or candidate roles."""
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np

from role_edge_interpolation import (
    EDGE, ROOT, document, gold_cases, load_file, save_json, torch,
)
from hier_full import CACHE
from hier_model import LABELS
from role_pair_weight_search import prepare_score, transition_scores
from segment_span_runtime import span_features
from segment_weight_write import engine
from span_posterior_decode import spans


def audit_rows(docs):
    torch.set_num_threads(1)
    parents={k:load_file(str(EDGE/f"role-edge-{k}.safetensors")) for k in ("exact","balanced")}
    vocab=json.loads((EDGE/"vocab.json").read_text())
    model=load_file(str(ROOT/"checkpoints/hierarchical/whole-segment-weights.safetensors"))
    scores=np.zeros((2,5,7),np.int64)
    intervals=[[],[]]
    for d in docs:
        with np.load(CACHE/(d["id"]+".npz")) as r:
            item=prepare_score(d,r["raw"],parents)
        edge=transition_scores(d["text"],d["units"],vocab,parents["exact"])
        args=(item.logits,edge,parents,item.token_role_counts.sum(1),model["proposals"],float(model["temperature"]))
        offsets=item.offsets
        units=set(offsets.tolist())
        n=len(offsets)-1
        for v in (0,1):
            arcs,*_=span_features(*args,posterior_ends=bool(v))
            a,b,k=arcs.T
            ids={(int(offsets[l]),int(offsets[r]),int(c)) for l,r,c in arcs}
            geometry={(l,r) for l,r,c in ids}
            for truth in d["gold_spans"]:
                l,r=truth["start"],truth["end"]
                i=LABELS.index(truth["label"])
                bucket=0 if l not in units or r not in units else 1 if (l,r) not in geometry else 2 if (l,r,i) not in ids else 3
                scores[v,bucket,i]+=1
                scores[v,4,i]+=1
            hit=(offsets[b]==item.gold_end[a])&(k==item.gold_label[a])
            dim=(n,n+1,7)
            keys=np.ravel_multi_index(arcs.T,dim)
            intervals[v].append((arcs,keys[hit],hit,n))
    return scores,intervals


def maximum_f1(rows,gold,decode):
    f1=0.
    for iteration in range(30):
        matches=predicted=0
        for arcs,keys,hit,n in rows:
            path=spans(decode(arcs,hit-f1*.5,n))
            find=np.ravel_multi_index(path.T,(n,n+1,7))
            matches+=int(np.isin(find,keys,assume_unique=True).sum())
            predicted+=len(path)
        next_=2*matches/(gold+predicted)
        if abs(next_-f1)<1e-13:
            return {"f1":next_,"matched":matches,"predicted":predicted,"gold":int(gold),"iterations":iteration+1}
        f1=next_
    raise RuntimeError("Oracle ratio did not converge")


def main():
    decode=engine()
    docs=[document(k,r) for k,r in gold_cases().items()]
    with ProcessPoolExecutor(4) as pool:
        reports=list(pool.map(audit_rows,[docs[i::4] for i in range(4)]))
    total=sum(x[0] for x in reports)
    names=("missing_raw_endpoint","offset_pair_missing_from_candidates","role_missing_for_present_offset_pair","exact_span_present","gold")
    records={}
    for kind,name in enumerate(("recombined","posterior_endpoints")):
        counts=total[kind]
        assert np.array_equal(counts[:4].sum(0),counts[4])
        rows=[r for _,docs_ in reports for r in docs_[kind]]
        combined={k:int(v) for k,v in zip(names,counts.sum(1))}
        records[name]={"counts":combined,"roles":{class_:{k:int(v) for k,v in zip(names,col)} for class_,col in zip(LABELS,counts.T)},
                       "exact_corpus_f1_upper_bound":maximum_f1(rows,combined["gold"],decode)}
        print(name,records[name]["counts"],records[name]["exact_corpus_f1_upper_bound"],flush=True)
    save_json(ROOT/"results/segment-coverage-audit.json",{"cases":len(docs),"candidates":records,
       "protocol":"Attribute gold spans to the first stage that cannot express them. Compute the exact maximum corpus F1 under each frozen candidate set and non-overlapping tiling constraint, using human annotations for analysis. The optimizer solves an additive path problem with cost equal to half the current F1 and repeats until convergence."})


if __name__=="__main__":
    main()
