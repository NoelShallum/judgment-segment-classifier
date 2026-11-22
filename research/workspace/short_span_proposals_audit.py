"""Measure the recall and valid-tiling ceiling of complete short-span proposals."""
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np

from role_edge_interpolation import EDGE, ROOT, document, gold_cases, load_file, save_json, torch
from hier_full import CACHE
from hier_model import CRF
from role_pair_weight_search import prepare_score, transition_scores
from segment_coverage_audit import maximum_f1
from segment_span_runtime import SHORT_SETTINGS, short_arcs, span_features
from segment_weight_write import engine
from span_posterior_decode import posterior

def audit(docs):
    torch.set_num_threads(1)
    profiles={k:load_file(str(EDGE/f"role-edge-{k}.safetensors")) for k in ("exact","balanced")}
    weights=load_file(str(ROOT/"checkpoints/hierarchical/whole-segment-weights.safetensors"))
    vocab=json.loads((EDGE/"vocab.json").read_text())
    result=[[] for _ in SHORT_SETTINGS]
    gold_total=0
    for d in docs:
        with np.load(CACHE/(d["id"]+".npz")) as z:
            item=prepare_score(d,z["raw"],profiles)
        edge=transition_scores(d["text"],d["units"],vocab,profiles["exact"])
        temp=float(weights["temperature"])
        base,*_=span_features(item.logits,edge,profiles,item.token_role_counts.sum(1),
                              weights["proposals"],temp,True)
        probability=posterior(item.logits["exact"],profiles["exact"][CRF],.2*edge,temp)
        gold_total+=int(item.truth_run_counts.sum())
        n=len(item.offsets)-1
        for ix,(width,top) in enumerate(SHORT_SETTINGS):
            extra=short_arcs(n,probability,width,top)
            bx=np.unique(np.vstack((base,extra)),axis=0)
            arcs=np.ascontiguousarray(bx[np.lexsort((bx[:,2],bx[:,0],bx[:,1]))],dtype=np.int32)
            a,b,k=arcs.T
            hit=(item.offsets[b]==item.gold_end[a])&(k==item.gold_label[a])
            dim=(n,n+1,7)
            ids=np.ravel_multi_index(arcs[hit].T,dim)
            result[ix].append((arcs,ids,hit,n))
    return result,gold_total


def main():
    decode=engine()
    docs=[document(k,r) for k,r in gold_cases().items()]
    with ProcessPoolExecutor(4) as pool:
        parts=list(pool.map(audit,[docs[i::4] for i in range(4)]))
    gold=sum(g for _,g in parts)
    output={}
    for ix,(width,top) in enumerate(SHORT_SETTINGS):
        rows=[r for x,_ in parts for r in x[ix]]
        total=sum(len(r[0]) for r in rows)
        hit=sum(len(r[1]) for r in rows)
        output[f"{width}:{top}"]={"max_units":width,"roles_per_interval":top,"candidates":total,
                                  "covered_gold":hit,"recall":hit/gold,"oracle":maximum_f1(rows,gold,decode)}
        print(output[f"{width}:{top}"],flush=True)
    save_json(ROOT/"results/short-span-proposals-audit.json",{"cases":len(docs),"runs":output,"gold":gold,
       "protocol":"Expand the prior posterior candidate set using every raw interval through a bounded length. Keep the two or three role values with the highest frozen exact-span posterior, regardless of whether Viterbi proposed the interval. Use annotations for coverage and exact oracle analysis only."})


if __name__=="__main__":
    main()
