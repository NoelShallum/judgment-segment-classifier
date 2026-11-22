"""Calculate the exact corpus F1 ceiling for the local span replacement family."""
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

import numpy as np

from hier_full import CACHE
from novel_span_repair import bank, choose, compress, engine
from role_edge_interpolation import ROOT, document, gold_cases, save_json, torch
from role_pair_weight_search import prepare_score, transition_scores
from short_span_weight_edit import setup


def pack(docs):
    torch.set_num_threads(1)
    parents,vocab,_,incumbent=setup()
    decoder=engine()
    counts=np.zeros(3,np.int64)
    results=[]
    for d in docs:
        with np.load(CACHE/(d["id"]+".npz")) as arrays:
            item=prepare_score(d,arrays["raw"],parents)
        e=transition_scores(d["text"],d["units"],vocab,parents["exact"])
        r=bank(item,e,parents,incumbent,decoder)
        counts+=item.score(r.base)[49:52]
        results.append(SimpleNamespace(run_count=r.run_count,groups=r.groups.astype(np.int32),
            tier=r.tier,gain=r.gain.astype(np.int8),count=r.count.astype(np.int8)))
    return counts,results


def max_f1(cases,base,scope):
    hits,predicted,gold=map(int,base)
    score=2*hits/(predicted+gold)
    for step in range(30):
        delta=np.zeros(3,np.int64)
        for r in cases:
            u=r.gain-score*.5*r.count
            ids=choose(r,compress(r,u,scope),0.)
            delta+=(r.gain[ids].sum(),r.count[ids].sum(),len(ids))
        achieved=2*(hits+delta[0])/(gold+predicted+delta[1])
        if abs(achieved-score)<1e-13:
            return {"max_f1":achieved,"oracle_edits":int(delta[2]),"matched":int(hits+delta[0]),
                    "predicted":int(predicted+delta[1]),"gold":gold,"iterations":step+1}
        score=achieved
    raise ValueError("Local-edit oracle did not converge")


def main():
    docs=[document(k,r) for k,r in gold_cases().items()]
    with ProcessPoolExecutor(4) as pool:
        batch=list(pool.map(pack,[docs[i::4] for i in range(4)]))
    base=sum(q[0] for q in batch)
    cases=[r for q in batch for r in q[1]]
    out={}
    for scope in ("short","all"):
        out[scope]=max_f1(cases,base,scope)
        print(scope,out[scope],flush=True)
    save_json(ROOT/"results/novel-span-repair-audit.json",
             {"cases":len(docs),"oracle":out,"baseline_counts":base.tolist(),
              "protocol":"Gold-assisted exact pooled F1 optimization over nonadjacent local replacements. For each trial F1, maximize additional exact matches minus half that F1 times the change in prediction count; repeat until convergence."})


if __name__=="__main__":
    main()
