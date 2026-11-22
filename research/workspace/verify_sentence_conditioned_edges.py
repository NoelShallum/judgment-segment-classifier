"""Verify the frozen contextual transition edit against the portable NumPy decoder."""
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np

from sentence_conditioned_edges import (
    DEST,DONOR,EDGE,PARENT,ROOT,HierModel,aligned_embeddings,cm_case,document,edge_values,
    file_hash,gold_cases,load_decoder,load_file,mapped,package,paired,row_summary,save_json,torch,
)
from role_edge_runtime import decode


def verify(docs):
    torch.set_num_threads(1)
    model=HierModel(DONOR)
    weights=load_file(str(DEST))
    parent=load_file(str(PARENT))
    mode,source,gain=weights["settings"]
    mode="interaction" if mode else "stay"
    vocab=json.loads((EDGE/"vocab.json").read_text())
    bank,x=package(docs,aligned_embeddings(model,docs,"test"),model,parent,vocab,load_decoder())
    edges=edge_values(mapped(x,weights),weights,mode,int(source))*gain
    current=[]
    prior=[]
    for a,b in zip(bank.sizes,bank.sizes[1:]):
        z=np.zeros((b-a-1,7,7),np.float32)
        prior.append(decode(bank.logits[a:b],bank.transition,z,0.))
        current.append(decode(bank.logits[a:b],bank.transition,edges[a+1:b],1.))
    a,b=np.concatenate(current),np.concatenate(prior)
    return [d["id"] for d in docs],cm_case(bank,a),cm_case(bank,b),int((a!=b).sum())


def main():
    root=ROOT/"results"
    freeze=root/"sentence-conditioned-edges-frozen.json"
    f=json.loads(freeze.read_text())
    assert f["parent_sha256"]==file_hash(PARENT) and f["weights_sha256"]==file_hash(DEST)
    cases=[document(k,v) for k,v in gold_cases().items()]
    with ProcessPoolExecutor(4) as pool:
        parts=list(pool.map(verify,[cases[i::4] for i in range(4)]))
    index={k:i for i,k in enumerate(q for p in parts for q in p[0])}
    order=[index[d["id"]] for d in cases]
    a,b=(np.concatenate([shard[i] for shard in parts])[order] for i in (1,2))
    archive=root/"known-sentence-numerical-archives.npz"
    details=json.loads((root/"known-sentence-numerical-archives.json").read_text())
    assert file_hash(archive)==details["score_file_sha256"]
    name=details["best_by_macro_f1"]
    with np.load(archive,allow_pickle=False) as data:
        np.testing.assert_array_equal(data["cases"],[d["id"] for d in cases])
        c=data[name]
    results={"cases":len(a),"sentences":int(a.sum()),"changed":sum(q[3] for q in parts),
             "candidate":row_summary(a.sum(0)),"parent":row_summary(b.sum(0)),
             "aligned_archive":row_summary(c.sum(0)),"paired":paired(a,b),
             "paired_to_aligned_archive":paired(a,c),"aligned_archive_name":name,
             "freeze_sha256":file_hash(freeze)}
    assert results["parent"]==json.loads((root/"sentence-joint-qa.json").read_text())["scores"]
    record=root/"sentence-conditioned-edges.json"
    if record.is_file() and "test" in (r:=json.loads(record.read_text()))["splits"]:
        for k in ("cases","sentences","changed","candidate","parent","paired"):
            assert results[k]==r["splits"]["test"][k],k
    save_json(root/"sentence-conditioned-edges-qa.json",results)
    print("Verified conditioned edges",len(a),results["changed"],results["candidate"]["macro_f1"],flush=True)


if __name__=="__main__":
    main()
