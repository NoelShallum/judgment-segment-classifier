"""Independently replay sentence decisions and isolate the transition contribution."""
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np
from safetensors.numpy import load_file

from sentence_joint_decoding import CHECKPOINT,PARENT,DONOR,EDGE,ROOT,HierModel,aligned_embeddings,cm_case,distribution,document,file_hash,gold_cases,load_decoder,package,paired,row_summary,save_json,torch,utility
from role_edge_runtime import decode


def local(bank,old,rank,values,physical,weight):
    _,_,rate,_=physical["settings"]
    matrix=np.zeros_like(bank.transition)
    matrix[3:,3:]=weight*physical["transition"]
    left=[v[0] for v in zip(bank.sizes,bank.sizes[1:])]
    right=bank.sizes[1:]
    labels=[]
    for a,b in zip(left,right):
        view=values[a:b]
        ref=old[a:b]
        if rate<1.:
            view=np.where((rank[a:b,None]<rate)|(np.arange(7)[None,:]==ref[:,None]),view,-1.e6)
        step=np.zeros((len(view)-1,7,7),np.float32)
        labels.append(decode(view,matrix,step,0.))
    return np.concatenate(labels)


def run(args):
    docs,kind=args
    torch.set_num_threads(1)
    model=HierModel(DONOR)
    parent=load_file(str(PARENT))
    physical=load_file(str(CHECKPOINT))
    vocab=json.loads((EDGE/"vocab.json").read_text())
    states=list(aligned_embeddings(model,docs,"test"))
    p,old,marg,rank,struct,context=package(docs,states,model,parent,vocab,load_decoder())
    read=None if kind=="posterior" else distribution(context if kind=="context" else struct,old,physical)
    macro,retain,rate,trans=physical["settings"]
    score=utility(marg,read,old,macro,physical,retain)
    combined=local(p,old,rank,score,physical,trans)
    plain=local(p,old,rank,score,physical,0.)
    change=combined!=old
    blocks=0;multi=0;paired_change=0
    for a,b in zip(p.sizes,p.sizes[1:]):
        sample=change[a:b]
        at=np.flatnonzero(np.diff(np.r_[False,sample,False]))
        length=at[1::2]-at[::2]
        blocks+=len(length)
        multi+=int((length>1).sum())
        paired_change+=int(np.maximum(length-1,0).sum())
    return [d["id"] for d in docs],cm_case(p,combined),cm_case(p,old),cm_case(p,plain),(int(change.sum()),blocks,multi,paired_change)


def main():
    root=ROOT/"results"
    frozen=root/"sentence-joint-frozen.json"
    f=json.loads(frozen.read_text())
    assert f["source_sha256"]==file_hash(PARENT) and f["weights_sha256"]==file_hash(CHECKPOINT)
    documents=[document(k,r) for k,r in gold_cases().items()]
    with ProcessPoolExecutor(4) as pool:
        chunks=list(pool.map(run,[(documents[i::4],f["choice"][0]) for i in range(4)]))
    ids=[q for shard in chunks for q in shard[0]]
    inverse={k:i for i,k in enumerate(ids)}
    order=[inverse[d["id"]] for d in documents]
    a,b,c=[np.concatenate([q[i] for q in chunks])[order] for i in (1,2,3)]
    n=np.asarray([g[4] for g in chunks]).sum(0)
    counts=dict(zip(("changed","blocks","multi_sentence_blocks","adjacent_changed_pairs"),map(int,n)))
    payload={"cases":len(a),"freeze_sha256":file_hash(frozen),"weights_sha256":file_hash(CHECKPOINT),
             "scores":row_summary(a.sum(0)),"preceding":row_summary(b.sum(0)),
             "without_transition_scores":row_summary(c.sum(0)),"paired_to_preceding":paired(a,b),
             "paired_to_no_transitions":paired(a,c),"change_counts":counts}
    archive=root/"known-sentence-numerical-archives.npz"
    provenance=json.loads((root/"known-sentence-numerical-archives.json").read_text())
    assert file_hash(archive)==provenance["score_file_sha256"]
    name=provenance["best_by_macro_f1"]
    with np.load(archive,allow_pickle=False) as d:
        np.testing.assert_array_equal(d["cases"],[q["id"] for q in documents])
        baseline=d[name]
    payload["aligned_archive_name"]=name
    payload["aligned_archive"]=row_summary(baseline.sum(0))
    payload["paired_to_aligned_archive"]=paired(a,baseline)
    proof=json.loads((root/"sentence-uncertainty-qa.json").read_text())
    assert payload["preceding"]==proof["scores"]
    report=root/"sentence-joint-edit.json"
    if report.is_file() and "test" in (r:=json.loads(report.read_text()))["splits"]:
        assert r["splits"]["test"]["candidate"]==payload["scores"]
    save_json(root/"sentence-joint-qa.json",payload)
    print("Independent NumPy replay",len(a),payload["scores"]["macro_f1"],counts,flush=True)


if __name__=="__main__":
    main()
