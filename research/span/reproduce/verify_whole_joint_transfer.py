"""Replay coherent sentence proposals and subtract the proposal contribution."""
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np

from whole_joint_transfer import (
    CACHE,JOINT,NEW,RAW_STATES,ROOT,bank,bootstrap,document,exact,file_hash,full_score,gold_cases,
    label,load_file,residual,save_json,setup,torch,
)
from span_posterior_decode import best_path,spans


def replay(docs):
    torch.set_num_threads(1)
    parents,vocab,_,base=setup()
    joint=load_file(str(JOINT))
    w=load_file(str(NEW))
    scale,cost,fee,keep=w["settings"]
    a=[];b=[];c=[];use=np.zeros(2,np.int64)
    for d in docs:
        with np.load(CACHE/(d["id"]+".npz")) as f:
            raw=f["raw"]
        state=np.load(RAW_STATES/(d["id"]+".npy"),allow_pickle=False)
        row=bank(d,state,raw,parents,base,joint,vocab)
        read=residual(row,w)
        y=label(row,best_path,read,scale,cost,fee,bool(keep))
        z=label(row,best_path,read,scale,cost,0.,False)
        orig=label(row,best_path,np.zeros(len(read)),0.,float(base["span_cost"]),0.,False)
        a.append(row.item.score(y))
        b.append(row.item.score(orig))
        c.append(row.item.score(z))
        n=len(y)
        key=np.ravel_multi_index(row.arcs.T,(n,n+1,7))
        path=np.ravel_multi_index(spans(y).T,(n,n+1,7))
        matched=np.isin(key,path,assume_unique=True)
        use[0]+=int((matched&row.novel).sum())
        use[1]+=int((matched&row.novel&exact(row)).sum())
    return np.stack(a),np.stack(b),np.stack(c),use


def main():
    root=ROOT/"results"
    freeze=root/"whole-joint-transfer-frozen.json"
    f=json.loads(freeze.read_text())
    assert f["joint_sha256"]==file_hash(JOINT) and f["weights_sha256"]==file_hash(NEW)
    docs=[document(k,v) for k,v in gold_cases().items()]
    with ProcessPoolExecutor(4) as pool:
        parts=list(pool.map(replay,[docs[i::4] for i in range(4)]))
    a,b,c=[np.concatenate([p[i] for p in parts]) for i in range(3)]
    use=sum(p[3] for p in parts)
    out={"cases":len(a),"candidate":full_score(a),"preceding":full_score(b),"original_arcs_only":full_score(c),
         "gap_ci95":bootstrap(a,b),"gap_ci95_to_original_arcs":bootstrap(a,c),
         "joint_arcs_selected":int(use[0]),"exact_joint_arcs_selected":int(use[1]),"freeze_sha256":file_hash(freeze)}
    before=json.loads((root/"segment-weight-qa.json").read_text())["scores"]
    assert out["preceding"]==before
    main_record=root/"whole-joint-transfer-edit.json"
    if main_record.is_file() and "test" in (test:=json.loads(main_record.read_text()))["splits"]:
        assert test["splits"]["test"]["candidate"]==out["candidate"]
    save_json(root/"whole-joint-transfer-qa.json",out)
    print("Joint proposals independently replayed",len(a),out["candidate"]["exact_labeled_spans"]["f1"],use.tolist(),flush=True)


if __name__=="__main__":
    main()
