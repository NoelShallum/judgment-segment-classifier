"""Verify direct sentence-joint transfer on raw text and compare raw accuracy profiles."""
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np

from whole_joint_transfer import CACHE,CRF,DONOR,EDGE,JOINT,OUT,RAW_STATES,ROOT,HierModel,bootstrap,document,file_hash,full_score,gold_cases,load_file,prepare_score,save_json,setup,torch,transition_scores
from hier_model import LABELS,path_labels
from predict_role_edge import decode_profile,infer
from role_edge_runtime import decode,decompose
from sentence_joint_runtime import joint_labels
from span_model import merge


def matched(args):
    docs,live=args
    torch.set_num_threads(1)
    parents,vocab,_,_=setup()
    direct=load_file(str(JOINT))
    legacy=load_file(str(OUT/"role-specific-balanced.safetensors"))
    model=HierModel(DONOR)
    gain=float(direct["edge.gain"])
    role_edge=parents["balanced"]
    g=float(role_edge["edge.gain"])
    mode="interaction" if int(role_edge["edge.mode"]) else "joint"
    exact=[];old=[];other=[];checked=[]
    for d in docs:
        with np.load(CACHE/(d["id"]+".npz")) as cache:
            raw=cache["raw"]
        state=np.load(RAW_STATES/(d["id"]+".npy"),allow_pickle=False)
        item=prepare_score(d,raw,{"joint":direct,"balanced":role_edge})
        logits=item.logits["joint"]
        phi=transition_scores(d["text"],d["units"],vocab,direct)
        y=decode_profile(logits,phi,direct,gain)
        n=[len(d["text"][a:b].strip()) for a,b in d["units"]]
        y=joint_labels(logits,phi,direct,gain,y,n,state)
        exact.append(item.score(y))
        other_phi=transition_scores(d["text"],d["units"],vocab,role_edge)
        z=decode(item.logits["balanced"],role_edge[CRF],decompose(other_phi,mode),g)
        other.append(item.score(z))
        old.append(item.score(path_labels(raw,legacy[CRF])))
        if d["id"] in live:
            actual=infer(d["text"],model,direct,vocab,direct,"joint",gain)
            assert actual["spans"]==merge(d["units"],y,LABELS),d["id"]
            checked.append(d["id"])
    return np.stack(exact),np.stack(old),np.stack(other),checked


def main():
    collection=[document(k,r) for k,r in gold_cases().items()]
    live={q["case"] for q in json.loads((EDGE/"qa.json").read_text())["cases"]}
    with ProcessPoolExecutor(3) as pool:
        parts=list(pool.map(matched,[(collection[i::3],live) for i in range(3)]))
    a,b,c=[np.concatenate([part[i] for part in parts]) for i in range(3)]
    assert set(q for p in parts for q in p[3])==live
    observed={"cases":len(a),"scores":full_score(a),"older_token_balanced":full_score(b),
              "role_edge_balanced":full_score(c),"paired_to_token_balanced":bootstrap(a,b),
              "paired_to_role_edge_balanced":bootstrap(a,c),"verified_live_cases":sorted(live),
              "weights_sha256":file_hash(JOINT)}
    old=json.loads((ROOT/"results/transition-token-tradeoff.json").read_text())["scores"]["test"]["candidate"]
    new=json.loads((EDGE/"results.json").read_text())["splits"]["test"]["balanced"]["candidate"]
    for recorded,expected in ((observed["older_token_balanced"],old),(observed["role_edge_balanced"],new)):
        assert all(recorded[k]==expected[k] for k in ("exact_labeled_spans","token_accuracy"))
    data=ROOT/"results/whole-joint-transfer-edit.json"
    if data.is_file() and "test" in (r:=json.loads(data.read_text()))["splits"]:
        assert r["splits"]["test"]["standalone_sentence_joint"]==observed["scores"]
    save_json(ROOT/"results/raw-joint-direct-qa.json",observed)
    print("Raw direct CPU replay",len(a),observed["scores"]["exact_labeled_spans"],observed["scores"]["token_accuracy"],flush=True)


if __name__=="__main__":
    main()
