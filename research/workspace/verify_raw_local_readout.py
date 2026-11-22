"""Replay the frozen local sentence edit on raw splitter units."""
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np

from verify_raw_joint_direct import (
    CACHE,EDGE,JOINT,OUT,RAW_STATES,ROOT,bootstrap,document,file_hash,full_score,gold_cases,load_file,
    prepare_score,save_json,torch,transition_scores,decode_profile,joint_labels,infer,merge,HierModel,DONOR,LABELS,setup,
)
from sentence_local_readout_runtime import labels

PRETRAINED=OUT/"sentence-local-readout-portable.safetensors"
TAG="raw-local-transfer"


def replay(args):
    docs,live=args
    torch.set_num_threads(1)
    _,vocab,_,_=setup()
    parent=load_file(str(JOINT))
    edited=load_file(str(PRETRAINED))
    assert all(np.array_equal(v,edited[k]) for k,v in parent.items())
    model=HierModel(DONOR)
    gain=float(parent["edge.gain"])
    base=[];positive=[];checked=[];count=0
    for d in docs:
        k=d["id"]
        with np.load(CACHE/(k+".npz")) as cache:
            raw=cache["raw"]
        h=np.load(RAW_STATES/(k+".npy"),allow_pickle=False)
        item=prepare_score(d,raw,{"joint":parent})
        logits=item.logits["joint"]
        edge=transition_scores(d["text"],d["units"],vocab,parent)
        initial=decode_profile(logits,edge,parent,gain)
        lengths=[len(d["text"][a:b].strip()) for a,b in d["units"]]
        before=joint_labels(logits,edge,parent,gain,initial,lengths,h)
        after=labels(logits,edge,edited,gain,initial,lengths,h)
        count+=int((before!=after).sum())
        base.append(item.score(before))
        positive.append(item.score(after))
        if k in live:
            report=infer(d["text"],model,edited,vocab,edited,"joint",gain)
            assert report["spans"]==merge(d["units"],after,LABELS),k
            checked.append(k)
    return np.stack(base),np.stack(positive),count,checked


def main():
    root=ROOT/"results"
    checked=json.loads((root/"sentence-local-readout-portable-qa.json").read_text())
    assert checked["weights_sha256"]==file_hash(PRETRAINED)
    docs=[document(k,v) for k,v in gold_cases().items()]
    live={x["case"] for x in checked["live_cases"]}
    with ProcessPoolExecutor(3) as pool:
        parts=list(pool.map(replay,[(docs[i::3],live) for i in range(3)]))
    a,b=(np.concatenate([q[i] for q in parts]) for i in (0,1))
    assert len(a)==len(b)==len(docs)==712
    checked_ids=set(k for _,_,_,ids in parts for k in ids)
    assert checked_ids==live
    base=full_score(a)
    new=full_score(b)
    expected=json.loads((root/"raw-joint-direct-qa.json").read_text())
    assert base==expected["scores"]
    report={"cases":len(docs),"local_weights_sha256":file_hash(PRETRAINED),"joint_weights_sha256":file_hash(JOINT),
            "parent":base,"candidate":new,"paired":bootstrap(b,a),"live_cases":sorted(checked_ids),
            "changed_splitter_units":sum(q[2] for q in parts),
            "protocol":"Use raw splitter intervals and cached states from the raw text. Apply the same frozen local readout as for supplied sentences, then merge neighboring equal roles and evaluate exact boundaries and token accuracy."}
    save_json(root/(TAG+".json"),report)
    print(TAG,new["exact_labeled_spans"]["f1"],new["token_accuracy"],report["paired"],flush=True)


if __name__=="__main__":
    main()
