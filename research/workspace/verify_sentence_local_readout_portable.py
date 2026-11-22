"""Verify the released CPU readout against official cached rows and live judgments."""
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np
from safetensors.numpy import save_file

from sentence_document_graph_write import (
    STEM,OUT,ROOT,PARENT,EDGE,DONOR,HierModel,torch,aligned_embeddings,document,gold_cases,
    load_file,file_hash,row_summary,cm,save_json,
)
from predict_role_edge import decode_profile,infer_sentences
from role_cues import cue_features
from role_edge_runtime import transition_scores
from hier_model import LABELS
from sentence_local_readout_runtime import labels

DEST=OUT/"sentence-local-readout-portable.safetensors"


def replay(docs,live,checkpoint=DEST):
    torch.set_num_threads(1)
    model=HierModel(DONOR)
    parent=load_file(str(checkpoint))
    vocab=json.loads((EDGE/"vocab.json").read_text())
    gain=float(parent["edge.gain"])
    confusion=np.zeros((7,7),np.int64)
    verified=[]
    for d,h in zip(docs,aligned_embeddings(model,docs,"test")):
        t=d["text"]
        units=d["reference_units"]
        words=[t[a:b] for a,b in units]
        pos=np.eye(5)[np.digitize([(a+b)*.5/len(t) for a,b in units],[.05,.25,.60,.91])]
        raw=model.logits(h)+cue_features(words)@parent["cue_head"]+pos@parent["position_head"]
        edge=transition_scores(t,units,vocab,parent)
        y0=decode_profile(raw,edge,parent,gain)
        pred=labels(raw,edge,parent,gain,y0,[len(w.strip()) for w in words],h)
        confusion+=cm(np.asarray(d["reference_roles"]),pred)
        if d["id"] in live:
            prediction=infer_sentences(words,model,parent,vocab,parent,"joint",gain)
            assert prediction["labels"]==[LABELS[i] for i in pred],d["id"]
            verified.append({"case":d["id"],"rows":len(words)})
    return confusion,verified


def shard(args):
    return replay(*args)


def main():
    root=ROOT/"results"
    freeze=root/(STEM+"-frozen.json")
    f=json.loads(freeze.read_text())
    arm,_,_,kind,budget,mix,guard,threshold=f["choice"]
    assert (arm,kind,guard)==("local","posterior",True)
    assert f["parent_sha256"]==file_hash(PARENT)
    source=OUT/(STEM+".safetensors")
    assert f["weights_sha256"]==file_hash(source)
    a=load_file(str(source))
    physical={**load_file(str(PARENT)),**{
        "local_readout.center":a["center"],"local_readout.scale":a["scale"],
        "local_readout.matrix":np.ascontiguousarray(a["matrix"][:,:7]),"local_readout.bias":np.ascontiguousarray(a["bias"][:7]),
        "local_readout.settings":np.asarray([budget,mix,threshold])}}
    save_file(physical,str(DEST))
    loaded=load_file(str(DEST))
    for k,v in physical.items():
        np.testing.assert_array_equal(v,loaded[k])
    cases=gold_cases()
    docs=[document(k,v) for k,v in cases.items()]
    live={q["case"] for q in json.loads((EDGE/"qa.json").read_text())["cases"]}
    with ProcessPoolExecutor(3) as pool:
        parts=list(pool.map(shard,[(docs[i::3],live) for i in range(3)]))
    got=row_summary(sum(q for q,_ in parts))
    receipts=[r for _,rows in parts for r in rows]
    assert {x["case"] for x in receipts}==live
    proof=json.loads((root/(STEM+".json")).read_text())
    assert got==proof["splits"]["test"]["candidate"] and proof["freeze_sha256"]==file_hash(freeze)
    save_json(root/"sentence-local-readout-portable-qa.json",{"scores":got,"official_cases":len(docs),
              "live_cases":receipts,"weights_sha256":file_hash(DEST),"freeze_sha256":file_hash(freeze),
              "width":int(a["matrix"].shape[0]),"source_sha256":file_hash(source)})
    print("Portable scores verified",len(docs),got["macro_f1"],receipts,flush=True)


if __name__=="__main__":
    main()
