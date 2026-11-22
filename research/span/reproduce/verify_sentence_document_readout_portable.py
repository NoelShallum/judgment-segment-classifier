"""Verify the two-hop readout through the same CPU sentence interface."""
from concurrent.futures import ProcessPoolExecutor
import json
from time import perf_counter

import numpy as np
from safetensors.numpy import save_file

from verify_sentence_local_readout_portable import replay
from sentence_document_graph_write import (
    STEM,OUT,ROOT,PARENT,EDGE,DONOR,HierModel,torch,document,gold_cases,load_file,file_hash,row_summary,save_json,
)
from predict_role_edge import infer_sentences

DEST=OUT/"sentence-document-readout-portable.safetensors"


def replay_portable(args):
    return replay(*args,checkpoint=DEST)


def time_profiles(docs):
    torch.set_num_threads(2)
    m=HierModel(DONOR)
    vocab=json.loads((EDGE/"vocab.json").read_text())
    paths={"sentence-local":OUT/"sentence-local-readout-portable.safetensors","sentence-document":DEST}
    weights={k:load_file(str(v)) for k,v in paths.items()}
    timing=[]
    for d in docs:
        t=d["text"]
        phrases=[t[a:b].strip() for a,b in d["reference_units"]]
        rms={}
        for name,w in weights.items():
            gain=float(w["edge.gain"])
            infer_sentences(phrases,m,w,vocab,w,"joint",gain)
            elapsed=[]
            for _ in range(3):
                start=perf_counter()
                infer_sentences(phrases,m,w,vocab,w,"joint",gain)
                elapsed.append(perf_counter()-start)
            rms[name]=float(np.median(elapsed))
        timing.append({"case":d["id"],"sentences":len(phrases),"cpu_seconds":rms})
    return timing


def main():
    root=ROOT/"results"
    f=root/(STEM+"-frozen.json")
    config=json.loads(f.read_text())
    c=json.loads((root/(STEM+"-comparison.json")).read_text())
    assert c["freeze_sha256"]==file_hash(f)
    arm,_,_,kind,budget,gain,guard,cut=config["choices_by_arm"]["two-hop"]
    assert (arm,kind,guard)==("two-hop","posterior",True)
    assert config["parent_sha256"]==file_hash(PARENT)
    source=OUT/(STEM+"-two-hop.safetensors")
    assert c["comparison_weights_sha256"]["two-hop"]==file_hash(source)
    w=load_file(str(source))
    p={**load_file(str(PARENT)),**{
        "local_readout.center":w["center"],"local_readout.scale":w["scale"],
        "local_readout.matrix":np.ascontiguousarray(w["matrix"][:,:7]),"local_readout.bias":np.ascontiguousarray(w["bias"][:7]),
        "local_readout.settings":np.asarray([budget,gain,cut]),"local_readout.two_hop":np.asarray(1,np.int8)}}
    save_file(p,str(DEST))
    r=load_file(str(DEST))
    for k,v in p.items():
        np.testing.assert_array_equal(r[k],v)
    docs=[document(k,v) for k,v in gold_cases().items()]
    live={v["case"] for v in json.loads((EDGE/"qa.json").read_text())["cases"]}
    with ProcessPoolExecutor(3) as pool:
        parts=list(pool.map(replay_portable,[(docs[i::3],live) for i in range(3)]))
    scores=row_summary(sum(x for x,_ in parts))
    cases=[s for _,live_cases in parts for s in live_cases]
    assert {q["case"] for q in cases}==live
    assert scores==c["splits"]["test"]["arms"]["two-hop"]["scores"]
    latency=time_profiles([d for d in docs if d["id"] in live])
    save_json(root/"sentence-document-readout-portable-qa.json",{"scores":scores,"official_cases":len(docs),
              "live_cases":cases,"latency":latency,"source_sha256":file_hash(source),"width":int(w["matrix"].shape[0]),
              "weights_sha256":file_hash(DEST),"freeze_sha256":file_hash(f)})
    print("Verified document portable profile",scores["macro_f1"],latency,flush=True)

if __name__=="__main__":
    main()
