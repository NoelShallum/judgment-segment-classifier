"""Check the pure CPU supplied-sentence interface against the joint benchmark."""
import json

import numpy as np
from safetensors.numpy import load_file,save_file

from sentence_joint_decoding import (
    CHECKPOINT,PARENT,DONOR,EDGE,OUT,ROOT,HierModel,aligned_embeddings,distribution,document,
    file_hash,gold_cases,load_decoder,package,predict,save_json,torch,utility,
)
from hier_model import LABELS
from predict_role_edge import infer_sentences

DEST=OUT/"known-sentence-joint-portable.safetensors"


def main():
    torch.set_num_threads(2)
    freeze=ROOT/"results/sentence-joint-frozen.json"
    f=json.loads(freeze.read_text())
    assert f["source_sha256"]==file_hash(PARENT) and f["weights_sha256"]==file_hash(CHECKPOINT)
    base=load_file(str(PARENT))
    edit=load_file(str(CHECKPOINT))
    kind=f["choice"][0]
    tensors={**base,**{"joint."+k:v for k,v in edit.items()},
             "joint.mode":np.asarray({"posterior":0,"structure":1,"context":2}[kind],np.int8)}
    save_file(tensors,str(DEST))
    portable=load_file(str(DEST))
    assert all(np.array_equal(v,portable[k]) for k,v in tensors.items())
    live={q["case"] for q in json.loads((EDGE/"qa.json").read_text())["cases"]}
    source=gold_cases()
    docs=[document(k,source[k]) for k in sorted(live)]
    m=HierModel(DONOR)
    v=json.loads((EDGE/"vocab.json").read_text())
    states=list(aligned_embeddings(m,docs,"test"))
    p,current,marg,rank,struct,context=package(docs,states,m,base,v,load_decoder())
    head=None if kind=="posterior" else distribution(context if kind=="context" else struct,current,edit)
    macro,retention,rate,strength=edit["settings"]
    values=utility(marg,head,current,macro,edit,retention)
    y=predict(p,current,rank,values,rate,edit["transition"],strength)
    observed=[]
    for d,a,b in zip(docs,p.sizes,p.sizes[1:]):
        inputs=[d["text"][l:r].strip() for l,r in d["reference_units"]]
        output=infer_sentences(inputs,m,portable,v,portable,"joint",float(base["edge.gain"]))
        assert output["labels"]==[LABELS[i] for i in y[a:b]],d["id"]
        observed.append({"case":d["id"],"rows":len(inputs)})
    save_json(ROOT/"results/sentence-joint-portable-qa.json",{"cases":observed,
            "freeze_sha256":file_hash(freeze),"weights_sha256":file_hash(DEST),
            "packed_weights_sha256":file_hash(CHECKPOINT)})
    print("Verified portable joint predictions",observed,flush=True)


if __name__=="__main__":
    main()
