"""Measure whether fixed quantized MiniLM weights reproduce labels per judgment."""
import json

import numpy as np
from safetensors.numpy import load_file

from sentence_minilm_fusion import (
    DONOR,EDGE,MAIN,OUT,ROOT,TOKEN_LIMIT,HierModel,MiniLM,aligned_embeddings,apply,case_view,
    file_hash,predictions,sample_whole,save_json,torch,
)


def main():
    torch.set_num_threads(1)
    source=ROOT/"results/sentence-minilm-fusion-frozen.json"
    frozen=json.loads(source.read_text())
    name="semantic_fusion"
    arm=frozen["choices"][name][0]
    path=OUT/"known-sentence-minilm-fusion.safetensors"
    assert file_hash(path)==frozen["weights_sha256"]
    data={k.split(".",1)[1]:v for k,v in load_file(str(path)).items() if k.startswith(name+".")}
    encoder=MiniLM()
    assert encoder.hash==frozen["graph_sha256"]
    model=HierModel(DONOR)
    old=load_file(str(MAIN))
    vocab=json.loads((EDGE/"vocab.json").read_text())
    cohort=sorted(sample_whole()["select"],key=lambda d:len(d["reference_units"]))
    examples=[cohort[q] for q in (len(cohort)//8,len(cohort)//2,7*len(cohort)//8)]
    receipts=[]
    for d,state in zip(examples,aligned_embeddings(model,examples,"select")):
        row=case_view(d,state,model,old,vocab,.30)
        disk=ROOT/"cache"/f"minilm-sentence-fusion-{encoder.hash[:12]}-{TOKEN_LIMIT}"/f"{row.case}-0.30.npz"
        with np.load(disk,allow_pickle=False) as stored:
            assert np.array_equal(stored["query"],row.query)
            row.semantic=stored["x"]
        score=apply(row,predictions(row,arm,data),*data["settings"])
        alone=encoder.encode([row.texts[k] for k in row.query])
        cosine=np.sum(alone*row.semantic,axis=1)
        delta=np.abs(alone-row.semantic)
        row.semantic=alone
        check=apply(row,predictions(row,arm,data),*data["settings"])
        r={"case":row.case,"rows":len(check),"queried":len(alone),
           "mean_cosine":float(cosine.mean()),"max_absolute_feature_difference":float(delta.max()),
           "label_disagreements":int((check!=score).sum())}
        print(r,flush=True)
        receipts.append(r)
    save_json(ROOT/"results/sentence-minilm-batch-audit.json",
       {"freeze_sha256":file_hash(source),"cases":receipts,
        "protocol":"Compare identical selected sentences encoded in the fixed cross-judgment benchmark batch schedule and within their own judgment. Both evaluations use the same frozen graph and readout. Select three validation cases by document size."})


if __name__=="__main__":
    main()
