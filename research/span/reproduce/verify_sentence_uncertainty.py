"""Check the standalone hierarchy correction at every official sentence boundary."""
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np
from safetensors.numpy import load_file,save_file

from sentence_minilm_fusion import (
    DONOR,EDGE,MAIN,OUT,ROOT,HierModel,aligned_embeddings,cm,document,file_hash,
    gold_cases,row_summary,save_json,torch,transition_scores,
)
from discourse_span_edit import text_features
from predict_role_edge import decode_profile,infer_sentences
from sentence_role_retention import paired
from sentence_uncertainty_runtime import uncertainty_labels

DESTINATION=OUT/"known-sentence-uncertainty.safetensors"


def check(pack):
    docs,live=pack
    torch.set_num_threads(1)
    params=load_file(str(DESTINATION))
    parent=load_file(str(OUT/"known-sentence-role-pair.safetensors"))
    model=HierModel(DONOR)
    vocab=json.loads((EDGE/"vocab.json").read_text())
    gain=float(params["edge.gain"])
    scores=[];b=[];c=[];seen=[]
    for d,state in zip(docs,aligned_embeddings(model,docs,"test")):
        y=d["reference_roles"]
        d={**d,"units":d["reference_units"]}
        texts=[d["text"][a:z].strip() for a,z in d["units"]]
        cues,position=text_features(d)
        x=model.logits(state)+cues@params["cue_head"]+position@params["position_head"]
        phi=transition_scores(d["text"],d["units"],vocab,params)
        current=decode_profile(x,phi,params,gain)
        earlier=decode_profile(x,phi,parent,gain)
        new=uncertainty_labels(x,phi,params,gain,current,[len(s) for s in texts])
        scores.append(cm(y,new))
        b.append(cm(y,current))
        c.append(cm(y,earlier))
        if d["id"] in live:
            result=infer_sentences(texts,model,params,vocab,params,"joint",gain)
            from hier_model import LABELS
            assert result["labels"]==[LABELS[i] for i in new]
            seen.append(d["id"])
    return np.stack(scores),np.stack(b),np.stack(c),seen


def main():
    root=ROOT/"results"
    path=OUT/"known-sentence-minilm-token-table.safetensors"
    freeze=root/"sentence-minilm-token-table-frozen.json"
    fitted=json.loads(freeze.read_text())
    assert fitted["weights_sha256"]==file_hash(path)
    baseline=load_file(str(MAIN))
    correct={k.replace("hierarchy_only.","uncertainty."):v
             for k,v in load_file(str(path)).items() if k.startswith("hierarchy_only.")}
    save_file({**baseline,**correct},str(DESTINATION))
    docs=[document(k,r) for k,r in gold_cases().items()]
    live={v["case"] for v in json.loads((EDGE/"qa.json").read_text())["cases"]}
    with ProcessPoolExecutor(3) as pool:
        chunks=list(pool.map(check,[(docs[i::3],live) for i in range(3)]))
    a,b,c=[np.concatenate([bit[i] for bit in chunks]) for i in range(3)]
    assert set(x for v in chunks for x in v[3])==live
    expected=json.loads((root/"sentence-minilm-token-table-edit.json").read_text())["splits"]["test"]
    assert row_summary(a.sum(0))==expected["profiles"]["hierarchy_only"]["scores"]
    assert row_summary(b.sum(0))==expected["preceding"]
    old=json.loads((root/"known-sentence-pair-edit-qa.json").read_text())["scores"]
    assert row_summary(c.sum(0))==old
    save_json(root/"sentence-uncertainty-qa.json",{"cases":len(a),"verified_live_cases":sorted(live),
         "scores":row_summary(a.sum(0)),"calibrated":row_summary(b.sum(0)),"context":row_summary(c.sum(0)),
         "paired_to_calibrated":paired(a,b),"paired_to_context":paired(a,c),
         "weights_sha256":file_hash(DESTINATION),"fit_freeze_sha256":file_hash(freeze)})
    print("Sentence uncertainty replay verified",len(a),row_summary(a.sum(0))["macro_f1"],flush=True)


if __name__=="__main__":
    main()
