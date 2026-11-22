"""Replay the portable selective sentence profile and compare aligned author predictions."""
import json

import numpy as np
from safetensors.numpy import load_file

from sentence_label_change_edit import (
    DONOR,EDGE,OUT,ROOT,HierModel,document,file_hash,gold_cases,save_json,torch,transition_scores
)
from sentence_role_retention import paired
from hier_full import CACHE,cm,row_summary
from hier_model import LABELS
from discourse_span_edit import text_features
from predict_role_edge import decode_profile,infer_sentences


def main():
    torch.set_num_threads(2)
    file=OUT/"known-sentence-label-change.safetensors"
    params=load_file(str(file))
    receipt=json.loads((ROOT/"results/sentence-label-change-frozen.json").read_text())
    assert file_hash(file)==receipt["weight_sha256"]
    expected=json.loads((ROOT/"results/sentence-label-change.json").read_text())["splits"]["test"]
    parent=load_file(str(OUT/"known-sentence-role-pair.safetensors"))
    vocab=json.loads((EDGE/"vocab.json").read_text())
    model=HierModel(DONOR)
    gain=float(params["edge.gain"])
    cases=gold_cases()
    to_test={x["case"] for x in json.loads((EDGE/"qa.json").read_text())["cases"]}
    observed=[]
    preceding=[]
    samples=[]
    for k,rr in cases.items():
        d=document(k,rr)
        d["units"]=d["reference_units"]
        pieces=[line for _,line,_ in rr if line.strip()]
        if len(pieces)==len(rr):
            with np.load(CACHE/(k+".npz")) as f:raw=f["context"]
        else:
            raw=model.logits(model.contextualize(model.words(pieces)))
        cues,positions=text_features(d)
        logits=raw+cues@params["cue_head"]+positions@params["position_head"]
        edges=transition_scores(d["text"],d["units"],vocab,params)
        result=decode_profile(logits,edges,params,gain)
        old=decode_profile(logits,edges,parent,gain)
        observed.append(cm(d["reference_roles"],result))
        preceding.append(cm(d["reference_roles"],old))
        if k in to_test:
            live=infer_sentences(pieces,model,params,vocab,params,"joint",gain)
            assert live["labels"]==[LABELS[y] for y in result]
            samples.append(k)
        if len(observed)%240==0:print("Checked",len(observed),flush=True)
    a,b=np.stack(observed),np.stack(preceding)
    np.testing.assert_array_equal(a.sum(0),expected["candidate"]["confusion"])
    np.testing.assert_array_equal(b.sum(0),expected["preceding"]["confusion"])
    assert set(samples)==to_test
    refs=json.loads((ROOT/"results/known-sentence-numerical-archives.json").read_text())
    archive_path=ROOT/"results/known-sentence-numerical-archives.npz"
    assert file_hash(archive_path)==refs["score_file_sha256"]
    archive=refs["best_by_macro_f1"]
    with np.load(archive_path) as score:
        np.testing.assert_array_equal(score["cases"],list(cases))
        named=score[archive]
    save_json(ROOT/"results/sentence-label-change-qa.json",{"cases":len(cases),"weights_sha256":file_hash(file),
        "verified_live_cases":samples,"scores":row_summary(a.sum(0)),"preceding":row_summary(b.sum(0)),
        "paired_to_preceding":paired(a,b),"best_aligned_archive":archive,"archive_scores":row_summary(named.sum(0)),
        "paired_to_archive":paired(a,named)})
    print("Portable sentence classification replayed",len(a),"judgments",flush=True)


if __name__=="__main__":
    main()
