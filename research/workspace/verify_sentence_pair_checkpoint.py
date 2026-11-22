"""Replay written sentence-specific decoder weights and verify the sentence input API."""
import json

import numpy as np
from safetensors.numpy import load_file

from role_pair_weight_search import (EDGE,DONOR,ROOT,HierModel,document,file_hash,
    gold_cases,save_json,torch)
from hier_full import CACHE, cm, row_summary
from hier_model import CRF,LABELS
from hier_trial import OUT
from discourse_span_edit import text_features
from predict_role_edge import infer_sentences
from role_edge_runtime import decode,transition_scores
from score_frozen_edge_sentence_rows import pair


def main():
    torch.set_num_threads(2)
    saved=OUT/"known-sentence-role-pair.safetensors"
    pin=json.loads((ROOT/"results/known-sentence-pair-edit-frozen.json").read_text())
    assert file_hash(saved)==pin["patch_sha256"]
    p=load_file(str(saved))
    parent=load_file(str(EDGE/"role-edge-balanced.safetensors"))
    vocab=json.loads((EDGE/"vocab.json").read_text())
    expected=json.loads((ROOT/"results/known-sentence-pair-edit.json").read_text())["splits"]["test"]["candidate"]
    gain=float(p["edge.gain"])
    model=HierModel(DONOR)
    matched=[]
    seen=0
    total=np.zeros((7,7),dtype=np.int64)
    clean=[]
    clean_parent=[]
    removed=[]
    selected={x["case"] for x in json.loads((EDGE/"qa.json").read_text())["cases"]}
    for k,rr in gold_cases().items():
        d=document(k,rr)
        d["units"]=d["reference_units"]
        sentences=[text for _,text,_ in rr if text.strip()]
        rows=np.asarray([i for i,(_,text,_) in enumerate(rr) if text.strip()])
        cue,pos=text_features(d)
        with np.load(CACHE/(k+".npz")) as ar:
            logit=ar["context"][rows]+cue@p["cue_head"]+pos@p["position_head"]
        edge=transition_scores(d["text"],d["units"],vocab,p)
        labels=decode(logit,p[CRF],edge,gain)
        total+=cm(d["reference_roles"],labels)
        if len(sentences)!=len(rr):
            raw_live=model.logits(model.contextualize(model.words(sentences)))
            logit=raw_live+cue@p["cue_head"]+pos@p["position_head"]
            labels=decode(logit,p[CRF],edge,gain)
            removed.append(k)
        control_edge=transition_scores(d["text"],d["units"],vocab,parent)
        control=decode(logit,parent[CRF],control_edge,float(parent["edge.gain"]))
        clean.append(cm(d["reference_roles"],labels))
        clean_parent.append(cm(d["reference_roles"],control))
        if k in selected:
            actual=infer_sentences(sentences,model,p,vocab,p,"joint",gain)
            assert actual["labels"]==[LABELS[y] for y in labels]
            matched.append(k)
        seen+=1
        if seen%240==0:print("Verified weights on",seen,"judgments",flush=True)
    change=int(np.abs(np.asarray(expected["confusion"])-total).sum())
    assert change==0
    assert set(matched)==selected
    a,b=np.stack(clean),np.stack(clean_parent)
    save_json(ROOT/"results/known-sentence-pair-edit-qa.json",{"case_count":seen,"sha256":file_hash(saved),
       "benchmark_confusion_l1_difference":change,"verified_live_cases":matched,
       "cases_recomputed_without_blank_rows":removed,"scores":row_summary(a.sum(0)),
       "parent_scores":row_summary(b.sum(0)),"gap_ci95_macro_f1_accuracy":pair(a,b)})
    print("Checkpoint replayed all",seen,"cases and passed live input checks",flush=True)


if __name__=="__main__":
    main()
