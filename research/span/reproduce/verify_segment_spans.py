"""Replay the portable weighted segment scorer over all official test cases."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np

from role_edge_interpolation import (
    EDGE, DONOR, HierModel, ROOT, document, file_hash, full_score, gold_cases, load_file, save_json, torch,
)
from hier_full import CACHE
from predict_role_edge import infer_segments
from role_pair_weight_search import prepare_score,transition_scores
from segment_span_runtime import segment_labels
from span_model import merge
from hier_model import LABELS

PROFILES = {
    "segment-weight": "whole-segment-weights.safetensors",
    "segment-posterior-ends": "whole-segment-posterior-ends-weights.safetensors",
}


def replay(job):
    stem,jobs,live=job
    torch.set_num_threads(1)
    params=load_file(str(ROOT/"checkpoints/hierarchical"/PROFILES[stem]))
    if stem=="segment-posterior-ends":
        params.setdefault("posterior_ends",np.asarray(True))
    parents={k:load_file(str(EDGE/f"role-edge-{k}.safetensors")) for k in ("exact","balanced")}
    vocab=json.loads((EDGE/"vocab.json").read_text())
    cases=[]
    for d in jobs:
        with np.load(CACHE/(d["id"]+".npz")) as source:
            item=prepare_score(d,source["raw"],parents)
        edge=transition_scores(d["text"],d["units"],vocab,parents["exact"])
        labels=segment_labels(item.logits,edge,parents,item.token_role_counts.sum(1),params)
        if d["id"] in live:
            answer=infer_segments(d["text"],HierModel(DONOR),parents,vocab,params)
            assert answer["spans"]==merge(d["units"],labels,LABELS),d["id"]
        cases.append(item.score(labels))
    return np.stack(cases)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("study",choices=PROFILES)
    args=p.parse_args()
    stem=args.study
    studies=ROOT/"results"
    freeze=studies/f"{stem}-frozen.json"
    result=json.loads((studies/f"{stem}-write.json").read_text())
    assert result["freeze_sha256"]==file_hash(freeze)
    path=ROOT/"checkpoints/hierarchical"/PROFILES[stem]
    assert file_hash(path)==json.loads(freeze.read_text())["weight_sha256"]
    cases=[document(k,r) for k,r in gold_cases().items()]
    qa=json.loads((EDGE/"qa.json").read_text())
    live={x["case"] for x in qa["cases"]}
    assert live <= {d["id"] for d in cases}
    with ProcessPoolExecutor(4) as pool:
        pieces=list(pool.map(replay,[(stem,cases[i::4],live) for i in range(4)]))
    checked=full_score(np.concatenate(pieces))
    expected=result["splits"]["test"]
    assert checked==expected["candidate"],(checked["exact_labeled_spans"],expected["candidate"]["exact_labeled_spans"])
    save_json(studies/f"{stem}-qa.json",
        {"cases":len(cases),"live_cases":sorted(live),"weights_sha256":file_hash(path),"scores":checked,
         "parent":expected["parent"],"gap_ci95":expected["gap_ci95"]})
    print("All cases match",stem,len(cases),checked["exact_labeled_spans"],flush=True)


if __name__=="__main__":
    main()
