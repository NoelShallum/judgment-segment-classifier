"""Replay the frozen boundary edit on 128 predeclared, case-disjoint validation judgments."""
from collections import defaultdict
import csv
import hashlib
import json

import numpy as np
from safetensors.numpy import load_file
from discourse_span_edit import text_features
from boundary_runtime import boundary_features,variable_decode
from contextual_boundary_edit import test_score
from hier_model import CRF,path_labels
from hier_trial import OUT
from program_trial import TO_ID,save_json
from seven_paths import ROOT
from span_eval import counts,document,summarize
from transition_span_edit import sample_whole
from weight_edit import file_hash

def main():
    edits=json.loads((ROOT/"results/contextual-boundary-write.json").read_text())
    frozen_path=ROOT/"astra-run-penalty/frozen-selection.json"
    read=json.loads(frozen_path.read_text())
    ids=read["selection_case_ids"]+read["confirmation_case_ids"]
    other=sample_whole()
    used={str(d["id"]) for ds in other.values() for d in ds}
    ids=list(map(str,ids))
    assert len(ids)==len(set(ids))==128 and not (set(ids)&used)
    selected=set(ids)
    bags=defaultdict(list)
    with (ROOT/"references/official/val.csv").open(newline="",encoding="utf-8-sig") as f:
        for i,r in enumerate(csv.DictReader(f)):
            if r["Index"] in selected:bags[r["Index"]].append((i,r["Text"],TO_ID[r["Label"]]))
    cache=ROOT/"astra-run-penalty/validation-cache"
    outputs={}
    for mode,value in edits["profiles"].items():
        path=OUT/f"whole-contextual-boundary-{mode}.safetensors"
        assert file_hash(path)==value["weights_sha256"]
        w=load_file(str(path));r=[];b=[]
        for key in ids:
            d=document(key,bags[key]);h=hashlib.sha256(d["text"].encode()).hexdigest()
            raw=np.load(cache/(h+".npy"),allow_pickle=False)
            assert raw.shape==(len(d["units"]),7)
            a,z=text_features(d)
            before=raw+a@w["cue_head"]+z@w["position_head"]
            pred=variable_decode(before,w[CRF],np.clip(boundary_features(raw)@w["boundary.weight"]+w["boundary.bias"],-6,6)*w["boundary.gain"])
            parent=path_labels(before,w[CRF])
            r.append(counts(d,d["units"],pred));b.append(counts(d,d["units"],parent))
        r=np.stack(r);b=np.stack(b)
        outputs[mode]={"candidate":summarize(r.sum(0)),"incumbent":summarize(b.sum(0)),"paired_f1_token_gap_95pct":test_score(r,b)}
        print(mode,outputs[mode]["candidate"]["exact_labeled_spans"]["f1"],outputs[mode]["incumbent"]["exact_labeled_spans"]["f1"],flush=True)
    save_json(ROOT/"results/contextual-boundary-independent-validation.json",{"cases":len(ids),"scores":outputs,
        "program_freeze_sha256":file_hash(ROOT/"results/contextual-boundary-write.json"),"cohort_sha256":file_hash(frozen_path),
        "method":"No refitting or reselection. Existing outputs from an independently drawn, case-disjoint 128-case raw judgment pool."})
if __name__=="__main__":
    main()
