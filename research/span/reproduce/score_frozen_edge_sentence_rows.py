"""Evaluate the frozen role-pair weights at supplied LegalSeg sentence boundaries."""
import json
import sys

import numpy as np
from safetensors.numpy import load_file

from discourse_span_edit import text_features
from hier_full import CACHE, gold_cases, cm, row_summary
from hier_model import CRF, path_labels
from program_trial import save_json
from seven_paths import ROOT
from span_eval import document
from weight_edit import file_hash

BUNDLE = ROOT / "astra-role-edge"
sys.path.insert(0,str(BUNDLE))
from role_edge_runtime import decode, decompose, transition_scores


def pair(first, second):
    rng = np.random.default_rng(17032)
    results = []
    for indexes in rng.integers(len(first), size=(1600,len(first))):
        a,z = row_summary(first[indexes].sum(0)),row_summary(second[indexes].sum(0))
        results.append([a["macro_f1"]-z["macro_f1"],a["accuracy"]-z["accuracy"]])
    return np.quantile(results,[.025,.975],axis=0).T.tolist()


def main():
    manifest = json.loads((BUNDLE/"deployment.json").read_text())
    vocab = json.loads((BUNDLE/"vocab.json").read_text())
    edits = {}
    for name, v in manifest["profiles"].items():
        pat = BUNDLE / v["file"]
        assert file_hash(pat) == v["sha256"]
        edits[name] = load_file(str(pat))
    values = {k:{"edited":[],"parent":[]} for k in edits}
    for i,(case,rows) in enumerate(gold_cases().items(),1):
        d = document(case,rows)
        d["units"] = d["reference_units"]
        selected = [j for j,(_,text,_) in enumerate(rows) if text.strip()]
        labels = np.array(d["reference_roles"])
        with np.load(CACHE / (case+".npz")) as record:
            logits = record["context"][selected]
        cues, position = text_features(d)
        edges = transition_scores(d["text"],d["units"],vocab, next(iter(edits.values())))
        for name,p in edits.items():
            raw = logits + cues @ p["cue_head"] + position @ p["position_head"]
            mode = "interaction" if p["edge.mode"].item() else "joint"
            pred = decode(raw,p[CRF],decompose(edges,mode),p["edge.gain"].item())
            values[name]["edited"].append(cm(labels,pred))
            values[name]["parent"].append(cm(labels,path_labels(raw,p[CRF])))
        if i % 180 == 0: print("Known-sentence cases",i,flush=True)
    scores = {}
    for name,entry in values.items():
        a,z = (np.stack(entry[k]) for k in ("edited","parent"))
        scores[name]={"candidate":row_summary(a.sum(0)),"parent":row_summary(z.sum(0)),
                      "macro_f1_accuracy_gap_ci95":pair(a,z)}
        print(name,scores[name]["candidate"]["macro_f1"],scores[name]["parent"]["macro_f1"],flush=True)
    save_json(ROOT/"results/role-edge-given-sentences.json",{"cases":i,"profiles":scores,
        "protocol":"Use original LegalSeg sentence boundaries and full document context, excluding completely blank rows. Frozen weights were selected for a separate raw-span task and are never retuned on this test. Report macro-F1 and accuracy on the seven source labels."})


if __name__ == "__main__":
    main()
