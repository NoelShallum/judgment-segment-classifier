"""Replay numerically aligned LegalSeg author predictions on identical nonblank rows."""
import csv
from collections import defaultdict

import numpy as np

from hier_full import cm,gold_cases,row_summary
from program_trial import save_json
from seven_paths import ROOT
from weight_edit import file_hash


def main():
    source=ROOT/"references/all-predictions"
    inlegal=source/"LegalSeg_InLegalBERT_Predictions"
    rows={
        "previous_predicted_role":inlegal/"InLegalBert(i-1, label_p, i).csv",
        "toinlegalbert":source/"LegalSeg_ToInLegalBert_Predictions"/"ToInlegalbertpredictions.csv",
        "standalone":inlegal/"InLegalBert(i)_predictions.csv",
        "gnn":source/"LegalSeg_GNN_Predictions"/"GNN_predictions.csv",
        "one_previous":inlegal/"InLegalBert(i-1, i)_predictions.csv",
        "look_both_sides":inlegal/"InLegalBert(i-1, i, i+1)_predictions.csv",
        "two_previous":inlegal/"InLegalBert(i-2, i-1, i)_predictions.csv",
        "position":inlegal/"InLegalBert(pos, i)_predictions.csv",
    }
    cases=gold_cases()
    all_rows=sorted(r for d in cases.values() for r in d)
    assert all(r[0]==i for i,r in enumerate(all_rows))
    gold=np.asarray([q[2] for q in all_rows])
    masks=[np.array([i for i,t,_ in rr if t.strip()],dtype=int) for rr in cases.values()]

    per_case={}
    data={}
    for tag,p in rows.items():
        with p.open(encoding="utf-8-sig",newline="") as stream:
            records=list(csv.DictReader(stream))
        targets=[int(d["true"]) for d in records]
        np.testing.assert_array_equal(gold,targets)
        predictions=np.asarray([int(d.get("pred",d.get("predicted"))) for d in records])
        per_case[tag]=np.stack([cm(gold[mask],predictions[mask]) for mask in masks])
        out=row_summary(per_case[tag].sum(0))
        data[tag]={"scores":out,"sha256":file_hash(p)}
        print(tag,out["macro_f1"],out["accuracy"],flush=True)
    root=ROOT/"results"
    artifact=root/"known-sentence-numerical-archives.npz"
    np.savez_compressed(artifact,cases=np.array(list(cases)),**per_case)
    name=max(data,key=lambda k:data[k]["scores"]["macro_f1"])
    save_json(root/"known-sentence-numerical-archives.json",
       {"cases":len(cases),"input_rows":len(gold),"evaluated_rows":sum(map(len,masks)),
        "score_file_sha256":file_hash(artifact),"best_by_macro_f1":name,"archives":data,
        "protocol":"All eight archived numerical gold sequences exactly match the original LegalSeg test rows. All scores use only the 149,868 nonblank source sentences from the same 712 judgments. Archived variants with filtered or ambiguous row alignment are excluded from this specific comparison."})


if __name__=="__main__":
    main()
