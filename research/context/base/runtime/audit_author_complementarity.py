"""Measure error overlap with matched author predictions without selecting model weights."""
from concurrent.futures import ProcessPoolExecutor
import csv
import json

import numpy as np

from embeddinggemma_routed_write import (
    DONOR,EDGE,ROOT,PARENT,HierModel,aligned_embeddings,cm,document,file_hash,
    gold_cases,load_file,row_summary,save_json,torch,
)
from sentence_joint_runtime import joint_labels
from role_edge_runtime import transition_scores
from predict_role_edge import decode_profile
from role_cues import cue_features

OUTFILE=ROOT/"results/sentence-author-complementarity.json"


def prediction(rows):
    torch.set_num_threads(1)
    parent=load_file(str(PARENT))
    vocab=json.loads((EDGE/"vocab.json").read_text())
    model=HierModel(DONOR)
    results=[]
    g=float(parent["edge.gain"])
    for d,x in zip(rows,aligned_embeddings(model,rows,"test")):
        text=d["text"]
        spans=d["reference_units"]
        segments=[text[a:b] for a,b in spans]
        pos=np.eye(5)[np.digitize([(a+b)*.5/len(text) for a,b in spans],[.05,.25,.60,.91])]
        base=model.logits(x)+cue_features(segments)@parent["cue_head"]+pos@parent["position_head"]
        lexical=transition_scores(text,spans,vocab,parent)
        initial=decode_profile(base,lexical,parent,g)
        results.append((d["id"],joint_labels(base,lexical,parent,g,initial,[len(s.strip()) for s in segments],x)))
    return results


def author_outputs(cases):
    files=ROOT/"references/all-predictions"
    inlegal=files/"LegalSeg_InLegalBERT_Predictions"
    rel={"previous_predicted_role":inlegal/"InLegalBert(i-1, label_p, i).csv",
         "toinlegalbert":files/"LegalSeg_ToInLegalBert_Predictions/ToInlegalbertpredictions.csv",
         "standalone":inlegal/"InLegalBert(i)_predictions.csv",
         "gnn":files/"LegalSeg_GNN_Predictions/GNN_predictions.csv",
         "one_previous":inlegal/"InLegalBert(i-1, i)_predictions.csv",
         "look_both_sides":inlegal/"InLegalBert(i-1, i, i+1)_predictions.csv",
         "two_previous":inlegal/"InLegalBert(i-2, i-1, i)_predictions.csv",
         "position":inlegal/"InLegalBert(pos, i)_predictions.csv"}
    lookup={i:(txt,y) for seq in cases.values() for i,txt,y in seq}
    order=np.asarray([i for seq in cases.values() for i,txt,y in seq if txt.strip()])
    raw=np.asarray([lookup[i][1] for i in range(len(lookup))])
    archive=json.loads((ROOT/"results/known-sentence-numerical-archives.json").read_text())
    answer={}
    for key,path in rel.items():
        assert archive["archives"][key]["sha256"]==file_hash(path)
        with path.open(encoding="utf-8-sig",newline="") as f:
            rows=list(csv.DictReader(f))
        np.testing.assert_array_equal([int(r["true"]) for r in rows],raw)
        answer[key]=np.asarray([int(r.get("pred",r.get("predicted"))) for r in rows],np.int8)[order]
    return raw[order],answer,archive


def per_role(y,base,other):
    return np.bincount(y[base & other],minlength=7).tolist()


def main():
    cases=gold_cases()
    docs=[document(k,v) for k,v in cases.items()]
    y,archive,proof=author_outputs(cases)
    with ProcessPoolExecutor(4) as pool:
        out=dict(each for chunk in pool.map(prediction,[docs[i::4] for i in range(4)]) for each in chunk)
    parent=np.concatenate([out[d["id"]] for d in docs])
    local=row_summary(cm(y,parent))
    saved=json.loads((ROOT/"results/sentence-joint-qa.json").read_text())
    assert local==saved["scores"]
    record={}
    for key,a in archive.items():
        ref=row_summary(cm(y,a))
        assert ref==proof["archives"][key]["scores"]
        correct=parent==y
        their=a==y
        achievable=correct|their
        record[key]={"archive":ref,"both_correct":int(np.count_nonzero(correct&their)),
                     "only_local_correct":int(np.count_nonzero(correct&~their)),
                     "only_archive_correct":int(np.count_nonzero(~correct&their)),
                     "neither_correct":int(np.count_nonzero(~achievable)),
                     "only_archive_correct_by_role":per_role(y,~correct,their),
                     "only_local_correct_by_role":per_role(y,correct,~their),
                     "correctness_oracle":row_summary(cm(y,np.where(achievable,y,parent)))}
    any_archive=np.any(np.column_stack([a==y for a in archive.values()]),axis=1)
    union=any_archive|(parent==y)
    all_sources=row_summary(cm(y,np.where(union,y,parent)))
    save_json(OUTFILE,{"cases":len(cases),"rows":len(y),"parent_sha256":file_hash(PARENT),
                      "parent":local,"comparisons":record,"all_source_correctness_oracle":all_sources,
                      "role_order":list(local["per_role_f1"]),
                      "protocol":"Counterfactual chooses the correct label whenever either source proposed it. If neither source was correct, retain the local model label. Accuracy is a ceiling for choosing among these predictions. The reported macro-F1 uses that specific fallback. Gold is used only for the diagnostic."})
    key=proof["best_by_macro_f1"]
    print(key,record[key]["only_archive_correct"],record[key]["only_local_correct"],
          record[key]["correctness_oracle"]["accuracy"],record[key]["correctness_oracle"]["macro_f1"],
          "all-source",all_sources["accuracy"],all_sources["macro_f1"],flush=True)


if __name__=="__main__":
    main()
