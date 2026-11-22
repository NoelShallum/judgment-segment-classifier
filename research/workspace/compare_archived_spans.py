"""Project existing author sentence outputs to spans using the official, supplied cuts."""
import argparse
import csv
import json

import numpy as np
from hier_full import gold_cases
from program_trial import TO_ID,save_json
from seven_paths import ROOT
from span_eval import counts,document,summarize
from weight_edit import file_hash

def main():
    roots=ROOT/"references/all-predictions"
    sources={
        "previous_predicted_role":roots/"LegalSeg_InLegalBERT_Predictions"/"InLegalBert(i-1, label_p, i).csv",
        "toinlegalbert":roots/"LegalSeg_ToInLegalBert_Predictions"/"ToInlegalbertpredictions.csv",
        "standalone":roots/"LegalSeg_InLegalBERT_Predictions"/"InLegalBert(i)_predictions.csv",
        "gnn":roots/"LegalSeg_GNN_Predictions"/"GNN_predictions.csv",
    }
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--additional",action="store_true")
    args=parser.parse_args()
    output=ROOT/"results"/"archived-sentence-span-references.json"
    all_scores={}
    if args.additional:
        all_scores=json.loads(output.read_text())["scored_release_files"]
        b=roots/"LegalSeg_InLegalBERT_Predictions"
        sources={k:b/f for k,f in (("one_previous","InLegalBert(i-1, i)_predictions.csv"),("look_both_sides","InLegalBert(i-1, i, i+1)_predictions.csv"),
                                  ("two_previous","InLegalBert(i-2, i-1, i)_predictions.csv"),("position","InLegalBert(pos, i)_predictions.csv"))}
    cases=gold_cases()
    ordered=sorted((r for rs in cases.values() for r in rs))
    assert all(i==r[0] for i,r in enumerate(ordered))
    truth=np.array([r[2] for r in ordered])
    docs=[(document(k,rs),[i for i,text,_ in rs if text.strip()]) for k,rs in cases.items()]
    for name,path in sources.items():
        with path.open(encoding="utf-8-sig",newline="") as f:
            rows=list(csv.DictReader(f))
        actual=np.array([int(r["true"]) for r in rows])
        if not np.array_equal(actual,truth):
            raise ValueError(f"Author gold does not align to official test: {name}")
        t=np.array([int(r.get("pred",r.get("predicted"))) for r in rows])
        results=np.stack([counts(d,d["reference_units"],t[ix]) for d,ix in docs])
        summary=summarize(results.sum(0));all_scores[name]={"bytes_sha256":file_hash(path),"scores":summary}
        print(name,summary["exact_labeled_spans"]["f1"],summary["token_accuracy"],flush=True)
    save_json(output,{
       "protocol":"The full 149,881 official test gold sequence matches each of the published files. Blank rows are excluded exactly as in raw full-judgment evaluation and adjacent same-role outputs are merged. Gold sentence offsets are supplied to these projection references. Metrics give a useful strong comparison, as the raw-input models must also infer candidate boundaries.",
       "cases":len(cases),"scored_release_files":all_scores})
if __name__=="__main__":
    main()
