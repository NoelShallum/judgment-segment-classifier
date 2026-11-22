"""Measure matched uploaded texts against the same saved Gemini decisions."""
from pathlib import Path
import hashlib
import json

import numpy as np

HERE=Path(__file__).resolve().parent


def main():
    ledger=json.loads((HERE/"excerpt-ledger.json").read_text())
    labels=json.loads((HERE/"protocol.json").read_text())["role_order"]
    path=HERE/"excerpt-predictions.npz"
    with np.load(path,allow_pickle=False) as d:
        pred={k:{sha:labels[int(y)] for sha,y in zip(d["doc_sha"],d[k],strict=True)} for k in d.files if k!="doc_sha"}
    out={}
    for panel,rows in ledger.items():
        use=list(dict.fromkeys(r["case"] for r in rows))
        by={c:i for i,c in enumerate(use)}
        case_ids=np.asarray([by[r["case"]] for r in rows])
        gold=np.asarray([r["gold"] for r in rows])
        expected={name:np.asarray([r["predictions"][name] for r in rows]) for name in (
            "Gemini Decisions","archived InLegalBERT excerpt MLP","current-20-percent")}
        models=expected|{name:np.asarray([p[r["row_sha256"]] for r in rows]) for name,p in pred.items()}
        np.testing.assert_array_equal(models["current-20-percent"],models["case model without edit"])
        gem=expected["Gemini Decisions"]==gold
        sizes=np.bincount(case_ids,minlength=len(use))
        draws=np.random.default_rng(78011).integers(len(use),size=(3000,len(use)))
        report={}
        for name,a in models.items():
            good=a==gold
            counts=np.bincount(case_ids,weights=good.astype(int)-gem.astype(int),minlength=len(use))
            ci=np.quantile(counts[draws].sum(1)/sizes[draws].sum(1),[.025,.975]).tolist()
            report[name]={"correct":int(good.sum()),"accuracy":float(good.mean()),"paired_accuracy_delta_vs_gemini_ci95":ci}
        out[panel]={"rows":len(rows),"unique_cases":len(use),"methods":report}
    answer={"panels":out,"episode":"The 384-case InLegalBERT edit and two refits evaluated on the previously archived identical isolated and paragraph inputs.",
            "predictions_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"same_frozen_gemini_requests":True,"new_gemini_api_calls":0}
    (HERE/"excerpt-report.json").write_text(json.dumps(answer,indent=2)+"\n")
    print({k:{m:s["correct"] for m,s in p["methods"].items()} for k,p in out.items()},flush=True)


if __name__=="__main__":
    main()
