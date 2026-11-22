"""Verify the T4 document replay and calculate case-paired comparisons."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
ROLES = ("None","Facts","Issue","Petitioner argument","Respondent argument","Reasoning","Decision")


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def cm(gold,pred):
    return np.bincount(gold.astype(int)*7+pred.astype(int),minlength=49).reshape(7,7)


def metrics(m):
    tp=np.diagonal(m,axis1=-2,axis2=-1)
    per=2*tp/(m.sum(-1)+m.sum(-2)).clip(min=1)
    return per.mean(-1),tp.sum(-1)/m.sum(axis=(-2,-1)),per


def main():
    package=HERE/"document-returned.zip"
    assert digest(package)=="7d0f114a065800f5ec88f24b8a6c99c0f624ec30982f55f773dc0424b2264533"
    with ZipFile(package) as z:
        assert z.testzip() is None
        expected=json.loads(z.read("SHA256SUMS.json"))
        assert set(z.namelist())==set(expected)|{"SHA256SUMS.json"}
        assert all(hashlib.sha256(z.read(k)).hexdigest()==v for k,v in expected.items())
        z.extractall(HERE/"document-returned")
    out=HERE/"document-returned"
    receipt=json.loads((out/"scores.json").read_text())
    assert receipt["freeze_sha256"]==digest(HERE/"freeze.json")
    assert receipt["device"]["devices"][0].split(",")[0].strip() in {"Tesla T4","NVIDIA T4","T4"}
    assert receipt["cached_baseline_prediction_differences"]==0
    plan=json.loads((ROUND/"round2-plan.json").read_text())
    with np.load(out/"predictions.npz",allow_pickle=False) as x, np.load(
            ROUND/"verified/query-results/fresh-predictions.npz",allow_pickle=False) as original:
        for k in ("case_ids","gold","offsets","review"):np.testing.assert_array_equal(x[k],original[k])
        np.testing.assert_array_equal(x["donor-baseline"],original["incumbent"])
        cases,starts,gold=x["case_ids"],x["offsets"],x["gold"]
        assert list(cases)==plan["groups"]["fresh"]
        roles=np.arange(7)
        draws=np.random.default_rng(220611).integers(len(cases),size=(4000,len(cases)))
        results={}; confusions={}
        baseline=x["donor-baseline"]
        for name in receipt["scores"]:
            pred=x[name];assert np.isin(pred,roles).all()
            case_cm=np.stack([cm(gold[a:b],pred[a:b]) for a,b in zip(starts,starts[1:])])
            confusions[name]=case_cm
            total=case_cm.sum(0)
            f,acc,per=metrics(total)
            assert abs(f-receipt["scores"][name]["macro_f1"])<1e-12
            changed=pred!=baseline
            results[name]={"macro_f1":float(f),"accuracy":float(acc),
                           "per_role_f1":dict(zip(ROLES,per.tolist())),
                           "none_to_reasoning":int(total[0,5]),"reasoning_to_none":int(total[5,0]),
                           "changed_vs_current":int(changed.sum()),
                           "corrected_vs_current":int(((baseline!=gold)&(pred==gold)).sum()),
                           "new_errors_vs_current":int(((baseline==gold)&(pred!=gold)).sum())}
        for name,item in results.items():
            reference="neurons28-baseline" if name.startswith("neurons28") else "donor-baseline"
            f,acc,_=metrics(confusions[name][draws].sum(1))
            rf,ra,_=metrics(confusions[reference][draws].sum(1))
            item["reference"]=reference
            item["paired_macro_f1_delta_ci95"]=np.quantile(f-rf,[.025,.975]).tolist()
            item["paired_accuracy_delta_ci95"]=np.quantile(acc-ra,[.025,.975]).tolist()
        select=json.loads((ROUND/"excerpt/capture-input/select-rows.json").read_text())
        selector_cases={str(row["case"]) for row in select}
        protected=[i for i,key in enumerate(cases) if str(key) not in selector_cases]
        sensitivity={}
        for name in results:
            m=confusions[name][protected].sum(0);f,acc,per=metrics(m)
            sensitivity[name]={"macro_f1":float(f),"accuracy":float(acc),"none_to_reasoning":int(m[0,5]),
                               "reasoning_to_none":int(m[5,0])}
    output={"source_sha256":digest(package),"frozen_sha256":digest(HERE/"freeze.json"),
            "note":"The targeted heads were selected on standalone sentences. This whole-judgment replay reuses validation cases, including 39 judgments represented in the selector. The sensitivity subset excludes those judgments and retains other previously inspected cases.",
            "judgments":len(cases),"sentences":len(gold),"bootstrap":"4,000 paired resamples by judgment",
            "scores":results, "excluding_selector_judgments":{"judgments":len(protected),"scores":sensitivity}}
    (HERE/"document-audit.json").write_text(json.dumps(output,indent=2)+"\n")
    for k,v in results.items():
        print(k,round(v["macro_f1"]*100,4),round(v["accuracy"]*100,4),
              v["none_to_reasoning"]+v["reasoning_to_none"],
              v["corrected_vs_current"],v["new_errors_vs_current"],v["paired_macro_f1_delta_ci95"])
    print("case-disjoint-from-selector",len(protected), {k:round(v["macro_f1"]*100,4) for k,v in sensitivity.items()})


if __name__ == "__main__":
    main()
