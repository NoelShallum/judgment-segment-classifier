"""Prospective confirmation of the actual text-conditioned bias on new-case outcomes."""
import hashlib
import json
import experiment as trial
from official import compute


def main():
    file="targeted-official-freeze.json"
    path=trial.HERE/file
    if not path.exists():
        prior_names=("shared","extended","separate-validation","transfer-confirm")
        prior_ids=set()
        for name in prior_names:
            prior_ids.update(json.loads((trial.HERE/(name+".json")).read_text())["ids"])
        prior_ids.update(json.loads((trial.HERE/"freeze.json").read_text())["ids"])
        salt="attoflow-conditional-native-confirmation-20261122-1400-v1"
        candidates=(str(k) for k in range(4984,6408) if str(k) not in prior_ids)
        ids=sorted(candidates,key=lambda k:hashlib.sha256((salt+"|"+k).encode()).digest())[:384]
        assert len(ids)==384 and not (set(ids)&prior_ids)
        spec={"strength":json.loads((trial.HERE/"freeze.json").read_text())["gain"],"primary":"targeted","salt":salt,"validation_ids":ids,
              "previous_case_count":len(prior_ids),"fit_sha256":trial.file_hash(trial.HERE/"fit.json"),"previous_results":prior_names,
              "gate":"Positive lower endpoint of the paired case-bootstrap change in seven-class macro-F1 and better accuracy versus donor; primary also exceeds both constant and shuffled controls on macro-F1. Choose no additional parameters. Run exactly one full official test for the native candidate upon passing."}
        trial.write(path,spec)
    spec=json.loads(path.read_text())
    confirmation=compute(spec["validation_ids"],"val","targeted-new-validation",spec,file)
    a=confirmation["arms"]
    passed=(a["targeted"]["paired_macro_delta_ci95"][0]>0 and a["targeted"]["accuracy"]>a["donor"]["accuracy"] and
            a["targeted"]["macro_f1"]>max(a["constant"]["macro_f1"],a["shuffled"]["macro_f1"]))
    print("Text-conditioned native gate",passed,flush=True)
    if passed:
        test=compute([str(k) for k in range(6408,7120)],"test","official-targeted",spec,file)
        assert test["cases"]==712 and test["rows"]==149868


if __name__=="__main__":
    main()
