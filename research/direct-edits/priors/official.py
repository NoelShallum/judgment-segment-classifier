"""Freeze the constant-prior follow-up and require independent validation before test replay."""
import json
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file
import experiment as p
from system import h95


def compute(ids,split,stage,spec,receipt_name="constant-official-freeze.json"):
    bank,model=p.make_cache(stage,ids,load_file(str(p.HERE/"priors.safetensors")),dataset=split)
    records={}
    base,pred0,_=p.scores(bank,model,0.,"constant")
    records["donor"]=p.stat(base)
    gold=np.concatenate([r["g"] for r in bank])
    allpred={"donor":pred0}
    allbias={}
    for arm in ("constant","targeted","shuffled"):
        cm,pred,bias=p.scores(bank,model,spec["strength"],arm)
        changed=pred!=pred0
        records[arm]={**p.stat(cm),"paired_macro_delta_ci95":h95(base,cm),"changed":int(changed.sum()),
                      "corrected":int(np.sum(changed&(pred==gold))),"broken":int(np.sum(changed&(pred0==gold)))}
        allpred[arm]=pred
        allbias[arm]=bias
    # Verify saved backbone activations against direct execution for a whole case.
    fresh_id,text,_=next(p.cases(split,{ids[0]}))
    primary=spec["primary"]
    b=allbias[primary][0]
    model.bias=b.copy()
    fresh=p.path_labels(model.logits(model.contextualize(model.words(text))),model.transitions)
    np.testing.assert_array_equal(fresh,allpred[primary][:len(text)])
    result={"cases":len(ids),"rows":len(gold),"ids":ids,"roles":p.LABELS,"split":split,"stage":stage,"arms":records,
            "freeze_sha256":p.file_hash(p.HERE/receipt_name),"native_replay":{"id":fresh_id,"arm":primary,"rows":len(text),"label_disagreements":0},
            "source_sha256":p.file_hash(p.ROOT/"references/official"/(split+".csv"))}
    target=p.HERE/(stage+"-predictions.npz")
    with target.with_suffix(".part").open("wb") as f:
        np.savez_compressed(f,keys=ids,offsets=np.cumsum([0]+[len(r["g"]) for r in bank]),gold=gold,**allpred)
    target.with_suffix(".part").replace(target)
    result["prediction_sha256"]=p.file_hash(target)
    p.write(p.HERE/(stage+".json"),result)
    print(stage,len(gold),[(k,round(v["macro_f1"],5),round(v["accuracy"],5)) for k,v in records.items()],flush=True)
    return result


def main():
    prior=p.HERE/"constant-official-freeze.json"
    orig=json.loads((p.HERE/"freeze.json").read_text())
    shared=json.loads(p.MANIFEST.read_text())["extended_validation_judgment_ids"]
    val=json.loads((p.ROOT/"results/sentence-graph-holdout-frozen.json").read_text())["cases"]
    omit=set(orig["ids"])|set(shared)
    ids=[k for k in val if k not in omit]
    spec={"strength":orig["gain"],"fit_source":p.file_hash(p.HERE/"fit.json"),"judgment_ids_for_confirmation":ids,"primary":"constant",
          "rules":"Before replaying official test, the frozen constant-prior native bias write must improve seven-role macro F1 on the validation subset, its paired judgment-bootstrap lower 95% endpoint must be positive, and its accuracy must improve. The primary is chosen after considering the earlier 48, which are removed here. Compare to native donor. Strong system reported separately.",
          "test_ids_if_gate_passes":[str(i) for i in range(6408,7120)]}
    if prior.exists(): assert json.loads(prior.read_text())==spec
    else:p.write(prior,spec)
    validation=compute(ids,"val","separate-validation",spec)
    c=validation["arms"]["constant"]
    if c["paired_macro_delta_ci95"][0]>0 and c["accuracy"]>validation["arms"]["donor"]["accuracy"]:
        result=compute(spec["test_ids_if_gate_passes"],"test","official-constant",spec)
        assert result["rows"]==149868 and result["cases"]==712
    else:print("Native constant-prior gate did not pass.",flush=True)


if __name__=="__main__":
    main()
