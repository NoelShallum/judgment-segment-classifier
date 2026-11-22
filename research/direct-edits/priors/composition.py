"""Score a preregistered composition of the 200-value forget edit and seven emission biases."""
import json
import sys
import numpy as np
import torch
from safetensors.numpy import load_file
import experiment as p
from system import h95

sys.path.insert(0,str(p.HERE.parent/"hierarchy"))
from retention_replay import load_arm


def run(split,ids,spec):
    torch.set_num_threads(1)
    base=load_arm("native-base-240-71.safetensors",spec["head"])
    gate=load_arm("uniform-retention-plus-head.safetensors",spec["head_and_gates"])
    cov=load_file(str(p.HERE/"priors.safetensors"))
    assert p.file_hash(p.HERE/"priors.safetensors")==spec["prior"]
    arms=["head","gate","head and text bias","gate and text bias","gate and shuffled bias","gate and constant bias"]
    case_map={}
    for k,text,gold in p.cases(split,set(ids)):
        words=base.words(text)
        f=p.targets(text,cov)
        arm_pred={}
        for model,label,pairs in [(base,"head",[("targeted","head and text bias")]),
                                  (gate,"gate",[("targeted","gate and text bias"),("shuffled","gate and shuffled bias"),("constant","gate and constant bias")])]:
            states=model.contextualize(words)
            raw=model.logits(states)
            pred=p.path_labels(raw,model.transitions)
            arm_pred[label]=pred
            for name,target in pairs:
                bias=p.adapted_bias({**f,"prediction":pred},name,spec["gain"],model)
                score=states@model.weight[3:].T+bias[3:]
                arm_pred[target]=p.path_labels(score,model.transitions)
        case_map[k]=(gold,arm_pred)
        if len(case_map)%64==0:print(split,len(case_map),len(ids),flush=True)
    assert set(case_map)==set(ids)
    counts={key:np.stack([p.cm(case_map[k][0],case_map[k][1][key]) for k in ids]) for key in arms}
    stats={key:p.stat(value) for key,value in counts.items()}
    primary=spec["primary"]
    gaps={key:h95(value,counts[primary]) for key,value in counts.items() if key!=primary}
    n=sum(len(case_map[k][0]) for k in ids)
    ans={"reserved_manifest_sha256":p.file_hash(p.HERE/"composition-freeze.json"),"ids":ids,"judgments":len(ids),"rows":n,
         "scores":stats,"paired_delta_ci95_for_primary":gaps}
    target=p.HERE/("composition-"+split+"-predictions.npz")
    with target.with_suffix(".part").open("wb") as f:
        np.savez_compressed(f,keys=ids,offsets=np.cumsum([0]+[len(case_map[k][0]) for k in ids]),
                            gold=np.concatenate([case_map[k][0] for k in ids]),
                            **{key:np.concatenate([case_map[k][1][key] for k in ids]) for key in arms})
    target.with_suffix(".part").replace(target)
    ans["prediction_sha256"]=p.file_hash(target)
    p.write(p.HERE/("composition-"+split+".json"),ans)
    print("composition",split,n,{k:round(v["macro_f1"],5) for k,v in stats.items()},flush=True)
    return ans


if __name__=="__main__":
    spec=json.loads((p.HERE/"composition-freeze.json").read_text())
    r=run("val",spec["reserved_ids"],spec)
    v=r["scores"]
    primary=v[spec["primary"]]
    components=("gate","head and text bias")
    passed=all(r["paired_delta_ci95_for_primary"][k][0]>0 and primary["accuracy"]>=v[k]["accuracy"] for k in components)
    passed&=all(primary["macro_f1"]>v[k]["macro_f1"] for k in ("gate and constant bias","gate and shuffled bias"))
    print("Composition gate",passed,flush=True)
    if passed:
        run("test",[str(i) for i in range(6408,7120)],spec)
