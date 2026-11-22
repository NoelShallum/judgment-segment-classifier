"""Check the dense CPU predictor against frozen benchmark predictions."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

import sentence_dense_readout_write as experiment
from hier_model import LABELS


def main():
    experiment.torch.set_num_threads(2)
    plan_file = experiment.FOLDER/(experiment.TAG+"-frozen.json")
    plan = experiment.read(experiment.TAG+"-frozen")
    arm = max(plan["arms"],key=lambda a:(plan["arms"][a]["selection_macro_f1"],plan["arms"][a]["selection_accuracy"]))
    runtime = experiment.load_predictor(arm)
    weights,spec = runtime[-2:]
    bank = experiment.prepare("confirm")
    estimate = experiment.output(experiment.infer_basis(bank[-1],arm,weights),weights)
    _,_,_,*action = spec["selection"]
    pred = experiment.replace(bank[3],bank[4],estimate,*action)
    docs = {d["id"]:d for d in experiment.groups_for("confirm")}
    eligible = sorted((b-a,k,a,b) for k,a,b in zip(bank[0],bank[1],bank[1][1:]) if np.any(pred[a:b]!=bank[3][a:b]))
    assert len(eligible)>=3
    checked = []
    for i,(n,k,a,b) in enumerate((eligible[0],eligible[len(eligible)//2],eligible[-1])):
        item = docs[k]
        inputs = [item["text"][p:q].strip() for p,q in item["reference_units"]]
        expected = {"labels":[LABELS[j] for j in pred[a:b]]}
        if i==0:
            with TemporaryDirectory(dir=experiment.ROOT/"qa") as temp:
                data = Path(temp)/"input.json"
                data.write_text(json.dumps(inputs))
                cli = [sys.executable,str(experiment.ROOT/"sentence_dense_readout_write.py"),
                       "predict","--arm",arm,"--input",str(data)]
                actual = json.loads(subprocess.check_output(cli,text=True))
        else:
            actual = experiment.predict(inputs,arm,runtime)
        assert actual==expected,k
        checked.append({"case":str(k),"sentences":int(n),"edited_labels":int(np.count_nonzero(pred[a:b]!=bank[3][a:b]))})
        print("Verified",checked[-1],flush=True)
    preserved = experiment.preserve_issues(bank[3],pred)
    guard_candidates = [(b-a,k,a,b) for k,a,b in zip(bank[0],bank[1],bank[1][1:])
                        if np.any(preserved[a:b]!=pred[a:b])]
    n,k,a,b = min(guard_candidates)
    item = docs[k]
    inputs = [item["text"][p:q].strip() for p,q in item["reference_units"]]
    with TemporaryDirectory(dir=experiment.ROOT/"qa") as temp:
        data = Path(temp)/"input.json"
        data.write_text(json.dumps(inputs))
        cli = [sys.executable,str(experiment.ROOT/"sentence_dense_readout_write.py"),
               "predict","--arm",arm,"--input",str(data),"--preserve-issues"]
        guarded = json.loads(subprocess.check_output(cli,text=True))
    assert guarded=={"labels":[LABELS[j] for j in preserved[a:b]]}
    guard_qa = {"case":str(k),"sentences":int(n),"preserved":int(np.count_nonzero(preserved[a:b]!=pred[a:b]))}
    output = {"freeze_sha256":experiment.file_hash(plan_file),"arm":arm,"weights_sha256":spec["sha256"],
              "backbone_parameters":runtime[0].parameters,"cases":checked,"issue_guard":guard_qa}
    experiment.save_json(experiment.FOLDER/(experiment.TAG+"-cli-qa.json"),output)


if __name__=="__main__":
    main()
