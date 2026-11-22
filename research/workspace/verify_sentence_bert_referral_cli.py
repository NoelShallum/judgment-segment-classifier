"""Check the normal experimental predictor on real judgments against scored outputs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

import sentence_bert_referral_write as experiment
from hier_model import LABELS


def main():
    arm = experiment.ARMS[2]
    rows, cache, receipts = experiment.encoded("confirm",(arm,))
    frozen = experiment.read(experiment.TAG+"-frozen")
    variant,*_,gain,margin = frozen["frozen_modes"][arm]
    source = experiment.OUT/(experiment.TAG+"-2.safetensors")
    weights = experiment.load_file(str(source))
    estimates = experiment.response(experiment.design(rows,cache[arm],variant,weights),weights)
    pred = experiment.apply(rows,estimates,gain,margin)
    choices = sorted((len(row[2]),i) for i,row in enumerate(rows)
                     if len(row[3])>=3 and np.any(pred[i]!=row[2]))[:2]
    assert len(choices)==2
    checked = []
    for n,idx in choices:
        row = rows[idx]
        expected = {"labels":[LABELS[i] for i in pred[idx]],"referred_sentences":len(row[3])}
        with tempfile.TemporaryDirectory(dir=experiment.ROOT/"qa") as loc:
            input_file = Path(loc)/"input.json"
            input_file.write_text(json.dumps(row[-1]))
            command = [sys.executable,str(experiment.ROOT/"sentence_bert_referral_write.py"),
                       "predict","--input",str(input_file),"--device","CPU"]
            output = json.loads(subprocess.check_output(command,env=os.environ,text=True))
        assert output==expected,row[0]
        checked.append({"case":row[0],"sentences":n,"referred":len(row[3]),
                        "changed_parent":int(np.count_nonzero(pred[idx]!=row[2]))})
        print("Live CPU predictor reproduced",checked[-1],flush=True)
    experiment.save_json(experiment.FOLDER/(experiment.TAG+"-cli-qa.json"),{
        "verified_cases":checked,"source_sha256":experiment.file_hash(source),
        "freeze_sha256":experiment.file_hash(experiment.FOLDER/(experiment.TAG+"-frozen.json")),
        "gpu_reference_feature_sha256":receipts[arm]["features_sha256"]})


if __name__ == "__main__":
    main()
