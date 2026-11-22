"""Replay frozen lexical readout decisions through the experimental CPU command."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

import sentence_phrase_readout as phrase
from hier_model import LABELS
from sentence_document_graph_write import file_hash,save_json,torch


def main():
    torch.set_num_threads(2)
    path=phrase.RESULT/(phrase.TAG+"-frozen.json")
    plan=json.loads(path.read_text())
    arm=max(phrase.ARMS[1:],key=lambda k:(plan["arms"][k]["select_macro_f1"],plan["arms"][k]["select_accuracy"]))
    state=phrase.load_predictor(arm)
    bank=phrase.dense.prepare("confirm")
    z=phrase.cached_scores("confirm")
    model,settings=state[-2:]
    answers=phrase.predict_rows(bank,phrase.dense.output(phrase.design(bank,z,arm,settings[0]),model),settings) if settings[4] else bank[3]
    cases={d["id"]:d for d in phrase.docs("confirm")}
    ranked=sorted((b-a,k,a,b) for k,a,b in zip(bank[0],bank[1],bank[1][1:]) if np.any(answers[a:b]!=bank[3][a:b]))
    if len(ranked)<3:
        ranked=sorted((b-a,k,a,b) for k,a,b in zip(bank[0],bank[1],bank[1][1:]))
    samples=[ranked[0],ranked[len(ranked)//2],ranked[-1]]
    proof={"freeze_sha256":file_hash(path),"arm":arm,"cases":[]}
    for i,(n,k,a,b) in enumerate(samples):
        d=cases[k]
        utterances=[d["text"][lo:hi].strip() for lo,hi in d["reference_units"]]
        expected={"labels":[LABELS[j] for j in answers[a:b]]}
        if not i:
            with TemporaryDirectory(dir=phrase.ROOT/"qa") as temp:
                src=Path(temp)/"judgment.json"
                src.write_text(json.dumps(utterances))
                cli=[sys.executable,str(phrase.ROOT/"sentence_phrase_readout.py"),"predict","--arm",arm,"--input",str(src)]
                result=json.loads(subprocess.check_output(cli,text=True))
        else:
            result=phrase.predict(utterances,arm,state)
        assert result==expected,k
        entry={"case":str(k),"sentences":int(n),"changes":int(np.count_nonzero(answers[a:b]!=bank[3][a:b]))}
        proof["cases"].append(entry)
        print("Matched live",entry,flush=True)
    save_json(phrase.RESULT/(phrase.TAG+"-cli-qa.json"),proof)


if __name__=="__main__":
    main()
