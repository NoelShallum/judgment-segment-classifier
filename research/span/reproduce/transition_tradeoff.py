"""Evaluate a second operating point that preserves development token accuracy."""
import json
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file,save_file
import torch
from hier_full import CACHE,gold_cases
from hier_model import CRF,HierModel
from hier_trial import DONOR,OUT
from program_trial import save_json
from seven_paths import ROOT
from span_eval import document
from transition_span_edit import features,report,sample_whole,write_transition
from weight_edit import file_hash

def main():
    torch.set_num_threads(2)
    source=json.loads((ROOT/"results/role-specific-transitions.json").read_text())["tasks"]["whole"]
    dev=source["scores"]["select"]["previous"]
    options=[v for v in source["grid"] if v[-1]>=dev["token_accuracy"]-.01]
    best=max([[None,dev["exact_labeled_spans"]["f1"],dev["token_accuracy"]]]+[[v[:-2],v[-2],v[-1]] for v in options],
             key=lambda v: (v[1],v[2]))
    spec=best[0]
    model=HierModel(DONOR)
    pub=load_file(str(OUT/"document.safetensors"))
    old=pub[CRF]
    new=old if spec is None else write_transition(model.transitions,*spec)
    groups=sample_whole()
    groups["test"]=[document(k,v) for k,v in gold_cases().items()]
    results={}
    for key,docs in groups.items():
        if key=="test":
            xs=[]
            for d in docs:
                with np.load(CACHE/(d["id"]+".npz")) as f:xs.append(f["raw"])
        else:
            xs=features(model,docs)
        results[key]=report(docs,xs,new,old)
    if spec is not None:
        path=OUT/"role-specific-balanced.safetensors"
        save_file({**pub,CRF:new},str(path))
        sha=file_hash(path)
    else:
        sha=file_hash(OUT/"document.safetensors")
    save_json(ROOT/"results"/"transition-token-tradeoff.json",
              {"protocol":"Secondary operating point motivated by the first run's observed trade-off. On the existing development grid, maximize exact span F1 subject to losing at most one point of token accuracy. Confirmation and test quantify the trade-off. This is a follow-up on already reported cohorts.",
               "eligible_variant_count":len(options),"settings":spec,"weights_sha256":sha,"scores":results})
    print(spec,[(k,v["candidate"]["exact_labeled_spans"]["f1"],v["candidate"]["token_accuracy"]) for k,v in results.items()],flush=True)
if __name__=="__main__":
    main()
