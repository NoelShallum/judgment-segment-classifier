"""Measure warmed CPU latency of whole-judgment editing profiles."""
import argparse
import json
import platform
import sys
import time

import numpy as np
from safetensors.numpy import load_file
import torch

from hier_full import gold_cases
from hier_model import HierModel
from hier_trial import DONOR
from program_trial import save_json
from seven_paths import ROOT
from span_eval import document
from predict_discourse_spans import infer as parent_infer

EDGE = ROOT/"astra-role-edge"
sys.path.insert(0,str(EDGE))
from predict_role_edge import infer, infer_segments, infer_sentences


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--segment",action="store_true")
    mode.add_argument("--sentence-uncertainty",action="store_true")
    mode.add_argument("--sentence-joint",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(2)
    cases = gold_cases()
    qa = json.loads((EDGE/"qa.json").read_text())
    vocabulary = json.loads((EDGE/"vocab.json").read_text())
    model = HierModel(DONOR)
    patch = load_file(str(EDGE/"role-edge-exact.safetensors"))
    policy = {"parent":lambda t:parent_infer(model,patch,t),
              "edited":lambda t:infer(t,model,patch,vocabulary,patch,"joint",float(patch["edge.gain"]))}
    output="role-edge-cpu-latency.json"
    if args.segment:
        weights=load_file(str(ROOT/"checkpoints/hierarchical/whole-segment-weights.safetensors"))
        profiles={"exact":patch,"balanced":load_file(str(EDGE/"role-edge-balanced.safetensors"))}
        policy={"role-edge exact":policy["edited"],
                "segment-exact":lambda t:infer_segments(t,model,profiles,vocabulary,weights)}
        output="segment-exact-cpu-latency.json"
    supplied=args.sentence_uncertainty or args.sentence_joint
    if supplied:
        directory=ROOT/"checkpoints/hierarchical"
        names={"sentence-uncertainty":"known-sentence-uncertainty.safetensors"}
        if args.sentence_joint:
            names["sentence-joint"]="known-sentence-joint-portable.safetensors"
        else:
            names["sentence-calibrated"]="known-sentence-label-change.safetensors"
        compared={k:load_file(str(directory/v)) for k,v in names.items()}
        def configured(p):
            return lambda t:infer_sentences(t,model,p,vocabulary,p,"joint",float(p["edge.gain"]))
        policy={k:configured(v) for k,v in compared.items()}
        output=("sentence-joint" if args.sentence_joint else "sentence-uncertainty")+"-cpu-latency.json"
    records = []
    for q in qa["cases"]:
        case = q["case"]
        doc=document(case,cases[case])
        text=doc["text"]
        payload=[text[a:b] for a,b in doc["reference_units"]] if supplied else text
        times = {k:[] for k in policy}
        for predict in policy.values():
            predict(payload)
        for cycle in range(3):
            order = list(policy) if cycle % 2 == 0 else list(reversed(policy))
            for k in order:
                now = time.perf_counter()
                result = policy[k](payload)
                if supplied:
                    assert len(result["labels"])==len(payload)
                else:
                    assert result["spans"][-1]["end"] == len(text)
                times[k].append(time.perf_counter()-now)
        records.append({"case":case,"whitespace_tokens":q["whitespace_tokens"],
                        "candidate_units":len(payload) if supplied else q["candidate_units"],
                        "warmed_seconds":times,"medians":{k:float(np.median(v)) for k,v in times.items()}})
        print(records[-1],flush=True)
    endpoint="Timings include supplied-sentence encoding and classification." if supplied else "Timings include text boundary detection and output spans."
    save_json(ROOT/"results"/output,
              {"platform":platform.machine(),"torch_threads":2,"cases":records,
               "protocol":f"Already-loaded donated hierarchy and weights. Paired alternating order, three measurements per profile, one warm-up. {endpoint} Other experiment processes were running on this laptop."})


if __name__ == "__main__":
    main()
