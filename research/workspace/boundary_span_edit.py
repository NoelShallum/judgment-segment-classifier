"""Check candidate-boundary coverage and evaluate expanded cuts with the frozen hierarchy."""
import argparse
import re
from pathlib import Path

from safetensors.numpy import load_file
import numpy as np
import torch

from fragment_trial import passages
from hier_full import gold_cases
from hier_model import CRF, HierModel
from hier_trial import DONOR, OUT
from program_trial import save_json
from seven_paths import ROOT
from span_eval import document
from transition_span_edit import sample_whole, upper_bound, features, scores, report
from span_eval import summarize

BEFORE_NUMBERED = re.compile(r"(?<=\s)(?=(?:\([ivxlcdm0-9]+\)|\d{1,3}[.)]|\([a-z]\))\s+)",re.I)
AFTER_SENTENCE = re.compile(r"[.!?](?:[”’\"')\]]*)\s+(?=[A-Za-z0-9\"“‘(\[])")
AFTER_CLAUSE = re.compile(r"[;:]\s+(?=[A-Z0-9\"“‘(\[])")
DISCOURSE = re.compile(r"(?<=\s)(?=(?:Learned counsel for (?:the )?|Counsel for (?:the )?|The question (?:before|which|is)|For the (?:foregoing|above) reasons|In the result|We accordingly)\b)")
ABBREVIATION = re.compile(r"(?:\b[A-Za-z]|\b(?:vs|art|sec|para|nos?|pvt|ltd|mr|mrs|ms|dr|inc|smt|etc|viz|shri|co)|(?:[A-Za-z]\.){2,})\.$",re.I)


def with_cuts(doc, policy):
    if policy=="current":
        return doc
    text=doc["text"]
    cuts={a for a,_ in doc["units"]}
    if policy in ("numbered","all"):
        cuts.update(m.start() for m in BEFORE_NUMBERED.finditer(text))
    if policy in ("sentence","all"):
        cuts.update(m.end() for m in AFTER_SENTENCE.finditer(text) if not ABBREVIATION.search(text[max(0,m.start()-30):m.start()+1]))
    if policy in ("clause","all"):
        cuts.update(m.end() for m in AFTER_CLAUSE.finditer(text))
    if policy in ("discourse","all"):
        cuts.update(m.start() for m in DISCOURSE.finditer(text))
    cuts=sorted(cuts|{len(text)})
    # Assign whitespace preceding a boundary to the prior unit, as the offsets protocol does.
    revised=[0]
    for cut in cuts[1:]:
        while cut<len(text) and text[cut].isspace():cut+=1
        if cut>revised[-1] and text[revised[-1]:cut].strip():revised.append(cut)
    if revised[-1]!=len(text):revised.append(len(text))
    return {**doc,"units":list(zip(revised,revised[1:]))}


POLICIES=("current","numbered","sentence","clause","discourse","all")

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coverage-only",action="store_true")
    args=ap.parse_args()
    whole_test=[document(k,v) for k,v in gold_cases().items()]
    fragment=passages()
    inventory={}
    for task,docs in [("whole",whole_test),("fragment",fragment["test"]),("matched",fragment["cloud"])]:
        inventory[task]={p:upper_bound([with_cuts(d,p) for d in docs]) for p in POLICIES}
        print(task,{k:round(v["reachable_gold_span_recall"],3) for k,v in inventory[task].items()},flush=True)
    save_json(ROOT/"results/boundary-span-coverage.json",inventory)
    if args.coverage_only:
        return
    torch.set_num_threads(2);m=HierModel(DONOR)
    outputs={}
    for name,groups in (("fragment",fragment),("whole",{**sample_whole(),"test":whole_test})):
        path=OUT/(("document" if name=="whole" else "fragment")+".safetensors")
        data=load_file(str(path));t=data[CRF] if name=="whole" else data["transition"]
        if name=="fragment" and not np.isclose(data["classifier.weight"][:,200:],0).all():
            raise ValueError("Incumbent fragment readout requires independent features")
        choice="current";best=(-1.,-1.);grid={}
        for p in POLICIES:
            docs=[with_cuts(d,p) for d in groups["select"]]
            values=features(m,docs)
            q=summarize(scores(docs,values,t).sum(0));key=q["exact_labeled_spans"]["f1"],q["token_accuracy"]
            grid[p]=q
            if key>best:best=key;choice=p
        print(name,choice,best,flush=True)
        out={}
        for fold,ds in groups.items():
            fitted=[with_cuts(d,choice) for d in ds]
            ft=features(m,fitted)
            previous=ft if choice=="current" else features(m,ds)
            out[fold]={"candidate":summarize(scores(fitted,ft,t).sum(0)),"incumbent":summarize(scores(ds,previous,t).sum(0)),"cases":len(ds)}
        outputs[name]={"selected":choice,"grid":grid,"results":out}
        save_json(ROOT/"results/boundary-span-validation.json",outputs)
        print(name,"test",out["test"]["candidate"]["exact_labeled_spans"]["f1"],flush=True)

if __name__=="__main__":
    main()
