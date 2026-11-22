"""Apply the published seven-role hierarchy with a written discourse residual."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from safetensors.numpy import load_file
import torch

from hier_model import CRF,EMITTER,HierModel,LABELS,path_labels
from role_cues import cue_features
from span_model import merge, units


def infer(model, reference, text):
    offsets=units(text)
    snippets=[text[a:b] for a,b in offsets]
    places=np.array([(a+b)*.5/len(text) for a,b in offsets])
    position=np.eye(5)[np.digitize(places,[.05,.25,.60,.91])]
    words=model.words(snippets)
    complete=model.contextualize(words)
    if EMITTER+"weight" in reference:
        raw=complete@reference[EMITTER+"weight"][3:].T+reference[EMITTER+"bias"][3:]
    else:
        raw=model.logits(complete)
    if "context.mix" in reference:
        minimum,width,strength=reference["context.mix"]
        if strength and len(text.split())>=minimum:
            from context_span_runtime import contextual
            local=contextual(model,words,int(width))
            states=np.concatenate((complete,local),axis=1)
            raw=states@reference["context.emission_weight"].T+reference["context.emission_bias"]
    emissions=raw+cue_features(snippets)@reference["cue_head"]+position@reference["position_head"]
    if "boundary.weight" in reference:
        from boundary_runtime import boundary_features,variable_decode
        logits=boundary_features(raw)@reference["boundary.weight"]+reference["boundary.bias"]
        delta=np.clip(logits,-6,6)*reference["boundary.gain"]
        roles=variable_decode(emissions,reference[CRF],delta)
    else:
        roles=path_labels(emissions,reference[CRF])
    return {"spans":merge(offsets,roles,LABELS)}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir",type=Path,required=True)
    ap.add_argument("--profile",type=Path,required=True)
    ap.add_argument("--text-file",type=Path)
    ap.add_argument("--threads",type=int,default=2)
    a=ap.parse_args()
    text=a.text_file.read_text() if a.text_file else sys.stdin.read()
    torch.set_num_threads(a.threads)
    model=HierModel(a.model_dir)
    out=infer(model,load_file(str(a.profile)),text)
    print(json.dumps(out,ensure_ascii=False))

if __name__=="__main__":
    main()
