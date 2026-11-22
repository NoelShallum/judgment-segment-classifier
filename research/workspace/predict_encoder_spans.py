"""Local inference from a pinned sub-billion encoder and a physically written seven-role profile."""
import argparse
import json
from pathlib import Path
from hashlib import file_digest
import sys

import numpy as np
from safetensors.numpy import load_file
from safetensors.torch import load_file as load_torch_file
import torch

from encoder_span_edit import Encoder, MODELS
from hier_model import HierModel, path_labels
from hier_trial import DONOR
from seven_paths import LABELS, ROOT
from span_model import merge, units


def segment(text, encoder, readout, mode, hierarchy=None):
    offsets=units(text)
    snippets=[text[left:right] for left,right in offsets]
    vectors=encoder.process([snippets],mode)[0]
    emissions=vectors@readout["classifier.weight"].T+readout["classifier.bias"]
    base_share=float(readout["donor_emission_scale"])
    if base_share:
        if hierarchy is None:
            raise ValueError("The written profile includes a contribution from the hierarchical donor.")
        words=hierarchy.words(snippets)
        emissions+=base_share*hierarchy.logits(hierarchy.contextualize(words))
    tags=path_labels(emissions,readout["crf.transitions"])
    return {"spans":merge(offsets,tags,LABELS)}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder",choices=MODELS,required=True)
    ap.add_argument("--task",choices=("whole","fragment"),required=True)
    ap.add_argument("--profile",choices=("selected","pure"),default="selected")
    ap.add_argument("--internal-edit",action="store_true")
    ap.add_argument("--text-file",type=Path)
    ap.add_argument("--threads",type=int,default=2)
    args=ap.parse_args()
    if args.internal_edit and args.profile!="selected":
        ap.error("The internal edit was selected with the combined profile.")
    torch.set_num_threads(args.threads)
    out=ROOT/"checkpoints"/("encoder-span-"+args.encoder)
    scores=ROOT/"results"/("encoder-span-"+args.encoder+".json")
    if not scores.is_file():scores=scores.with_name(scores.stem+"-accelerated.json")
    profile=json.loads(scores.read_text())["tasks"][args.task]["best" if args.profile=="selected" else "standalone_encoder"]
    tensor_file=out/(args.task+"-"+args.profile+".safetensors")
    with tensor_file.open("rb") as f:check=file_digest(f,"sha256").hexdigest()
    if check!=profile["weights_sha256"]:raise ValueError("Weights do not match the selected study profile")
    readout=load_file(str(tensor_file))
    enc=Encoder(args.encoder)
    if args.internal_edit:
        delta=load_torch_file(str(out/(args.task+"-final-normalization.safetensors")))
        with torch.no_grad():
            enc.model.final_norm.weight.copy_(delta["final_norm.weight"])
    hierarchy=HierModel(DONOR) if float(readout["donor_emission_scale"]) else None
    text=args.text_file.read_text() if args.text_file else sys.stdin.read()
    print(json.dumps(segment(text,enc,readout,profile["mode"],hierarchy),ensure_ascii=False))

if __name__=="__main__":
    main()
