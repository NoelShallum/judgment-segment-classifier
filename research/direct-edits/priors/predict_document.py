"""Run the seven-role case-conditioned LegalSeg hierarchy on a CPU."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file,save_file

from hier_model import HierModel,LABELS,path_labels
from prior_model import adapted_bias,targets


class Predictor:
    def __init__(self, directory, method="document-prior"):
        if method not in ("document-prior","retention","retention-prior","source"):
            raise ValueError("Unknown method")
        directory=Path(directory)
        manifest=json.loads((directory/"model.json").read_text())
        for name,expected in manifest["files"].items():
            with (directory/name).open("rb") as stream:
                digest=hashlib.file_digest(stream,"sha256").hexdigest()
            if digest!=expected:
                raise ValueError(f"Checkpoint checksum mismatch for {name}")
        self.model=HierModel(directory)
        self.prior=load_file(str(directory/"priors.safetensors"))
        self.strength=manifest["strength"]
        self.method=method
        if method in ("retention","retention-prior"):
            change=load_file(str(directory/"retention.safetensors"))
            self.model.weight=change["emitter.hidden2tag.weight"]
            self.model.bias=change["emitter.hidden2tag.bias"]
            with torch.no_grad():
                for direction in ("bias_ih_l0","bias_ih_l0_reverse"):
                    getattr(self.model.doc,direction).copy_(torch.from_numpy(change["emitter.lstm."+direction]))

    def predict(self, value):
        single=isinstance(value,str)
        if not single and not isinstance(value,list):
            raise ValueError("Supply one string or an ordered list of strings.")
        given=[value] if single else value
        if not given or any(not isinstance(x,str) or not x.strip() for x in given):
            raise ValueError("Every input unit must contain text.")
        parts=[s.strip() for s in given]
        state=self.model.contextualize(self.model.words(parts))
        native=path_labels(self.model.logits(state),self.model.transitions)
        if self.method in ("document-prior","retention-prior"):
            evidence=targets(parts,self.prior)
            bias=adapted_bias({**evidence,"prediction":native},"targeted",self.strength,self.model)
            scores=state@self.model.weight[3:].T+bias[3:]
            pred=path_labels(scores,self.model.transitions)
        else:
            pred,bias=native,self.model.bias
        labels=[LABELS[i] for i in pred]
        if single:
            return {"label":labels[0]},bias
        starts=np.flatnonzero(np.r_[True,pred[1:]!=pred[:-1]]).tolist()
        spans=[{"start_unit":a,"end_unit_exclusive":b,"label":labels[a]} for a,b in zip(starts,starts[1:]+[len(parts)])]
        return {"labels":labels,"spans":spans},bias


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input",type=Path,help="JSON file containing one string or an ordered array of sentences or paragraphs.")
    parser.add_argument("--model-dir",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--method",choices=("document-prior","retention","retention-prior","source"),default="document-prior")
    parser.add_argument("--bias-output",type=Path,help="Save this document's exact ten-entry native output bias as safetensors.")
    parser.add_argument("--threads",type=int,default=2)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    result,bias=Predictor(args.model_dir,args.method).predict(json.loads(args.input.read_text()))
    if args.bias_output:
        save_file({"emitter.hidden2tag.bias":np.ascontiguousarray(bias)},str(args.bias_output))
    print(json.dumps(result,ensure_ascii=False))
