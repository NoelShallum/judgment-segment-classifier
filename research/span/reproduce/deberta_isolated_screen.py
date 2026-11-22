"""Finish the isolated DeBERTa screen while contextual embeddings are computed."""
import hashlib

import numpy as np
import torch

import encoder_span_edit as study
from deberta_span_write import compare_discourse, readouts
from program_trial import fixture_rows, official_sample, save_json
from role_fixtures import ANCHORS, CHECKS
from seven_paths import ROOT
from span_eval import sets


def main():
    torch.set_num_threads(1)
    study.MODES = ("isolated",)
    folder = study.CACHE / "deberta-v3-xsmall" / "isolated"

    def load(texts):
        digest = hashlib.sha256(repr(texts).encode()).hexdigest()
        return np.load(folder/(digest+".npy"),allow_pickle=False)

    examples,_,_ = official_sample("train",128)
    crafted = fixture_rows(ANCHORS) + fixture_rows(CHECKS)
    h = {}
    for name,rows in (("official train",examples),("authored",crafted)):
        h[name] = list(readouts(np.stack([load([r[2]])[0] for r in rows]),
                                np.array([r[3] for r in rows]),name))
    groups = {"whole":{k:{str(d["id"]):d for d in v} for k,v in sets().items() if k!="cloud"}}
    bank = {f"whole/{split}/{k}":load([d["text"][a:b] for a,b in d["units"]])
            for split,v in groups["whole"].items() for k,d in v.items()}
    print("Loaded frozen isolated activations",len(bank),"documents",flush=True)
    r = study.evaluate("deberta-isolated",groups,{"isolated":bank},{"isolated":h["official train"]+h["authored"]})
    base = compare_discourse(groups)
    save_json(ROOT/"results/deberta-isolated-screen.json",
        {"model":study.MODELS["deberta"],"encoder_parameters":70682112,
         "validation_selected":r,"current_discourse_control":base,"authored_keys":len(crafted),
         "source_keys":{"isolated":len(examples)},"steps":0,
         "method":"Each detected unit runs independently through the frozen bidirectional DeBERTa. Two sources of role keys write a seven-way readout. Units are then scored jointly by the same type of sequence decoder. This isolated encoder arm was prespecified with the parallel contextual arm."})
    for k,v in r["whole"].items():
        if k!="grid":print(k,v["edit"],v["scores"]["test"]["candidate"]["exact_labeled_spans"]["f1"],flush=True)


if __name__ == "__main__":
    main()
