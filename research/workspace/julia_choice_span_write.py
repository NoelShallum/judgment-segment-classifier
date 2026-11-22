"""Write a seven-role calibration matrix for the open Julia 1 decision model."""
import hashlib
import json
import runpy
import time

import numpy as np
import onnxruntime as ort
from safetensors.numpy import load_file, save_file
from scipy.linalg import solve
from transformers import PreTrainedTokenizerFast

from fragment_trial import passages
from hier_model import CRF, LABELS, path_labels
from hier_trial import OUT
from program_trial import fixture_rows, save_json
from raw_emitter_write import paired
from role_fixtures import ANCHORS, CHECKS
from seven_paths import ROOT
from span_eval import counts, summarize
from weight_edit import file_hash

MODEL = ROOT / "references/julia-1-onnx"
OPTIONS = (
    "None: court headings, appearances, citations or other text with no assigned rhetorical role",
    "Facts: the case events, prior proceedings or background recounted by the court",
    "Issue: a question the court must resolve",
    "Petitioner argument: a contention or request made by the party bringing the case or appeal",
    "Respondent argument: a contention or objection made by the party answering the case or appeal",
    "Reasoning: the deciding court's analysis, law application or justification",
    "Decision: the court's operative ruling, relief, order or final disposition",
)
QUESTION = "What rhetorical role does the FOCUS passage play in this court judgment?"


class Julia:
    def __init__(self):
        config = json.loads((MODEL / "tokenizer_config.json").read_text())
        tokens = {k: config[k] for k in ("cls_token", "sep_token", "bos_token", "eos_token", "mask_token", "pad_token", "unk_token")}
        self.tok = PreTrainedTokenizerFast(tokenizer_file=str(MODEL / "tokenizer.json"), **tokens)
        self.encode = runpy.run_path(str(MODEL / "source-data.py"))["sequence"]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.model = ort.InferenceSession(str(MODEL / "model.onnx"), options,
                                          providers=["CPUExecutionProvider"])

    def cache(self, text):
        dest = ROOT / "cache/julia-choice-1"
        dest.mkdir(parents=True, exist_ok=True)
        outputs = {}
        missing = []
        for i, item in enumerate(text):
            name = hashlib.sha256(repr((item, QUESTION, OPTIONS)).encode()).hexdigest()
            cache = dest / (name + ".npy")
            if cache.exists():
                outputs[i] = np.load(cache, allow_pickle=False)
            else:
                state = {"state": item, "type": "choice", "question": QUESTION, "options": OPTIONS}
                instance = self.encode(self.tok, state, max_length=512, head_length=256, strict=False)
                missing.append((i, cache, instance))
        missing.sort(key=lambda x: len(x[2]["ids"]))
        for start in range(0,len(missing),8):
            group = missing[start:start+8]
            ids = np.full((len(group), max(len(z["ids"]) for _,_,z in group)),
                          self.tok.pad_token_id, dtype=np.int64)
            attention = np.zeros_like(ids)
            for j, (_,_,z) in enumerate(group):
                n = len(z["ids"])
                ids[j,:n] = z["ids"]
                attention[j,:n] = 1
            markers = np.array([z["markers"] for _,_,z in group], dtype=np.int64)
            feeds = {"input_ids":ids,"attention_mask":attention,"marker_pos":markers,
                     "marker_mask":np.ones_like(markers,bool),"qtype":np.zeros(len(ids),np.int64)}
            native = self.model.run(["logits"], feeds)[0]
            assert native.shape == (len(group),len(OPTIONS)) and np.isfinite(native).all()
            for (i,cache,_), row in zip(group,native):
                outputs[i] = row
                with cache.with_suffix(".part").open("wb") as write:
                    np.save(write,row,allow_pickle=False)
                cache.with_suffix(".part").replace(cache)
            if start % 64 == 0 or start + 8 >= len(missing):
                print("Native Julia requests",min(start+8,len(missing)),"/",len(missing),flush=True)
        return np.stack([outputs[i] for i in range(len(text))])


def encode_text(text, units):
    parts = [" ".join(text[a:b].split()) for a,b in units]
    return [
        "FOCUS: "+ q[:1450] + (("\nPREVIOUS CONTEXT: "+parts[i-1][-420:]) if i else "") +
        (("\nFOLLOWING CONTEXT: "+parts[i+1][:420]) if i + 1 < len(parts) else "")
        for i,q in enumerate(parts)
    ]


def program(scores, roles, damping):
    z = scores - scores.mean(1, keepdims=True)
    scale = z.std()
    x = np.column_stack((z / scale, np.ones(len(z))))
    # Use a small unit margin in the existing option logit space.
    target = np.eye(7)[roles] * 1.5
    weights = solve(x.T @ x + damping*len(x)*np.eye(x.shape[1]),
                    x.T @ (target-x[:,:7]), assume_a="pos")
    center = np.eye(7) - np.full((7,7),1/7)
    return (center @ weights[:-1]).astype("f4"), (weights[-1]*scale).astype("f4")


def measure(data, transition, w, bias, model_gain):
    return np.stack([counts(d,d["units"],path_labels(model_gain*(x @ w + bias), transition))
                     for d,x in data])


def main():
    start = time.monotonic()
    model = Julia()
    groups = passages()
    anchors = fixture_rows(ANCHORS) + fixture_rows(CHECKS)
    source = model.cache(["FOCUS: " + x[2] for x in anchors])
    y = np.array([x[3] for x in anchors])
    presets = {(float(d),float(g)):(np.eye(7,dtype=np.float32)+g*w,g*b)
               for d in (.05,.5) for w,b in [program(source,y,d)] for g in (.25,.5,1.)}
    presets[(0.,0.)] = np.eye(7,dtype=np.float32), np.zeros(7,dtype=np.float32)
    print("Authored keys",len(anchors),"native matches",int((source.argmax(1)==y).sum()),flush=True)
    base = load_file(str(OUT / "fragment-discourse-exact.safetensors"))[CRF]
    zero = np.zeros_like(base)
    def prepare(block):
        items = [encode_text(d["text"],d["units"]) for d in block]
        x = model.cache([s for group in items for s in group])
        return list(zip(block,np.split(x,np.cumsum(list(map(len,items)))[:-1])))
    def rank(v):
        return v["exact_labeled_spans"]["f1"],v["token_accuracy"]
    select = prepare(groups["select"][:24])
    grid = []
    best_by_type = {}
    for (d,g),(w,b) in presets.items():
        kind = "edited" if g else "native"
        for transition in (0., .5, 1.):
            for gain in (1.,2.):
                setting = d,g,transition,gain
                result = rank(summarize(measure(select,transition*base,w,b,gain).sum(0)))
                grid.append([*setting,*result])
                if kind not in best_by_type or result > best_by_type[kind][0]:
                    best_by_type[kind] = result,setting,w,b
    path = ROOT / "checkpoints/julia-choice-span.safetensors"
    locked = {}
    for kind,(_,spec,w,b) in best_by_type.items():
        locked[kind+".weight"] = (spec[3]*w).astype("f4")
        locked[kind+".bias"] = (spec[3]*b).astype("f4")
        locked[kind+".transition"] = (spec[2]*base).astype("f4")
    save_file(locked,str(path))
    frozen = ROOT / "results/julia-choice-frozen.json"
    save_json(frozen,{"sha256":file_hash(path),"settings":{k:v[1] for k,v in best_by_type.items()},
                      "validation_cases":[d["id"] for d,_ in select],"grid":grid})
    print("Frozen candidates", {k:v[1] for k,v in best_by_type.items()},flush=True)
    data = {"select":select,"confirm":prepare(groups["confirm"][:24]),"gemini_matched":prepare(groups["cloud"])}
    reference = json.loads((ROOT/"results/focused-span-gemini-replay.json").read_text())
    p = load_file(str(path))
    results = {}
    for kind,items in data.items():
        arrays = {k:measure(items,p[k+".transition"],p[k+".weight"],p[k+".bias"],1.) for k in best_by_type}
        raw = measure(items,zero,np.eye(7,dtype="f4"),np.zeros(7,dtype="f4"),1.)
        results[kind] = {"cases":len(items),"scores":{**{k:summarize(v.sum(0)) for k,v in arrays.items()},
                                                      "raw_native":summarize(raw.sum(0))},
                         "edited_native_ci95":paired(arrays["edited"],arrays["native"])}
        if kind == "gemini_matched":
            results[kind]["scores"]["gemini"] = reference["frozen_gemini_span_reference"]
        print(kind,{label:rank(m) for label,m in results[kind]["scores"].items()},flush=True)
    save_json(ROOT/"results/julia-choice-span-write.json",
              {"model_revision":"82a2fadf8fccfccdc5fd4e1009ba8f1a265eb7a8","parameters":144300000,
               "method":"Write a seven-by-seven role score matrix and bias in closed form from authored examples. Fixed Julia model produces option-conditioned logits for each detected passage unit with its immediate neighbors. Validation selects score calibration and CRF weight before twelve archived Gemini passages are read. No gradient steps.",
               "scores":results,"freeze_sha256":file_hash(frozen),"seconds":time.monotonic()-start})


if __name__ == "__main__":
    main()
