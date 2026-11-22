"""Program role-pair transition evidence from training counts and evaluate frozen weights."""
import csv
from collections import Counter, defaultdict
from itertools import groupby, product
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from safetensors.numpy import load_file, save_file
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "astra-run-penalty"))
from run_entry import bootstrap, full_score, metric, prepare_score
from discourse_span_edit import text_features
from hier_full import CACHE, gold_cases
from hier_model import CRF, HierModel, path_labels
from hier_trial import DONOR, OUT
from preserved_emitter_edit import embeddings
from program_trial import TO_ID, save_json
from span_eval import counts, document
from transition_span_edit import sample_whole
from weight_edit import file_hash
from role_edge_runtime import FEATURE_MASS, decode, decompose, signatures, transition_scores

MIN_SUPPORT = 16
PRIOR_STRENGTH = 60.0
STRENGTHS = (.2, .5, 1., 2.)
MODE_OPTIONS = ("interaction", "joint")


def majority_roles(d):
    words = np.fromiter((m.start() for m in re.finditer(r"\S+", d["text"])), dtype=np.int32)
    raw_starts = [a for a, _ in d["units"]]
    source_starts = [a for a, _ in d["reference_units"]]
    units = np.searchsorted(raw_starts, words, side="right") - 1
    reference = np.searchsorted(source_starts, words, side="right") - 1
    labels = np.asarray(d["reference_roles"])[reference]
    histogram = np.bincount(7 * units + labels, minlength=7 * len(raw_starts)).reshape(-1, 7)
    return histogram.argmax(1)


def write_from_train():
    source = ROOT / "references/official/train.csv"
    stats = defaultdict(Counter)
    prior = np.zeros(49, dtype=np.int64)
    case_ids = set()
    n_units = 0
    n_rows = 0
    with source.open(encoding="utf-8-sig", newline="") as stream:
        for key, rows in groupby(csv.DictReader(stream), key=lambda r: r["Index"]):
            if key in case_ids:
                raise ValueError("Training rows must group contiguous case IDs")
            case_ids.add(key)
            source_rows = [(i, r["Text"], TO_ID[r["Label"]]) for i, r in enumerate(rows)]
            d = document(key, source_rows)
            labels = majority_roles(d)
            pair = 7 * labels[:-1] + labels[1:]
            prior += np.bincount(pair, minlength=49)
            for keys, relation in zip(signatures(d["text"], d["units"]), pair):
                for k in keys:
                    stats[k][int(relation)] += 1
            n_rows += len(source_rows)
            n_units += len(labels)
            if len(case_ids) % 1000 == 0:
                print("Read official training cases", len(case_ids), flush=True)
    vocab = {k: i+1 for i, k in enumerate(sorted(k for k, c in stats.items() if c.total() >= MIN_SUPPORT))}
    prior_probability = (prior + .5) / (prior.sum() + .5 * len(prior))
    table = np.zeros((len(vocab) + 1, 7, 7), np.float32)
    for k, i in vocab.items():
        c = stats[k]
        observed = np.array([c[j] for j in range(49)])
        posterior = (observed + PRIOR_STRENGTH * prior_probability) / (c.total() + PRIOR_STRENGTH)
        log_ratio = np.clip(np.log(posterior / prior_probability), -4., 4.)
        table[i] = FEATURE_MASS[k.split(':', 1)[0]] * log_ratio.reshape(7, 7)
    save_json(HERE / "vocab.json", vocab)
    save_file({"edge.log_ratio": table}, str(HERE / "role_edge.safetensors"))
    save_json(HERE / "train.json", {"cases": len(case_ids), "first_case": min(map(int,case_ids)),
              "last_case":max(map(int,case_ids)),"source_rows": n_rows, "inferred_units": n_units,
              "training_edges": int(prior.sum()), "prior_counts": prior.reshape(7,7).tolist(),
              "vocabulary": len(vocab), "parameters": int(table.size),
              "vocab_sha256": file_hash(HERE / "vocab.json"), "weights_sha256": file_hash(HERE / "role_edge.safetensors"),
              "source_sha256": file_hash(source), "min_support":MIN_SUPPORT, "prior_strength": PRIOR_STRENGTH,
              "feature_mass":FEATURE_MASS, "modes":list(MODE_OPTIONS), "strengths":list(STRENGTHS)})
    print("Weights written", len(vocab), n_units, flush=True)


def prepare(docs, split, model, parents, vocab, weights):
    if split in ("select", "confirm"):
        states = embeddings(model, docs, "raw-boundary-write-" + split)
        raw = [model.logits(s) for s in states]
    elif split == "independent":
        folder = ROOT / "astra-run-penalty/validation-cache"
        raw = [np.load(folder / (hashlib.sha256(d["text"].encode()).hexdigest() + ".npy")) for d in docs]
    else:
        raw = []
        for d in docs:
            with np.load(CACHE / (d["id"] + ".npz")) as package:
                raw.append(package["raw"])
    bank = []
    for i,(d,x) in enumerate(zip(docs,raw)):
        if len(x)!=len(d["units"]):raise ValueError(f"Mismatched case {d['id']}")
        item = prepare_score(d, x, parents)
        phi = transition_scores(d["text"], d["units"], vocab, weights)
        if i < 2:
            for name, p in parents.items():
                parent = path_labels(item.logits[name], p[CRF])
                np.testing.assert_array_equal(decode(item.logits[name],p[CRF],phi,0), parent)
                np.testing.assert_array_equal(item.score(parent)[:55],counts(d,d["units"],parent))
        bank.append((item, phi))
    print("Prepared", split, len(bank), flush=True)
    return bank


def independent_validation(excluded):
    pins=json.loads((ROOT/"astra-run-penalty/frozen-selection.json").read_text())
    ids=list(map(str, pins["selection_case_ids"]+pins["confirmation_case_ids"]))
    assert len(ids)==len(set(ids))==128 and not set(ids)&excluded
    buckets=defaultdict(list)
    with (ROOT/"references/official/val.csv").open(encoding="utf-8-sig", newline="") as f:
        subset=set(ids)
        for i,r in enumerate(csv.DictReader(f)):
            if r["Index"] in subset:
                buckets[r["Index"]].append((i,r["Text"],TO_ID[r["Label"]]))
    return [document(k,buckets[k]) for k in ids]


def score(bank, name, transition, mode, strength, shift=False):
    records=[]
    for item, phi in bank:
        before=item.logits[name]
        if not strength:
            path=path_labels(before,transition)
        else:
            residual=decompose(phi,mode)
            if shift and len(residual)>1:
                residual=np.roll(residual,max(len(residual)//2,1),axis=0)
            path=decode(before,transition,residual,strength)
        records.append(item.score(path))
    return np.stack(records)


def select(bank, parents):
    choices={}; grid={}
    for name, p in parents.items():
        base=score(bank,name,p[CRF],"joint",0.)
        optimum=metric(base.sum(0)); floor=optimum[1]-.01
        choice=("joint",0.);grid[name]=[[*choice,*optimum]]
        for mode, strength in product(MODE_OPTIONS,STRENGTHS):
            vector=score(bank,name,p[CRF],mode,strength)
            value=metric(vector.sum(0));grid[name].append([mode,strength,*value])
            if (name!="balanced" or value[1]>=floor) and value>optimum:
                optimum=value;choice=(mode,strength)
        choices[name]=list(choice)
    return choices,grid


def measure(bank,parents,choices):
    out={}
    for name,p in parents.items():
        mode,strength=choices[name]
        candidate=score(bank,name,p[CRF],mode,strength)
        parent=score(bank,name,p[CRF],mode,0.)
        controls={"unary":score(bank,name,p[CRF],"unary",strength),
                  "pooled_change":score(bank,name,p[CRF],"pooled",strength),
                  "misaligned":score(bank,name,p[CRF],mode,strength,shift=True)}
        out[name]={"selected":list(choices[name]),"cases":len(bank),"candidate":full_score(candidate),
                   "parent":full_score(parent),"paired_gap_95pct":bootstrap(candidate,parent),
                   "controls":{k:{"scores":full_score(v),"candidate_minus_control_95pct":bootstrap(candidate,v)} for k,v in controls.items()}}
    return out


def main():
    torch.set_num_threads(2)
    if not (HERE/"train.json").exists():write_from_train()
    train=json.loads((HERE/"train.json").read_text())
    assert file_hash(HERE/"role_edge.safetensors")==train["weights_sha256"]
    assert file_hash(HERE/"vocab.json")==train["vocab_sha256"]
    weights=load_file(str(HERE/"role_edge.safetensors"));vocab=json.loads((HERE/"vocab.json").read_text())
    parents={k:load_file(str(OUT/f"whole-discourse-{k}.safetensors")) for k in ("exact","balanced")}
    model=HierModel(DONOR)
    options=sample_whole()
    selection=prepare(options["select"],"select",model,parents,vocab,weights)
    choices,grid=select(selection,parents)
    frozen={"selection_case_ids":[d["id"] for d in options["select"]], "method":"Direct write of smoothed role-pair log likelihood ratios for lexical features at automatically proposed edges. Full joint and role-conditional change-only score variants are compared. Label each training unit by the dominant gold whitespace token count. No optimizer updates.","heldout_policy":"Frozen validation selection. Confirmation, independent validation, and all official test judgments are scored after selection. Balanced allows at most one point token accuracy loss on selection. The 712-case test has previously been inspected across experiments.",
            "weights_sha256":file_hash(HERE/"role_edge.safetensors"), "parents_sha256":{k:file_hash(OUT/f"whole-discourse-{k}.safetensors") for k in parents}, "choices": choices, "grid":grid}
    save_json(HERE/"frozen.json",frozen)
    print("Frozen",choices,flush=True)
    result={"train":train,"selection_sha256":file_hash(HERE/"frozen.json"),"profiles":choices,"splits":{}}
    result["splits"]["select"]=measure(selection,parents,choices)
    del selection
    excluded={d["id"] for ds in options.values() for d in ds}
    split_docs={"confirm":options["confirm"],"independent":independent_validation(excluded)}
    split_docs["test"]=[document(k,r) for k,r in gold_cases().items()]
    for split,docs in split_docs.items():
        bank=prepare(docs,split,model,parents,vocab,weights)
        result["splits"][split]=measure(bank,parents,choices)
        save_json(HERE/"results.json",result)
        print(split,[(k,m["candidate"]["exact_labeled_spans"]["f1"],m["parent"]["exact_labeled_spans"]["f1"]) for k,m in result["splits"][split].items()],flush=True)
        del bank

if __name__=="__main__":main()
