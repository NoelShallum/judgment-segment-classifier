"""Write a CRF-conditioned margin correction into the published hierarchy emitter."""
import argparse
from collections import Counter
from itertools import product

import numpy as np
from safetensors.numpy import load_file, save_file
from scipy.linalg import solve
import torch

from discourse_span_edit import text_features
from hier_full import CACHE as RAW_TEST, gold_cases
from hier_model import CRF, EMITTER, LABELS, HierModel, path_labels
from hier_trial import DONOR, OUT
from preserved_emitter_edit import edit_documents, embeddings
from program_trial import save_json
from seven_paths import ROOT
from span_eval import counts, document, sets, summarize
from transition_span_edit import sample_whole
from weight_edit import file_hash


def map_roles(d):
    """Project gold spans onto raw units, retaining purity and exactly reachable endpoints."""
    gold = d["gold_spans"]
    starts = np.array([s["start"] for s in gold])
    ends = np.array([s["end"] for s in gold])
    tags = np.array([LABELS.index(s["label"]) for s in gold])
    assigned, purity, on_edge = [], [], []
    for a, b in d["units"]:
        first = max(0, np.searchsorted(starts, a, side="right") - 1)
        stop = np.searchsorted(starts, b, side="left")
        by_role = np.bincount(tags[first:stop],
                             weights=np.minimum(ends[first:stop], b) - np.maximum(starts[first:stop], a),
                             minlength=7)
        role = by_role.argmax()
        assigned.append(role)
        purity.append(by_role[role] / (b - a))
        on_edge.append((a > 0 and a == starts[first]) or (b < len(d["text"]) and b == ends[stop - 1]))
    return np.array(assigned), np.array(purity), np.array(on_edge)


def readout_data(d, x, model, parent):
    cue, pos = text_features(d)
    raw = model.logits(x)
    return np.column_stack((x, np.full(len(x), .5))), raw + cue @ parent["cue_head"] + pos @ parent["position_head"]


def sufficient_statistics(data, model, parent, rank_by_edge):
    keys, targets, weights = [], [], []
    trans = parent[CRF]
    for doc, feature in data:
        y, purity, edge = map_roles(doc)
        key, emission = readout_data(doc, feature, model, parent)
        pred = path_labels(emission, trans)
        previous = np.r_[1, y[:-1] + 3]
        following = np.r_[y[1:] + 3, 2]
        condition = emission + trans[previous, 3:] + trans[3:, following].T
        original = condition[np.arange(len(y)), y].copy()
        condition[np.arange(len(y)), y] = -np.inf
        rival = condition.argmax(1)
        error_scale = np.clip(.75 + condition[np.arange(len(y)), rival] - original, 0., 5.)
        wrong = pred != y
        error_scale *= wrong
        delta = np.zeros_like(emission)
        delta[np.arange(len(y)), y] = .5 * error_scale
        delta[np.arange(len(y)), rival] = -.5 * error_scale
        quality = np.where(purity >= .9, purity, 0.)
        if rank_by_edge:
            quality *= 1. + 2. * edge
        quality *= np.where(wrong, 1., .5)
        keys.append(key)
        targets.append(delta)
        weights.append(quality)
    x = np.concatenate(keys)
    y = np.concatenate(targets)
    w = np.concatenate(weights)
    gram = x.T @ (x * w[:, None])
    right = x.T @ (y * w[:, None])
    return gram, right, len(x), int(np.count_nonzero(w))


def compare(banks, gains, parent):
    counts_all = {}
    for k, records in banks.items():
        counts_all[k] = np.stack([
            counts(d, d["units"], path_labels(raw + x @ gains, parent[CRF]))
            for d, x, raw in records
        ])
    return counts_all


def paired(q, b):
    rng = np.random.default_rng(13453)
    draws = []
    for ids in rng.integers(len(q), size=(1800, len(q))):
        a, z = summarize(q[ids].sum(0)), summarize(b[ids].sum(0))
        draws.append([a["exact_labeled_spans"]["f1"] - z["exact_labeled_spans"]["f1"],
                      a["token_accuracy"] - z["token_accuracy"]])
    return np.quantile(draws, [.025, .975], axis=0).T.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-test", action="store_true")
    opt = parser.parse_args()
    torch.set_num_threads(2)
    model = HierModel(DONOR)
    train = edit_documents()
    fixed = list(zip(train, embeddings(model, train, "raw-emitter-train")))
    dev = sample_whole()
    bank = {k: list(zip(v, embeddings(model, v, "raw-boundary-write-" + k))) for k, v in dev.items()}
    receipt = {}
    zero = np.zeros((model.weight.shape[1]+1, 7), np.float32)
    for role in ("exact", "balanced"):
        parent = load_file(str(OUT / f"whole-discourse-{role}.safetensors"))
        built = {k: [(d, *readout_data(d, x, model, parent)) for d, x in v] for k, v in bank.items()}
        base = compare(built, zero, parent)
        base_metric = summarize(base["select"].sum(0))
        best = (base_metric["exact_labeled_spans"]["f1"], base_metric["token_accuracy"])
        choice = zero
        spec = None
        grid = []
        for by_edge in (False, True):
            gram, right, train_units, retained_units = sufficient_statistics(fixed, model, parent, by_edge)
            for reg in (.05, .5, 2.):
                kernel = solve(gram + reg * gram.trace()/len(gram) * np.eye(len(gram)), right, assume_a="pos")
                for gain in (.25, .5, 1.):
                    w = (gain * kernel).astype("f4")
                    q = compare({"select": built["select"]}, w, parent)["select"]
                    s = summarize(q.sum(0))
                    key = s["exact_labeled_spans"]["f1"], s["token_accuracy"]
                    grid.append([by_edge, reg, gain, *key])
                    if role == "balanced" and key[1] < base_metric["token_accuracy"] - .01:
                        continue
                    if key > best:
                        best, choice, spec = key, w, [by_edge, reg, gain]
        written = model.weight.copy()
        written[3:] += choice[:-1].T
        written_bias = model.bias.copy()
        written_bias[3:] += .5 * choice[-1]
        dest = OUT / f"whole-raw-conditioned-emitter-{role}.safetensors"
        save_file({**parent, EMITTER+"weight": written, EMITTER+"bias": written_bias}, str(dest))
        # Score the delta actually persisted at float32 precision.
        choice = np.vstack(((written[3:] - model.weight[3:]).T,
                            2. * (written_bias[3:] - model.bias[3:])))
        evals = compare(built, choice, parent)
        receipt[role] = {"selected": spec, "train_units": train_units, "weighted_units": retained_units,
                         "grid": grid, "patch_sha256": file_hash(dest),
                         "scores": {k: {"cases": len(q), "candidate": summarize(q.sum(0)),
                                        "parent": summarize(base[k].sum(0)), "gap_ci95": paired(q, base[k])}
                                    for k, q in evals.items()}}
        print(role, spec, {k: summarize(q.sum(0))["exact_labeled_spans"]["f1"]
                          for k,q in evals.items()}, flush=True)
    frozen = ROOT / "results/raw-conditioned-emitter-frozen.json"
    save_json(frozen, receipt)
    if not any(v["selected"] is not None for v in receipt.values()):
        return
    cases = ([document(k, rows) for k, rows in gold_cases().items()] if opt.all_test else sets()["test"])
    bank = embeddings(model, cases, "raw-emitter-full-test" if opt.all_test else "whole-test")
    for name, trial in receipt.items():
        path = OUT / f"whole-raw-conditioned-emitter-{name}.safetensors"
        weights = load_file(str(path))
        assert file_hash(path) == trial["patch_sha256"]
        a, b = [], []
        for d, x in zip(cases, bank):
            cues, positions = text_features(d)
            authored = cues @ weights["cue_head"] + positions @ weights["position_head"]
            with np.load(RAW_TEST / (d["id"] + ".npz")) as data:
                raw = data["raw"]
            if not np.allclose(raw, model.logits(x), atol=2e-5):
                raise ValueError("Raw feature cache differs from the full benchmark")
            z = x @ weights[EMITTER+"weight"][3:].T + weights[EMITTER+"bias"][3:] + authored
            a.append(counts(d, d["units"], path_labels(z, weights[CRF])))
            b.append(counts(d, d["units"], path_labels(raw + authored, weights[CRF])))
        q,p = np.stack(a),np.stack(b)
        trial["scores"]["test"] = {"cases":len(q),"candidate":summarize(q.sum(0)),
                                   "parent":summarize(p.sum(0)),"gap_ci95":paired(q,p)}
        print(name, trial["scores"]["test"]["candidate"]["exact_labeled_spans"]["f1"], flush=True)
    save_json(ROOT / ("results/raw-conditioned-emitter-" + ("full" if opt.all_test else "screen") + ".json"),
              {"profiles":receipt,"frozen_sha256":file_hash(frozen),
               "method":"Closed-form correction of local CRF-conditioned margins on errors, optionally emphasizing reachable boundaries. Statistics use official training cases and the same detected raw units as inference. No optimizer or backpropagation."})


if __name__ == "__main__":
    main()
