"""Write document and phase-specific role prevalence corrections."""
import argparse
from itertools import product
import json

import numpy as np
from safetensors.numpy import load_file, save_file

import sentence_minilm_whitened_write as mini
import sentence_pairwise_consensus as consensus
import sentence_pairwise_repair as repair
import sentence_phase_readout_write as phase
import sentence_referral_actual_replay as replay
from sentence_document_graph_write import OUT, cm, file_hash, paired, row_summary, save_json


P = repair.P
TAG = "sentence-role-prevalence"
MODEL = OUT / (TAG + ".safetensors")
PHASES = 5


def inputs(split, parent, weights, recipe):
    if split != "fit":
        return phase.view(split, parent, weights, recipe)
    docs = mini.docs(split)
    bank = mini.build(docs, split)
    lex = mini.independent_lexical(docs)
    base, unary, _ = repair.incumbent(bank, lex, recipe)["sequence only"]
    mode, scale, top, threshold, guard = parent["arms"]["sequence only"]["choice"]
    pair = repair.score_pairs(repair.dense_basis(bank, lex), weights, mode)
    contest = repair.proposals(pair, base, unary, scale, top)
    current = repair.decide(contest, base, np.zeros(len(base), bool), threshold, guard)
    return bank, lex, current, unary


def group_structure(bank, current, unary):
    offsets = bank[1]
    eye = np.eye(7, dtype=np.float32)
    posterior = np.exp(unary - unary.max(1, keepdims=True))
    posterior /= posterior.sum(1, keepdims=True)
    values = np.column_stack((eye[current], posterior, bank[4]))
    design, rates, sizes, index = [], [], [], np.empty((len(current), 2), np.int32)
    for a, b in zip(offsets, offsets[1:]):
        whole = values[a:b].mean(0)
        index[a:b, 0] = len(design)
        design.append(np.r_[whole, whole, np.log1p(b-a), 1., np.zeros(PHASES)])
        rates.append(whole[:7])
        sizes.append(b-a)
        for k, rows in enumerate(np.array_split(np.arange(a, b), PHASES)):
            if not len(rows):
                continue
            local = values[rows].mean(0)
            index[rows, 1] = len(design)
            design.append(np.r_[whole, local, np.log1p(b-a), 0., np.eye(PHASES)[k]])
            rates.append(local[:7])
            sizes.append(len(rows))
    return np.asarray(design), np.asarray(rates), np.asarray(sizes), index


def write(bank, x, current_rate, size, index):
    y = np.eye(7)[bank[2]]
    totals = np.zeros((len(x), 7))
    for k in range(2):
        np.add.at(totals, index[:, k], y)
    np.testing.assert_array_equal(totals.sum(1), size)
    np.testing.assert_allclose(current_rate.sum(1), 1., atol=1e-6)
    delta = (totals - size[:, None] * current_rate) / (size[:, None] + 8.)
    np.testing.assert_allclose(delta.sum(1), 0., atol=1e-6)
    center = x.mean(0)
    scale = 1 / np.maximum(x.std(0), .025)
    z = expand(x, center, scale)
    importance = np.sqrt(np.minimum(size, 100))
    gram = (z.T * importance) @ z
    link = (z.T * importance) @ delta
    beta = phase.solve(gram, link, .12)
    np.testing.assert_allclose(beta.sum(1), 0., atol=1e-6)
    return {"center": center, "scale": scale, "matrix": np.ascontiguousarray(beta)}


def expand(x, center, scale):
    return np.column_stack((np.clip((x - center) * scale, -5., 5.), np.ones(len(x))))


def proposed_bias(groups, w):
    x, current, _, index = groups
    delta = expand(x, w["center"], w["scale"]) @ w["matrix"]
    np.testing.assert_allclose(delta.sum(1), 0., atol=1e-5)
    base = current + .035
    nudged = np.maximum(base + delta, .018)
    ratio = np.clip(np.log(nudged / base), -.6, .6)
    return ratio[index[:, 0]], ratio[index[:, 1]]


def apply(current, unary, bias, params):
    share, strength, threshold, preserve = params
    row = np.arange(len(current))
    u = unary.copy()
    u[row, current] = np.maximum(u[row, current], u.max(1) + .005)
    u += strength * (share * bias[0] + (1-share) * bias[1])
    pred = u.argmax(1)
    accept = u[row, pred] > u[row, current] + threshold
    if preserve:
        accept &= (pred == 2) == (current == 2)
    return np.where(accept, pred, current).astype(np.int8)


def run(splits):
    parent, weights, parent_sha = consensus.inputs()
    recipe = repair.load_recipe()
    bank, _, current, unary = inputs("fit", parent, weights, recipe)
    assert not set(bank[0]) & set(parent["fit_cases"])
    w = write(bank, *group_structure(bank, current, unary))
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    save_file(w, str(MODEL))
    saved = load_file(str(MODEL))
    for name, value in w.items():
        np.testing.assert_array_equal(value, saved[name])
    w = saved
    dev, _, base, scores = inputs("select", parent, weights, recipe)
    assert not set(bank[0]) & set(dev[0])
    bias = proposed_bias(group_structure(dev, base, scores), w)
    baseline = row_summary(cm(dev[2], base))
    best = (baseline["macro_f1"], baseline["accuracy"])
    pick = (1., 0., 0., True)
    grid = []
    for option in product((0., .5, 1.), (.15, .35, .65, 1.), (0., .08, .20), (True, False)):
        pred = apply(base, scores, bias, option)
        stats = row_summary(cm(dev[2], pred))
        rank = (stats["macro_f1"], stats["accuracy"])
        grid.append([option, rank])
        if stats["accuracy"] >= baseline["accuracy"] - .002 and rank > best:
            best, pick = rank, option
    freeze = repair.receipt(TAG)
    save_json(freeze, {"parent_sha256": parent_sha, "model_sha256": file_hash(MODEL), "gradient_steps": 0,
              "fit_cases": bank[0].tolist(), "fit_rows": len(current), "selection_cases": dev[0].tolist(),
              "choice": pick, "selection_grid": grid})
    receipt = {"freeze_sha256": file_hash(freeze), "splits": {}}
    print("Selected", pick, "development", best, "base", baseline["macro_f1"], flush=True)
    for split in splits:
        data, _, old, logits = inputs(split, parent, weights, recipe)
        assert not set(bank[0]) & set(data[0])
        if split != "select":
            assert not set(dev[0]) & set(data[0])
        shift = proposed_bias(group_structure(data, old, logits), w)
        pred = apply(old, logits, shift, pick)
        pos = np.flatnonzero(pred != old)
        _, counts, trial = replay.scores(data, old, pred, pos)
        _, ref, base_trial = replay.scores(data, old, old, np.empty(0, np.int64))
        compare = paired(counts, ref)
        receipt["splits"][split] = {"cases": len(data[0]), "baseline": base_trial, "candidate": trial,
                                    "paired_to_base": compare}
        save_json(P.RESULT / (TAG + ".json"), receipt)
        print(split, len(pos), base_trial["labels"]["macro_f1"], trial["labels"]["macro_f1"],
              compare["macro_f1"], flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("splits", nargs="*")
    args = cli.parse_args()
    run(args.splits or ("select", "confirm", "holdout", "test"))
