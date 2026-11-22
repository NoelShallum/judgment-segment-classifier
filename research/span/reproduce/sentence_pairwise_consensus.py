"""Reconcile separately written role-pair margins into seven class scores."""
import argparse
from itertools import product
import json

import numpy as np
from safetensors.numpy import load_file

import sentence_pairwise_repair as repair
from sentence_document_graph_write import OUT, cm, file_hash, measure, save_json

P = repair.P
TAG = "sentence-pairwise-consensus"
PAIRS = np.asarray(repair.PAIRS)
INCIDENCE = np.eye(7)[PAIRS[:, 1]] - np.eye(7)[PAIRS[:, 0]]


def inputs():
    source = repair.receipt(repair.TAG)
    frozen = json.loads(source.read_text())
    weights = OUT / (repair.TAG + ".safetensors")
    assert file_hash(weights) == frozen["checkpoint_sha256"]
    assert frozen["sources"] == {tag: file_hash(repair.receipt(tag)) for tag in repair.SOURCES}
    return frozen, load_file(str(weights)), file_hash(source)


def reconcile(pair_score, unary, method):
    """Project pairwise margins onto consistent seven-role log-score differences."""
    if method == "uniform":
        return (pair_score @ INCIDENCE / 7).astype(np.float32)
    prior = np.exp(unary - unary.max(1, keepdims=True))
    prior /= prior.sum(1, keepdims=True)
    weight = np.sqrt(prior[:, PAIRS[:, 0]] * prior[:, PAIRS[:, 1]]) + .008
    # The weighted graph Laplacian, with a weak diagonal regularizer to
    # select a unique score vector while retaining the pair differences.
    ga = weight[:, :, None] * INCIDENCE
    laplacian = np.einsum("pi,npj->nij", INCIDENCE, ga)
    rhs = np.einsum("np,npi->ni", pair_score, ga)
    laplacian[:, np.arange(7), np.arange(7)] += .012
    return np.linalg.solve(laplacian, rhs[..., None]).squeeze(-1).astype(np.float32)


def scores(x, weights, bases):
    pair = {mode: repair.score_pairs(x, weights, mode) for mode in repair.MODES}
    return {name: {(mode, rule): reconcile(pair[mode], unary, rule)
                   for mode, rule in product(repair.MODES, ("uniform", "posterior weighted"))}
            for name, (_, unary, _) in bases.items()}


def decide(power, base, unary, protected, scale, threshold, guard):
    row = np.arange(len(base))
    delta = (power - power[row, base, None]) - scale * np.maximum(unary[row, base, None] - unary, 0.)
    return repair.decide(delta, base, protected, threshold, guard)


def fit():
    prior, weight, previous = inputs()
    bank, x, bases = repair.views("select", repair.load_recipe())
    assert bank[0].tolist() == prior["select_cases"]
    combined = scores(x, weight, bases)
    # Pair differences originating from a consistent synthetic vector
    # project back to that vector up to an additive offset.
    rng = np.random.default_rng(1367)
    v = rng.normal(size=(90, 7))
    v -= v.mean(axis=1, keepdims=True)
    pair = v @ INCIDENCE.T
    np.testing.assert_allclose(reconcile(pair, v, "uniform"), v, atol=2.e-7)
    w = reconcile(pair, v, "posterior weighted")
    assert np.mean((w - v) ** 2) < .004
    arms = {}
    for name, (base, unary, protection) in bases.items():
        before = measure(cm(bank[2], base))
        best = (float(before[0]), float(before[1]), 0)
        choice = None
        history = []
        for (mode, rule), alt in combined[name].items():
            for strength, threshold, guard in product((.08, .18, .35, .6, 1.0), (.01, .06, .13, .26), (False, True)):
                p = decide(alt, base, unary, protection, strength, threshold, guard)
                m, a, *_ = measure(cm(bank[2], p))
                outcome = (float(m), float(a), -int(np.count_nonzero(p != base)))
                option = (mode, rule, strength, threshold, guard)
                history.append([option, outcome])
                if outcome > best and a >= before[1] - .001:
                    best, choice = outcome, option
        arms[name] = {"choice": choice, "selection": best, "trials": history}
    dest = repair.receipt(TAG)
    save_json(dest, {"repair_freeze_sha256": previous, "selection_cases": bank[0].tolist(),
                     "gradient_steps": 0, "arms": arms})
    print("Frozen", {k: (q["choice"], q["selection"]) for k, q in arms.items()}, flush=True)


def evaluate(splits):
    prior, weight, source_hash = inputs()
    plan = repair.receipt(TAG)
    frozen = json.loads(plan.read_text())
    assert source_hash == frozen["repair_freeze_sha256"]
    recipe = repair.load_recipe()
    whole = {"freeze_sha256": file_hash(plan), "splits": {}}
    for split in splits:
        bank, x, bases = repair.views(split, recipe)
        assert not set(prior["fit_cases"]) & set(bank[0])
        if split != "select":
            assert not set(frozen["selection_cases"]) & set(bank[0])
        record = whole["splits"][split] = {"cases": len(bank[0]), "arms": {}}
        for name, (base, unary, protection) in bases.items():
            pick = frozen["arms"][name]["choice"]
            p = base
            if pick:
                mode, rule, strength, threshold, guard = pick
                z = reconcile(repair.score_pairs(x, weight, mode), unary, rule)
                p = decide(z, base, unary, protection, strength, threshold, guard)
                np.testing.assert_array_equal(p[protection], base[protection])
                if guard:
                    np.testing.assert_array_equal(p == 2, base == 2)
            r = record["arms"][name] = repair.metrics(bank, p, base)
            if split == "select":
                assert r["labels"]["macro_f1"] == frozen["arms"][name]["selection"][0]
        save_json(P.RESULT / (TAG + ".json"), whole)
        print(split, {k: (r["baseline"]["macro_f1"], r["labels"]["macro_f1"],
                          r["baseline"]["accuracy"], r["labels"]["accuracy"], r["changed"])
                      for k, r in record["arms"].items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("fit", "evaluate"))
    parser.add_argument("splits", nargs="*")
    args = parser.parse_args()
    if args.action == "fit":
        fit()
    else:
        evaluate(args.splits or repair.COHORTS)
