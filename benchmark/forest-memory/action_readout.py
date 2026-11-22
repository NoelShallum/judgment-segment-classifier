"""Fit a fixed-partition estimate of the benefit of changing each role decision."""

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from forest_runtime import EYE, NONE, REASONING, ROLE_COUNT, is_pair, readout, tree_indices
from run_forest import (CONT, HERE, bank, evaluation, features, fit_values, leaf_moments, measure,
                        permitted, protocol, read, save_tensors, shuffle_within, size_of, summarize, write)
from audit import paired_total


DEST = HERE / "action"
ALTERNATIVES = {"rival-utility": ("rival-macro", False), "pair-utility": ("pair-macro", True)}
SHRINKAGE = (8., 48.)
PENALTIES = (.03, .12, .32)
MINIMUM_ADVANTAGES = (.015, .07, .16)
MAX_SCORE_GAPS = (.70, 1.5, 3.)


def utility_table(train):
    matrix = evaluation.matrix(train.gold, train.base)
    den = (matrix.sum(0) + matrix.sum(1)).clip(min=1)
    table = 2. * (EYE * den[None] - matrix.diagonal()[None]) / den[None] ** 2
    typical = np.median(np.diag(table))
    return (table / typical).astype(np.float32)


def outcomes(b, table, pair):
    value = table[b.gold] - table[b.gold, b.base][:, None]
    if pair:
        value = np.where(is_pair(b.base)[:, None] & is_pair(np.arange(ROLE_COUNT))[None, :], value, 0.)
    lengths = np.repeat(np.diff(b.ends), np.diff(b.ends))
    scale = (lengths / np.median(lengths)) ** -.15
    return value, (scale / scale.mean()).astype(np.float32)


def decisions(score, estimate, base, temp, params):
    penalty, minimum, max_gap, pair = (params[k] for k in ("penalty", "minimum", "max_gap", "pair_only"))
    row = np.arange(len(base))
    gaps = (score[row, base, None] - score) / temp
    z = np.where(gaps <= max_gap, estimate - penalty * gaps, -np.inf)
    if pair:
        z[:, ~is_pair(np.arange(ROLE_COUNT))] = -np.inf
    proposed = z.argmax(1)
    gain = np.maximum(z[row, proposed], 0.)
    allowed = gain > minimum
    if pair:
        allowed &= is_pair(base)
    return np.where(allowed, proposed, base).astype(np.int8), gain


def lock(archive):
    spec = protocol(archive)
    original = read(HERE / "freeze.json")
    receipt = {
        "source_freeze": evaluation.digest(HERE / "freeze.json"), "topology": ALTERNATIVES,
        "target": "First-order per-role macro-F1 change for reallocating prediction mass from the incumbent role to each possible role, derived only from the fitting confusion. Leaf means use fixed routing values and shrink toward zero.",
        "selection": "Same 48 cases and zero comparator. Require higher full/no-key F1, nonincreasing full/no-key pair errors, and accuracy above the existing floor. Freeze before opening the two confirmation groups for this follow-up.",
        "shrinkage": SHRINKAGE, "penalties": PENALTIES, "minimum_benefits": MINIMUM_ADVANTAGES, "maximum_score_gaps": MAX_SCORE_GAPS,
        "controls": "Equal-norm residual permutation within incumbent/rival identity plus zero.",
        "official_test_read": False,
    }
    p = DEST / "protocol.json"
    if p.exists():
        assert read(p) == json.loads(json.dumps(receipt))
    else:
        write(p, receipt)
    return spec, original


def fit(archive, spec, source):
    train, selector = (bank(archive, g, spec) for g in ("fit", "select"))
    assert not set(train.keys) & set(selector.keys)
    temp = source["temperature"]
    ti, w, r = features(train, temp)
    di, base, rival = features(selector, temp)
    old = evaluation.stats(selector, base)
    confusion_table = utility_table(train)
    cases = np.repeat(np.arange(len(train.keys)), np.diff(train.ends))
    order = shuffle_within(w, r)
    arms, eligible = {}, {}
    for method, (tree, pair) in ALTERNATIVES.items():
        item = source["arms"][tree]["files"]["edited"]
        file = HERE / item["file"]
        assert item["sha256"] == evaluation.digest(file)
        state = {k: v for k, v in load_file(file).items() if k != "leaf.weight"}
        t, d = tree_indices(ti, w, r, state), tree_indices(di, base, rival, state)
        values, mass = outcomes(train, confusion_table, pair)
        moment = leaf_moments(t, cases, values, mass, size_of(state))
        best, observed, grid = None, None, []
        for s in SHRINKAGE:
            out = readout(d, fit_values(moment, s))
            for p, a, g in product(PENALTIES, MINIMUM_ADVANTAGES, MAX_SCORE_GAPS):
                cfg = {"penalty": p, "minimum": a, "max_gap": g, "pair_only": pair}
                guess, _ = decisions(selector.scores, out, base, temp, cfg)
                if np.array_equal(guess, base):
                    continue
                score = evaluation.stats(selector, guess)
                rank = summarize(guess, base, score)
                passes = permitted(old, score)
                grid.append({"arm": {**cfg, "shrinkage": s}, "rank": rank, "eligible": passes})
                test = (rank, s, cfg, score)
                if passes and (best is None or rank > best[0]):
                    best = test
                if observed is None or rank > observed[0]:
                    observed = test
        if best is None and observed is None:
            arms[method] = {"no_changes": True}
            continue
        rank, s, cfg, tested = best or observed
        chosen = fit_values(moment, s)
        shuffle = fit_values(leaf_moments(t, cases, values[order], mass[order], size_of(state)), s)
        shuffle *= np.linalg.norm(chosen.astype(np.float64)) / max(1e-12, np.linalg.norm(shuffle.astype(np.float64)))
        files = {}
        for tag, weights in (("edited", chosen), ("shuffled", shuffle), ("zero", np.zeros_like(chosen))):
            path = DEST / (method + "-" + tag + ".safetensors")
            files[tag] = {"file": path.name, "sha256": save_tensors(path, {**state, "leaf.weight": weights}),
                          "norm": float(np.linalg.norm(weights.astype(np.float64)))}
        if best is not None:
            eligible[method] = rank
        arms[method] = {"eligible": best is not None, "parameters": {**cfg, "shrinkage": s}, "selection": tested, "files": files, "grid": grid}
        print("action selection", method, best is not None, rank, cfg, flush=True)
    write(DEST / "freeze.json", {"protocol_sha256": evaluation.digest(DEST / "protocol.json"), "selected": max(eligible, key=eligible.__getitem__) if eligible else "zero",
                                 "arms": arms, "temperature": temp, "fit_cases": train.keys.tolist(), "select_cases": selector.keys.tolist()})


def evaluate(archive, spec):
    frozen = read(DEST / "freeze.json")
    assert frozen["protocol_sha256"] == evaluation.digest(DEST / "protocol.json")
    groups = {}
    seen = set(frozen["fit_cases"] + frozen["select_cases"])
    for g in ("select", "confirm", "fresh"):
        b = bank(archive, g, spec)
        if g != "select":
            assert not seen & set(b.keys)
            seen.update(b.keys)
        x, old, rival = features(b, frozen["temperature"])
        scores = {"zero": measure(b, old, old, np.zeros(len(old), dtype=np.float32))}
        predictions = {"zero": old}
        for name, item in frozen["arms"].items():
            if "files" not in item:
                continue
            for suffix, info in item["files"].items():
                path = DEST / info["file"]
                assert info["sha256"] == evaluation.digest(path)
                params = load_file(path)
                z = tree_indices(x, old, rival, params)
                guess, gain = decisions(b.scores, readout(z, params["leaf.weight"]), old, frozen["temperature"], item["parameters"])
                if suffix == "zero":
                    np.testing.assert_array_equal(guess, old)
                    continue
                key = name if suffix == "edited" else name + "-shuffled"
                scores[key] = measure(b, old, guess, gain)
                for group, keep in (("all", np.ones(len(old), bool)), ("novel", b.novel)):
                    scores[key][group]["deltas_95"]["pair_errors"] = paired_total(b, old, guess, keep)
                predictions[key] = guess
                if g == "select" and suffix == "edited":
                    assert scores[key]["all"]["confusion"] == item["selection"]["all"]["confusion"]
        groups[g] = {"cases": len(b.keys), "rows": len(b.gold), "no_exact_key": int(b.novel.sum()), "methods": scores}
        np.savez_compressed(DEST / (g + "-predictions.npz"), case_ids=b.keys, offsets=b.ends, gold=b.gold, novel=b.novel, **predictions)
        print(g, {name: (round(z["all"]["macro_f1"] * 100, 4), round(z["novel"]["macro_f1"] * 100, 4), z["all"]["pair_errors"]) for name, z in scores.items()}, flush=True)
    write(DEST / "results.json", {"freeze_sha256": evaluation.digest(DEST / "freeze.json"), "selected": frozen["selected"],
                                 "groups": groups, "official_test_read": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("protocol", "fit", "score"))
    parser.add_argument("--bank", type=Path, default=CONT / "legal-context/calibration-banks.zip")
    args = parser.parse_args()
    DEST.mkdir(exist_ok=True)
    plan, state = lock(args.bank)
    if args.action == "fit":
        fit(args.bank, plan, state)
    elif args.action == "score":
        evaluate(args.bank, plan)
