"""Closed-form native BERT head edits for the None/Reasoning margin.

The selector is separate from the two diagnostic evaluation groups. Earlier
experiments inspected their outcomes, so those checks are exploratory.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from safetensors.numpy import load_file
from scipy.linalg import cho_factor, cho_solve

HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
sys.path.insert(0, str(ROUND / "excerpt"))
from edit import (HERE as EXCERPT, ORIGINAL, Group, Tail, confusion, get_group, metrics,
                  paired, read, save_patch, score, sha, verify_protocol, write)
from head_followup import new_sample, pool_edit

NONE, REASONING = 0, 5
GAINS = (0.25, 0.5, 1.0)
REGULARIZATIONS = (0.03, 0.3, 3.0)
TARGET_MARGINS = (0.5, 1.5)
PROTECTION = (1.0, 4.0)


def sources():
    data, features = EXCERPT / "capture-input", EXCERPT / "excerpt-captured.zip"
    plan, capture = verify_protocol(data, features, EXCERPT / "frozen-capture-protocol.json")
    assert sha(ORIGINAL) == plan["donor_sha256"]
    first = read(EXCERPT / "results/freeze.json")
    patch = EXCERPT / "results/role.safetensors"
    assert sha(patch) == first["weights"]["role"]["sha256"]
    return data, features, plan, capture, load_file(str(patch))


def fit_groups(data, features, plan, capture, patch, tail):
    groups = [get_group(data, features, plan, capture, name) for name in ("fit", "select")]
    banks = {"donor": groups}
    banks["neurons28"] = [replace(group, pooled=pool_edit(group, tail, patch)) for group in groups]
    return banks


def ridge_basis(group):
    inputs = group.pooled.astype(np.float64)
    mean = inputs.mean(0)
    scale = np.maximum(inputs.std(0), 0.025)
    return np.column_stack(((inputs - mean) / scale, np.ones(len(inputs)))), mean, scale


def expected_adjustment(group, tail, margin, labels=None):
    labels = group.labels[group.owners] if labels is None else labels
    logits = tail.class_scores(group.pooled)
    direction = np.where(labels == REASONING, 1.0, -1.0)
    included = (labels == NONE) | (labels == REASONING)
    current_margin = logits[:, REASONING] - logits[:, NONE]
    response = direction * np.clip(margin - direction * current_margin, 0, 4.0)
    return np.where(included, response, 0.0), included


def factors(group, X, response, included, protection):
    weights = np.where(included, 1.0, protection) * group.window_weights()
    gram = X.T @ (X * weights[:, None])
    rhs = X.T @ (response * weights)
    penalty = np.diag(gram).mean()
    return gram, rhs, penalty


def solve_delta(gram, rhs, penalty, ridge, mean, scale):
    matrix = gram.copy()
    matrix.flat[::len(matrix)+1] += ridge * penalty
    matrix[-1, -1] -= 0.99 * ridge * penalty
    coef = cho_solve(cho_factor(matrix, check_finite=False), rhs, check_finite=False)
    weight = coef[:-1] / scale
    bias = coef[-1] - weight @ mean
    return np.asarray(weight, np.float32), np.float32(bias)


def scores_with_delta(group, tail, delta_weight, delta_bias, gain):
    logits = tail.class_scores(group.pooled)
    change = gain * (group.pooled @ delta_weight + delta_bias)
    logits[:, REASONING] += 0.5 * change
    logits[:, NONE] -= 0.5 * change
    return logits


def classifier_patch(tail, weight, bias, gain):
    W, b = tail.wc.copy(), tail.bc.copy()
    W[REASONING] += 0.5 * gain * weight
    W[NONE] -= 0.5 * gain * weight
    b[REASONING] += 0.5 * gain * bias
    b[NONE] -= 0.5 * gain * bias
    return {"classifier.weight": W, "classifier.bias": b}


def pair_errors(group, predictions):
    y = group.labels
    return {"none_as_reasoning": int(((y == NONE) & (predictions == REASONING)).sum()),
            "reasoning_as_none": int(((y == REASONING) & (predictions == NONE)).sum()),
            "total": int((((y == NONE) & (predictions == REASONING)) |
                          ((y == REASONING) & (predictions == NONE))).sum())}


def fit():
    HERE.mkdir(exist_ok=True, parents=True)
    if (HERE / "freeze.json").exists():
        raise ValueError("The targeted experiment is already frozen.")
    data, features, plan, capture, patch = sources()
    tail = Tail(ORIGINAL)
    banks = fit_groups(data, features, plan, capture, patch, tail)
    frozen, outcome = {}, {}
    for name, (train, dev) in banks.items():
        X, mean, scale = ridge_basis(train)
        baseline = dev.predict(tail.class_scores(dev.pooled))
        initial = score(dev, baseline)
        best = (initial["eligible_population_reweighted_macro_f1"],
                initial["eligible_population_reweighted_accuracy"])
        choice = None
        grid = []
        for margin in TARGET_MARGINS:
            target, included = expected_adjustment(train, tail, margin)
            for protection in PROTECTION:
                gram, rhs, pen = factors(train, X, target, included, protection)
                for reg in REGULARIZATIONS:
                    w, b = solve_delta(gram, rhs, pen, reg, mean, scale)
                    for gain in GAINS:
                        pred = dev.predict(scores_with_delta(dev, tail, w, b, gain))
                        stat = score(dev, pred)
                        key = (stat["eligible_population_reweighted_macro_f1"],
                               stat["eligible_population_reweighted_accuracy"])
                        record = {"margin": margin, "protection": protection, "ridge": reg, "gain": gain,
                                  "weighted_f1": key[0], "weighted_accuracy": key[1],
                                  "balanced_f1": stat["macro_f1"], "pair_errors": pair_errors(dev, pred)}
                        grid.append(record)
                        if key > best:
                            best, choice = key, (record, w.copy(), b)
        if choice is None:
            frozen[name] = {"selected": "identity", "selection": initial, "grid": grid}
            print(name, "identity selected", best, flush=True)
            continue
        config, weight, bias = choice
        # Shuffle labels only within the two targeted roles. Match the full joint
        # classifier weight and bias delta norm to the selected program.
        labels = train.labels[train.owners].copy()
        positions = np.flatnonzero((labels == NONE) | (labels == REASONING))
        labels[positions] = np.random.default_rng(654713).permutation(labels[positions])
        target, included = expected_adjustment(train, tail, config["margin"], labels)
        gram, rhs, pen = factors(train, X, target, included, config["protection"])
        sw, sb = solve_delta(gram, rhs, pen, config["ridge"], mean, scale)
        target_norm = np.linalg.norm(np.r_[weight, bias].astype(np.float64))
        shuffle_norm = max(np.linalg.norm(np.r_[sw, sb].astype(np.float64)), 1e-12)
        ratio = target_norm / shuffle_norm
        selected = classifier_patch(tail, weight, bias, config["gain"])
        shuffle = classifier_patch(tail, sw * ratio, np.float32(sb * ratio), config["gain"])
        files = {}
        for kind, contents in (("targeted", selected), ("shuffled", shuffle)):
            path = HERE / (name + "-" + kind + ".safetensors")
            files[kind] = {"sha256": save_patch(path, contents)}
        baseline_W, baseline_b = tail.wc, tail.bc
        norm = lambda v: float(np.linalg.norm(np.r_[v["classifier.weight"].ravel()-baseline_W.ravel(),
                                                    v["classifier.bias"]-baseline_b].astype(np.float64)))
        assert abs(norm(selected)-norm(shuffle)) < 1e-5
        frozen[name] = {"selected": config, "selection_baseline": initial, "files": files, "grid": grid,
                        "matched_delta_l2": [norm(selected), norm(shuffle)]}
        outcome[name] = best
        print(name, config, flush=True)
    result = {"method": "Closed-form ridge fit of a signed None-vs-Reasoning margin correction into the native two classifier rows.",
              "backbone": "The pretrained LegalSeg checkpoint and its previously frozen 28-neuron variant.",
              "training_and_selector": {"train_sentences": 4900, "select_sentences": 1400,
                  "case_and_exact_text_disjoint": True, "selector_objective": "Macro-F1 on the balanced selector reweighted to the full eligible selector role distribution, with accuracy tiebreak. Identity is eligible."},
              "protocol_note": "All comparison cohorts were inspected in previous research. The 28-neuron variant was chosen for this new analysis after inspecting its natural-sample result. Evaluation is exploratory.",
              "gradient_steps": 0, "donor_sha256": sha(ORIGINAL), "neuron28_sha256": sha(EXCERPT / "results/role.safetensors"),
              "source_plan_sha256": sha(data / "protocol.json"), "program_sha256": sha(Path(__file__)),
              "arms": frozen}
    write(HERE / "freeze.json", result)
    print("frozen", sha(HERE / "freeze.json"), outcome, flush=True)


def metrics_for(group, predictions, baseline):
    result = score(group, predictions)
    result["none_reasoning"] = pair_errors(group, predictions)
    result["vs_baseline"] = paired(group, baseline, predictions)
    return result


def evaluate():
    freeze = read(HERE / "freeze.json")
    assert freeze["program_sha256"] == sha(Path(__file__))
    data, features, plan, capture, patch = sources()
    assert freeze["source_plan_sha256"] == sha(data / "protocol.json")
    tail = Tail(ORIGINAL)
    balanced = get_group(data, features, plan, capture, "confirm")
    a = SimpleNamespace(natural_data=EXCERPT/"natural-input", natural_features=EXCERPT/"natural-captured.zip",
                        data=data, initial=EXCERPT/"results")
    natural, _, _ = new_sample(a)
    arrays, results = {}, {}
    for name, group in (("balanced_2100", balanced), ("natural_4900", natural)):
        results[name] = {"rows": len(group.rows), "judgments": len({r["case"] for r in group.rows}), "arms": {}}
        arrays[name + "/gold"] = group.labels
        arrays[name + "/id"] = np.array([r["id"] for r in group.rows])
        for backbone in ("donor", "neurons28"):
            pool = group.pooled if backbone == "donor" else pool_edit(group, tail, patch)
            baseline = group.predict(tail.class_scores(pool))
            control = freeze["arms"][backbone]
            results[name]["arms"][backbone] = {"baseline": score(group, baseline), "baseline_pair": pair_errors(group, baseline)}
            arrays[name + "/" + backbone + "/baseline"] = baseline
            for kind in ("targeted", "shuffled"):
                if control["selected"] == "identity":
                    pred = baseline
                else:
                    path = HERE / (backbone + "-" + kind + ".safetensors")
                    assert sha(path) == control["files"][kind]["sha256"]
                    weights = load_file(str(path))
                    pred = group.predict(pool @ weights["classifier.weight"].T + weights["classifier.bias"])
                results[name]["arms"][backbone][kind] = metrics_for(group, pred, baseline)
                arrays[name + "/" + backbone + "/" + kind] = pred
                print(name, backbone, kind, round(100*results[name]["arms"][backbone][kind]["macro_f1"], 3),
                      round(100*results[name]["arms"][backbone][kind]["accuracy"], 3),
                      pair_errors(group, pred), flush=True)
    predictions = HERE / "predictions.npz"
    np.savez_compressed(predictions, **arrays)
    write(HERE / "results.json", {"freeze_sha256": sha(HERE / "freeze.json"),
          "prediction_sha256": sha(predictions), "exploratory_reuse": True, "groups": results})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fit", "evaluate"))
    args = parser.parse_args()
    {"fit": fit, "evaluate": evaluate}[args.action]()
