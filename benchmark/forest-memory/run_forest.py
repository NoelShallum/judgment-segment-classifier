"""Write signed role residuals into fixed random partitions, then replay validation."""

import argparse
from collections import defaultdict
import csv
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
from safetensors.numpy import load_file, save_file

from forest_runtime import (EYE, NONE, REASONING, ROLE_COUNT, decide, design, is_pair, projected, readout,
                            row_probabilities, tree_indices)


HERE = Path(__file__).resolve().parent
CONT = HERE.parent
ROOT = CONT.parent
CALIBRATION = CONT / "doc-calibration"
sys.path.append(str(CALIBRATION))
import calibrate as evaluation  # Reuse gold/case alignment, no-key masks and score metrics.

GROUPS = ("fit", "select", "confirm", "fresh")
SHRINKAGE = (6., 32., 106.)
GAINS = (.18, .38, .70, 1.1, 1.65)
MINIMUM_GAPS = (0., .025, .075)
QUANTILES = (.18, .34, .50, .66, .82)
SEED = 651105
ACCURACY_FLOOR = .0025
TOPOLOGIES = {
    "axis": {"trees": 48, "depth": 4, "width": 1, "with_rival": False, "seed": SEED + 1},
    "oblique-rival": {"trees": 64, "depth": 5, "width": 3, "with_rival": True, "seed": SEED + 2},
    "pair-oblique": {"trees": 56, "depth": 4, "width": 3, "with_rival": False, "seed": SEED + 3},
}
ARMS = {
    "axis-posterior": ("axis", "posterior"),
    "axis-macro": ("axis", "macro"),
    "rival-macro": ("oblique-rival", "macro"),
    "pair-macro": ("pair-oblique", "pair"),
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, record):
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def save_tensors(file, params):
    temp = file.with_suffix(".part")
    save_file({key: np.ascontiguousarray(value) for key, value in params.items()}, temp)
    loaded = load_file(temp)
    for key, value in params.items():
        np.testing.assert_array_equal(loaded[key], value)
    temp.replace(file)
    return evaluation.digest(file)


def protocol(archive):
    # The zip also contains a test member. Read the development file names and hashes only.
    with ZipFile(archive) as package:
        manifest = json.loads(package.read("manifest.json"))
        banks = {g: manifest[f"scorebank-{g}.npz"] for g in GROUPS}
    main = CONT / "legal-context/legal-results.zip"
    with ZipFile(main) as artifact:
        frozen = sha256(artifact.read("freeze.json")).hexdigest()
        assert json.loads(artifact.read("freeze.json"))["selected"] == "doc-semantic"
    spec = {
        "score_banks": banks, "doc_semantic_freeze_sha256": frozen,
        "development_reference_sha256": evaluation.digest(main),
        "cohorts": {g: n for g, n in zip(GROUPS, (192, 48, 72, 96))},
        "train_csv_sha256": evaluation.digest(ROOT / "references/official/train.csv"),
        "masks": {g: evaluation.digest(CONT / "scale" / (g + "-mask.npz")) for g in GROUPS[1:]},
        "fitting": "Fixed random partition layouts. Train-only input normalization and unsupervised cut points. Independent, sample-weighted residual means shrink toward zero, with zero values on fewer than three distinct fitting judgments. There are no gradient updates or tree optimization.",
        "features": "Final role scores, rival and original labels, four donor probability streams, PCA features, relative case position, and left/right score differences in within-case neighborhoods.",
        "partitions": TOPOLOGIES, "arms": ARMS, "threshold_quantiles": QUANTILES,
        "shrinkage": SHRINKAGE, "score_gains": GAINS, "minimum_score_gaps": MINIMUM_GAPS,
        "role_weight": "Inverse pair/role support in the macro targets, judgment-length exponent −0.15. Pair writes add mirrored None and Reasoning values and permit changes only between those roles.",
        "selection": "48 case-disjoint validation judgments. Require higher macro-F1 and no higher pair counts on all and no-exact-key rows, with accuracy drop at most 0.25 points on both. Rank by mean of the two F1 scores, then pair swaps, then full accuracy and unchanged rows. Include the zero write. For diagnostics retain the best nonzero configuration in any rejected family.",
        "confirmation": "Score the frozen candidates on the prior 72 and 96 case-disjoint validation judgments. Whole-judgment paired bootstrap. No-key rows are those without a matching train context at widths 1, 3, 5 and 9. Historical validation exposure. The frozen base document head fitted these official training cases too.",
        "controls": "Zero output table and a residual shuffle within each correctable prediction/rival orientation. Preserve residual weight/target pairs and match output table Frobenius norm.",
        "flip_calibration": "Four train-only case-held-out leaf fits with fixed covariate cuts and selected deployment rule. Use held-out score-gap tertiles and smooth fix/break frequencies. Assess same bins on the validation cohorts. Calibration is descriptive and does not drive selection or routing.",
        "accuracy_tolerance": ACCURACY_FLOOR, "seed": SEED, "device": "CPU", "official_test_read": False,
    }
    path = HERE / "protocol.json"
    if path.exists():
        assert read(path) == json.loads(json.dumps(spec))
    else:
        write(path, spec)
    return spec


def bank(archive, group, spec):
    assert group in GROUPS, "Experiment scope ends at validation"
    b = evaluation.load_bank(archive, group)
    assert b.signature == spec["score_banks"][group] and len(b.keys) == spec["cohorts"][group]
    if group != "fit":
        assert evaluation.digest(CONT / "scale" / (group + "-mask.npz")) == spec["masks"][group]
    return b


def verify_training_cases(b):
    path = ROOT / "references/official/train.csv"
    expected = read(CONT / "context-memory/protocol.json")["groups"]["fit"]
    np.testing.assert_array_equal(b.keys, expected)
    records = defaultdict(list)
    want = set(b.keys.tolist())
    name_to_role = {name: i for i, name in enumerate(evaluation.ROLES)}
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            key = row["Index"]
            if key in want and row["Text"].strip():
                records[key].append(name_to_role[row["Label"]])
    for key, a, z in zip(b.keys, b.ends[:-1], b.ends[1:]):
        np.testing.assert_array_equal(records[key], b.gold[a:z])


def features(b, temperature):
    return design(b.scores, b.experts, b.old, b.ends, temperature)


def topology(train, spec):
    t, depth, width = (spec[k] for k in ("trees", "depth", "width"))
    rng = np.random.default_rng(spec["seed"])
    center = train.mean(0, dtype=np.float64).astype(np.float32)
    scale = (1. / np.maximum(train.std(0, dtype=np.float64), .025)).astype(np.float32)
    x = np.clip((train - center) * scale, -5., 5.)
    variable = np.flatnonzero(x.std(0, dtype=np.float64) > .02)
    selected = np.stack([rng.choice(variable, width, replace=False) for _ in range(t * depth)]).reshape(t, depth, width)
    direction = rng.normal(size=(t, depth, width)).astype(np.float32)
    direction /= np.linalg.norm(direction, axis=2, keepdims=True).clip(min=1e-9)
    fraction = rng.choice(QUANTILES, size=(t, depth))
    thresholds = []
    for columns, weights, q in zip(selected, direction, fraction):
        p = projected(x, columns, weights)
        thresholds.append([np.quantile(p[:, k], share) for k, share in enumerate(q)])
    return {"center": center, "scale": scale, "columns": selected.astype(np.int32), "direction": direction,
            "threshold": np.asarray(thresholds, dtype=np.float32), "uses_rival": np.array([int(spec["with_rival"])], np.int8)}


def make_targets(b, temperature, method):
    p = row_probabilities(np.asarray(b.scores / temperature, dtype=np.float32))
    gold = b.gold
    counts = np.bincount(gold, minlength=ROLE_COUNT).clip(min=1)
    denominators = counts + np.bincount(b.base, minlength=ROLE_COUNT)
    rarity = 0. if method == "posterior" else .25
    lengths = np.repeat(np.diff(b.ends), np.diff(b.ends))
    w = (counts[gold] / counts.mean()) ** -rarity * (lengths / np.median(lengths)) ** -.15
    if method == "pair":
        pair_p = p[:, [NONE, REASONING]]
        p_diff = (pair_p[:, 1] - pair_p[:, 0]) / pair_p.sum(1).clip(min=1e-8)
        pair_rows = is_pair(b.base) & is_pair(gold)
        signed = np.where(pair_rows, (2 * (gold == REASONING) - 1) - p_diff, 0.)
        target = .5 * signed[:, None] * (EYE[REASONING] - EYE[NONE])[None]
        w *= np.where(pair_rows & (b.base != gold), 1.6, 1.)
    else:
        target = EYE[gold] - p
        if method == "macro":
            per_role = (np.median(denominators) / denominators).clip(.32, 3.0)
            target = target * per_role[None]
            target -= target.mean(1, keepdims=True)
    return target.astype(np.float32), (w / w.mean()).astype(np.float32)


def leaf_moments(indices, labels, values, mass, leaves):
    trees, n = indices.shape
    cases = int(labels.max() + 1)
    assert len(values) == len(mass) == len(labels) == n
    sum_w = np.zeros((trees, leaves), dtype=np.float64)
    sums = np.zeros((trees, leaves, ROLE_COUNT), dtype=np.float64)
    supports = np.zeros((trees, leaves), dtype=np.int64)
    voted = values * mass[:, None]
    for j, at in enumerate(indices):
        sum_w[j] = np.bincount(at, weights=mass, minlength=leaves)
        for label in range(ROLE_COUNT):
            sums[j, :, label] = np.bincount(at, weights=voted[:, label], minlength=leaves)
        once = np.unique(at.astype(np.int64) * cases + labels) // cases
        supports[j] = np.bincount(once, minlength=leaves)
    return sum_w, sums, supports


def fit_values(moments, shrinkage):
    weight, linked, support = moments
    adjustment = linked / (weight + shrinkage)[..., None]
    return np.where((support >= 3)[..., None], adjustment, 0.).astype(np.float32)


def size_of(params):
    return (ROLE_COUNT ** (1 + int(params["uses_rival"][0]))) * (1 << params["threshold"].shape[1])


def shuffle_within(winner, rival):
    strata = winner * ROLE_COUNT + rival
    ordered = np.arange(len(strata))
    rng = np.random.default_rng(SEED + 80)
    for k in np.unique(strata):
        selected = np.flatnonzero(strata == k)
        ordered[selected] = rng.permutation(selected)
    np.testing.assert_array_equal(strata, strata[ordered])
    return ordered


def summarize(scores, old, report):
    full, novel = (report[key] for key in ("all", "novel"))
    return (.5 * (full["macro_f1"] + novel["macro_f1"]), -full["pair_errors"] - novel["pair_errors"],
            full["accuracy"], -int(np.count_nonzero(scores != old)))


def permitted(b, estimate):
    return all(estimate[k]["macro_f1"] > b[k]["macro_f1"] and estimate[k]["pair_errors"] <= b[k]["pair_errors"]
               and estimate[k]["accuracy"] >= b[k]["accuracy"] - ACCURACY_FLOOR for k in ("all", "novel"))


def corrections(truth, base, selected, keep):
    changed = keep & (base != selected)
    fixed = int(np.count_nonzero(changed & (selected == truth)))
    broken = int(np.count_nonzero(changed & (base == truth)))
    count = int(changed.sum())
    return {"rows_changed": count, "fixed": fixed, "broken": broken, "both_wrong": count - fixed - broken,
            "correction_precision": fixed / count if count else None,
            "precision_among_accuracy_changes": fixed / (fixed + broken) if (fixed + broken) else None}


def calibrate(train, winner, indices, method, count, temperature, opt):
    # Case crossfit keeps each training judgment's response labels out of its leaf-value write.
    folds = {key: int.from_bytes(sha256(f"{SEED}|{key}".encode()).digest()[:4], "big") % 4 for key in train.keys}
    row_case = np.repeat(np.arange(len(train.keys)), np.diff(train.ends))
    case_fold = np.asarray([folds[key] for key in train.keys])
    lengths = np.diff(train.ends)
    fold = np.repeat(case_fold, lengths)
    oof = winner.astype(np.int8).copy()
    gaps = np.zeros(len(winner), dtype=np.float32)
    config = (temperature, opt["gain"], opt["minimum_gap"], opt["pair_only"])
    for j in range(4):
        held = fold == j
        assert held.any() and (~held).any()
        keep = ~held
        retained = SimpleNamespace(scores=train.scores[keep], gold=train.gold[keep], base=winner[keep],
                                   ends=np.r_[0, np.cumsum(lengths[case_fold != j])])
        target, mass = make_targets(retained, temperature, method)
        moments = leaf_moments(indices[:, keep], row_case[keep], target, mass, count)
        delta = readout(indices[:, held], fit_values(moments, opt["shrinkage"]))
        oof[held], gaps[held] = decide(train.scores[held], delta, winner[held], *config)
    changed = oof != winner
    n = int(changed.sum())
    summary = corrections(train.gold, winner, oof, np.ones(len(oof), bool))
    if n == 0:
        return {"train_folds": folds, "summary": summary, "upper_cuts": [], "bins": []}
    cuts = sorted(set(np.quantile(gaps[changed], (1 / 3, 2 / 3)).tolist()))
    bins = np.searchsorted(cuts, gaps[changed], side="right")
    fixed = (oof[changed] == train.gold[changed])
    broken = (winner[changed] == train.gold[changed])
    base_fix, base_break = float(fixed.mean()), float(broken.mean())
    out = []
    for i in range(len(cuts) + 1):
        here = bins == i
        m = int(here.sum())
        out.append({"count": m, "expected_fix": (float(fixed[here].sum()) + 12 * base_fix) / (m + 12),
                    "expected_break": (float(broken[here].sum()) + 12 * base_break) / (m + 12)})
    return {"train_folds": folds, "summary": summary, "upper_cuts": cuts, "bins": out}


def assess_calibration(info, gold, base, out, gaps, subset):
    keep = subset & (base != out)
    n = int(keep.sum())
    if not info["bins"] or n == 0:
        return {"count": n, "bins": []}
    group = np.searchsorted(info["upper_cuts"], gaps[keep], side="right")
    fixed = out[keep] == gold[keep]
    broken = base[keep] == gold[keep]
    bins = []
    for i, source in enumerate(info["bins"]):
        selected = group == i
        count = int(selected.sum())
        bins.append({"changes": count, "expected_fix": source["expected_fix"], "expected_break": source["expected_break"],
                     "observed_fix": float(fixed[selected].mean()) if count else None,
                     "observed_break": float(broken[selected].mean()) if count else None})
    predicted = np.asarray([p["expected_fix"] for p in info["bins"]])[group]
    return {"count": n, "bins": bins, "brier_fix": float(np.mean((predicted - fixed) ** 2)),
            "fixed_expected": float(predicted.sum()), "fixed_observed": int(fixed.sum())}


def fit(archive, spec):
    train, select = (bank(archive, g, spec) for g in ("fit", "select"))
    verify_training_cases(train)
    assert not set(train.keys) & set(select.keys)
    temperature = evaluation.baseline_scale(train)
    tr, winner, rival = features(train, temperature)
    dev, base, runner = features(select, temperature)
    row_case = np.repeat(np.arange(len(train.keys)), np.diff(train.ends))
    sham_order = shuffle_within(winner, rival)
    trees = {}
    for name, cfg in TOPOLOGIES.items():
        state = topology(tr, cfg)
        trees[name] = state, tree_indices(tr, winner, rival, state), tree_indices(dev, base, runner, state)
    baseline = evaluation.stats(select, base)
    advances = {}
    frozen = {}
    for method, (layout, formula) in ARMS.items():
        state, ti, di = trees[layout]
        count = size_of(state)
        tgt, mass = make_targets(train, temperature, formula)
        actual = leaf_moments(ti, row_case, tgt, mass, count)
        accepted, observed = None, None
        grid = []
        for shrinkage in SHRINKAGE:
            values = fit_values(actual, shrinkage)
            delta = readout(di, values)
            for gain, gap in product(GAINS, MINIMUM_GAPS):
                out, _ = decide(select.scores, delta, base, temperature, gain, gap, formula == "pair")
                if np.array_equal(out, base):
                    continue
                stats = evaluation.stats(select, out)
                grade = summarize(out, base, stats)
                allowed = permitted(baseline, stats)
                choice = (grade, shrinkage, gain, gap, stats)
                grid.append({"shrinkage": shrinkage, "gain": gain, "minimum_gap": gap, "eligible": allowed,
                             "rank": grade, "full_f1": stats["all"]["macro_f1"], "novel_f1": stats["novel"]["macro_f1"]})
                if allowed and (accepted is None or grade > accepted[0]):
                    accepted = choice
                if observed is None or grade > observed[0]:
                    observed = choice
        pick = accepted or observed
        if pick is None:
            frozen[method] = {"no_changes": True, "grid": grid}
            continue
        grade, shrinkage, gain, gap, select_stats = pick
        prediction_values = fit_values(actual, shrinkage)
        sham = fit_values(leaf_moments(ti, row_case, tgt[sham_order], mass[sham_order], count), shrinkage)
        norm = float(np.linalg.norm(prediction_values.astype(np.float64)))
        sham *= norm / max(float(np.linalg.norm(sham.astype(np.float64))), 1e-12)
        zero = np.zeros_like(prediction_values)
        files = {}
        for suffix, values in (("edited", prediction_values), ("shuffled", sham), ("zero", zero)):
            path = HERE / (method + "-" + suffix + ".safetensors")
            files[suffix] = {"file": path.name, "sha256": save_tensors(path, {**state, "leaf.weight": values}),
                             "value_sha256": sha256(values.tobytes()).hexdigest(),
                             "frobenius": float(np.linalg.norm(values.astype(np.float64))),
                             "nonzero_entries": int(np.count_nonzero(values))}
        assert np.isclose(files["edited"]["frobenius"], files["shuffled"]["frobenius"], rtol=2e-7)
        settings = {"shrinkage": shrinkage, "gain": gain, "minimum_gap": gap, "pair_only": formula == "pair"}
        precision = calibrate(train, winner, ti, formula, count, temperature, settings)
        frozen[method] = {"selection_eligible": accepted is not None, "rank": grade, "selector_scores": select_stats,
                          "params": settings, "layout": layout, "target": formula, "files": files,
                          "calibration": precision, "grid": grid,
                          "leaf_support": {"leaves": int(np.size(actual[2])), "at_least_three_cases": int((actual[2] >= 3).sum())}}
        if accepted is not None:
            advances[method] = grade
        print("selector", method, accepted is not None, grade, settings, flush=True)
    choice = max(advances, key=advances.__getitem__) if advances else "zero"
    write(HERE / "freeze.json", {"protocol_sha256": evaluation.digest(HERE / "protocol.json"), "selected": choice,
                                 "fit_cases": train.keys.tolist(), "selection_cases": select.keys.tolist(),
                                 "fit_rows": len(train.gold), "temperature": temperature, "selector_baseline": baseline,
                                 "arms": frozen, "official_test_read": False})
    print("frozen", choice, flush=True)


def measure(b, before, out, gaps, arm=None):
    data = evaluation.stats(b, out)
    for name, mask in (("all", np.ones(len(out), bool)), ("novel", b.novel)):
        d = data[name]
        d["deltas_95"] = evaluation.interval(b, before, out, mask)
        d["switch_outcomes"] = corrections(b.gold, before, out, mask)
        if arm is not None:
            d["flip_calibration"] = assess_calibration(arm["calibration"], b.gold, before, out, gaps, mask)
    return data


def score(archive, spec):
    freeze_path = HERE / "freeze.json"
    freeze = read(freeze_path)
    assert freeze["protocol_sha256"] == evaluation.digest(HERE / "protocol.json")
    used = set(freeze["fit_cases"] + freeze["selection_cases"])
    groups = {}
    for group in GROUPS[1:]:
        b = bank(archive, group, spec)
        if group != "select":
            assert not set(b.keys) & used
            used.update(b.keys)
        x, base, rival = features(b, freeze["temperature"])
        predictions = {"zero": base}
        results = {"zero": measure(b, base, base, np.zeros(len(base), dtype=np.float32))}
        for method, entry in freeze["arms"].items():
            if entry.get("no_changes"):
                continue
            opts = entry["params"]
            shared = None
            for suffix, record in entry["files"].items():
                file = HERE / record["file"]
                assert evaluation.digest(file) == record["sha256"]
                p = load_file(file)
                assert sha256(p["leaf.weight"].tobytes()).hexdigest() == record["value_sha256"]
                if shared is None:
                    shared = tree_indices(x, base, rival, p)
                final, gap = decide(b.scores, readout(shared, p["leaf.weight"]), base, freeze["temperature"], opts["gain"],
                                    opts["minimum_gap"], opts["pair_only"])
                if suffix == "zero":
                    np.testing.assert_array_equal(final, base)
                    continue
                name = method if suffix == "edited" else method + "-shuffled"
                results[name] = measure(b, base, final, gap, entry if suffix == "edited" else None)
                if suffix == "edited" and group == "select":
                    for subset in ("all", "novel"):
                        assert results[name][subset]["confusion"] == entry["selector_scores"][subset]["confusion"]
                predictions[name] = final
            for subset, keep in (("all", np.ones(len(base), bool)), ("novel", b.novel)):
                results[method][subset]["edited_vs_shuffled_95"] = evaluation.interval(b, predictions[method + "-shuffled"], predictions[method], keep)
        groups[group] = {"cases": len(b.keys), "rows": len(b.gold), "no_exact_key": int(b.novel.sum()), "results": results}
        path = HERE / (group + "-predictions.npz")
        np.savez_compressed(path, case_ids=b.keys, offsets=b.ends, gold=b.gold, novel=b.novel, **predictions)
        print(group, {k: (round(v["all"]["macro_f1"] * 100, 4), round(v["novel"]["macro_f1"] * 100, 4), v["all"]["pair_errors"]) for k, v in results.items()}, flush=True)
    write(HERE / "results.json", {"freeze_sha256": evaluation.digest(freeze_path), "selected": freeze["selected"],
                                  "groups": groups, "official_test_read": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("protocol", "fit", "score"))
    parser.add_argument("--bank", type=Path, default=CONT / "legal-context/calibration-banks.zip")
    args = parser.parse_args()
    spec = protocol(args.bank)
    if args.action == "fit":
        fit(args.bank, spec)
    elif args.action == "score":
        score(args.bank, spec)
