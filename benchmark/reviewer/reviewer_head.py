"""Program the deployed InLegalBERT review readout with closed-form weight edits."""

import argparse
from dataclasses import dataclass
import hashlib
import io
from itertools import product
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from safetensors.numpy import load_file, save_file
from scipy.linalg import solve
from scipy.special import softmax


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / "live-nov22/bert-t4"
ROUND2 = ROOT / "live-nov22/reviewer-round2"
POOLS = HERE.parent / "t4-return"
PLAN = ROUND2 / "round2-plan.json"
ROLES = ("None", "Facts", "Issue", "Petitioner argument", "Respondent argument", "Reasoning", "Decision")
EYE = np.eye(len(ROLES))
NONE, REASONING = 0, 5
DIRECTION = EYE[REASONING] - EYE[NONE]
SEED = 460198
REGULARIZATION = (.02, .2, 2.)
GAINS = (.15, .35, .70, 1.)
FOCUS = (0., .4)
ACCURACY_TOLERANCE = .001


@dataclass
class Bank:
    keys: np.ndarray
    offsets: np.ndarray
    review_offsets: np.ndarray
    positions: np.ndarray
    gold: np.ndarray
    parent: np.ndarray
    original: np.ndarray
    prior: np.ndarray
    aux: np.ndarray
    pool: np.ndarray
    reference: np.ndarray
    reference_scores: np.ndarray


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    temporary = path.with_suffix(".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_bank(group, plan, receipts=None):
    shards = ROUND2 / "new-cases" if group == "fresh" else OLD / "cases"
    keys = plan["groups"][group]
    source = POOLS / (group + ".npz")
    if receipts is not None:
        hardware = receipts["device"]["devices"]
        assert hardware and all("Tesla T4" in d for d in hardware)
        assert receipts["model_sha256"] == plan["donor_sha256"]
        assert sha(source) == receipts["groups"][group]["file_sha256"]
    columns = ("gold", "base", "positions", "original", "prior", "aux")
    rows = []
    for key in keys:
        with np.load(shards / (key + ".npz"), allow_pickle=False) as case:
            assert case["case_sha256"].item() == plan["case_text_sha256"][key]
            rows.append({column: case[column] for column in columns})
    offsets = np.r_[0, np.cumsum([len(row["gold"]) for row in rows])]
    review_offsets = np.r_[0, np.cumsum([len(row["positions"]) for row in rows])]
    with np.load(source, allow_pickle=False) as x:
        np.testing.assert_array_equal(x["case_ids"], keys)
        np.testing.assert_array_equal(x["offsets"], offsets)
        np.testing.assert_array_equal(x["review_offsets"], review_offsets)
        pool = x["pool"]
        reference = x["reference"]
        reference_scores = x["scores"]
    assert pool.shape == (review_offsets[-1], 768) and pool.dtype == np.float32 and np.isfinite(pool).all()
    assert reference.shape == (offsets[-1],)
    assert reference_scores.shape == (review_offsets[-1], 7) and np.isfinite(reference_scores).all()

    def join(column):
        return np.concatenate([part[column] for part in rows])

    positions = np.concatenate([part["positions"] + start for part, start in zip(rows, offsets)])
    assert len(np.unique(positions)) == len(positions)
    return Bank(np.asarray(keys), offsets, review_offsets, positions, join("gold"), join("base"),
                join("original"), join("prior"), join("aux"), pool, reference, reference_scores)


def check_archived(group, bank):
    if group == "fresh":
        reference = ROUND2 / "none-reasoning/document-returned/predictions.npz"
        with np.load(reference, allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["case_ids"], bank.keys)
            np.testing.assert_array_equal(archive["offsets"], bank.offsets)
            np.testing.assert_array_equal(archive["gold"], bank.gold)
            np.testing.assert_array_equal(archive["review"], bank.positions)
            labels = archive["donor-baseline"]
    elif group == "confirm":
        reference = OLD / "full-results.zip"
        with ZipFile(reference) as f, np.load(io.BytesIO(f.read("confirm-predictions.npz")), allow_pickle=False) as a:
            np.testing.assert_array_equal(a["case_ids"], bank.keys)
            np.testing.assert_array_equal(a["offsets"], bank.offsets)
            np.testing.assert_array_equal(a["gold"], bank.gold)
            labels = a["current_actual"]
    else:
        return None
    np.testing.assert_array_equal(labels, bank.reference)
    return {"path": str(reference.relative_to(ROOT)), "sha256": sha(reference), "label_differences": 0}


def standard_design(bank, params, donor):
    author = softmax(bank.pool @ donor["weight"].T + donor["bias"], axis=1)
    x = np.column_stack((bank.aux, author, author * bank.aux[:, 40:47], bank.pool))
    return (x - params["center"]) * params["scale"]


def normalized(raw):
    v = np.maximum(raw, 0.)
    return v / np.maximum(v.sum(1, keepdims=True), 1e-8)


def decisions(raw, parent, original, prior, rule):
    gain, gap, veto = rule
    q = normalized(raw)
    if gain != 1.:
        q = (1. - gain) * (.5 * prior + .5 * EYE[parent]) + gain * q
    new = q.argmax(1)
    row = np.arange(len(new))
    take = q[row, new] - q[row, parent] >= gap
    if veto:
        take &= new != original
    return np.where(take, new, parent)


def apply(bank, raw, rule, guard):
    answer = bank.parent.copy()
    y = decisions(raw, bank.parent[bank.positions], bank.original, bank.prior, rule)
    if guard:
        incumbent = bank.reference[bank.positions]
        y = np.where(is_pair(y) & is_pair(incumbent), y, incumbent)
    answer[bank.positions] = y
    return answer


def is_pair(labels):
    return (labels == NONE) | (labels == REASONING)


def fit_ridge(z, w, target, ridge):
    weighted = z * w[:, None]
    gram = z.T @ weighted
    ridge_scale = np.mean(np.diagonal(gram)[:-1]) if len(gram) > 1 else np.mean(np.diagonal(gram))
    q = np.arange(len(gram))
    gram[q, q] += (ridge * ridge_scale) * np.r_[np.ones(len(gram) - 1), .05]
    return solve(gram, weighted.T @ target, assume_a="pos", check_finite=False)


def role_weights(gold, focus):
    count = np.maximum(np.bincount(gold, minlength=len(ROLES)), 1)
    rare = (count[gold] / len(gold)) ** -focus
    return rare / rare.mean()


def projections(bank, raw, gold, rule):
    """Find signed target margin changes that the existing decision rule can use."""
    current = bank.reference[bank.positions]
    parent = bank.parent[bank.positions]
    sign = np.where(gold == REASONING, 1., -1.)
    magnitude = np.maximum(np.maximum(raw, 0.).sum(1), .25)
    low, high = np.zeros(len(raw)), 2.5 * magnitude
    direction = sign[:, None] * DIRECTION

    def prediction(step):
        return decisions(raw + step[:, None] * direction, parent, bank.original, bank.prior, rule)

    eligible = is_pair(gold) & is_pair(current) & (current != gold) & (prediction(high) == gold)
    for _ in range(13):
        cut = .5 * (low + high)
        okay = prediction(cut) == gold
        low, high = np.where(okay, low, cut), np.where(okay, cut, high)
    return np.where(eligible, sign * (high + .04 * magnitude), 0.), eligible


def target_for(arm, bank, raw, gold, rule):
    current = bank.reference[bank.positions]
    magnitude = np.maximum(np.maximum(raw, 0.).sum(1), .25)
    pair = is_pair(gold)
    signs = np.where(gold == REASONING, 1., -1.)
    if arm == "pair-boundary":
        target, acted = projections(bank, raw, gold, rule)
        importance = np.where(acted, 1., np.where(pair, .45, 1.))
        return target[:, None], importance, int(acted.sum())
    if arm.startswith("pair-"):
        anchor = .5 * (raw[:, REASONING] - raw[:, NONE])
        target = .5 * signs * magnitude - anchor
        target = np.where(pair, target, 0.)
        importance = np.where(current == gold, .40, 1.)
        return target[:, None], importance, int(pair.sum())
    target = EYE[gold] * magnitude[:, None] - raw
    target -= target.mean(1, keepdims=True)
    importance = np.where(current == gold, .40, 1.)
    return target, importance, len(gold)


def weights_for(arm, train, z, raw, rule, focus, ridge, scrambled=False):
    rng = np.random.default_rng(SEED)
    gold = train.gold[train.positions].astype(np.int64)
    if scrambled:
        gold = gold.copy()
        eligible = is_pair(gold) if arm.startswith("pair-") else np.ones(len(gold), dtype=bool)
        gold[eligible] = rng.permutation(gold[eligible])
    target, importance, support = target_for(arm, train, raw, gold, rule)
    if arm == "pair-bias":
        z = z[:, -1:]
    elif arm == "pair-auxiliary":
        z = np.column_stack((z[:, :82], z[:, -1]))
    coeff = fit_ridge(z, importance * role_weights(gold, focus), target, ridge)
    if arm == "pair-bias":
        coeff = np.vstack((np.zeros((850, 1)), coeff))
    elif arm == "pair-auxiliary":
        coeff = np.vstack((coeff[:-1], np.zeros((768, 1)), coeff[-1:]))
    if arm.startswith("pair-"):
        coeff = coeff @ DIRECTION[None]
    return coeff, support


def confusions(bank, answer):
    y = bank.gold.astype(np.int64) * len(ROLES) + answer
    return np.stack([np.bincount(y[a:b], minlength=49).reshape(7, 7) for a, b in zip(bank.offsets[:-1], bank.offsets[1:])])


def scores(conf):
    hits = np.diagonal(conf, axis1=-2, axis2=-1)
    f1 = 2 * hits / np.maximum(conf.sum(-1) + conf.sum(-2), 1)
    pair = conf[..., NONE, REASONING] + conf[..., REASONING, NONE]
    return f1.mean(-1), hits.sum(-1) / conf.sum((-2, -1)), pair, f1


def summary(bank, answer):
    matrix = confusions(bank, answer).sum(0)
    f1, acc, pair, by_role = scores(matrix)
    change = answer != bank.reference
    return {"macro_f1": float(f1), "accuracy": float(acc),
            "none_reasoning_errors": int(pair), "none_as_reasoning": int(matrix[NONE, REASONING]),
            "reasoning_as_none": int(matrix[REASONING, NONE]),
            "changed": int(np.count_nonzero(change)),
            "corrected": int(np.count_nonzero(change & (answer == bank.gold))),
            "harmed": int(np.count_nonzero(change & (bank.reference == bank.gold))),
            "per_role_f1": dict(zip(ROLES, by_role.tolist())), "confusion": matrix.tolist()}


def paired(bank, new, old):
    now, before = confusions(bank, new), confusions(bank, old)
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(now), size=(5000, len(now)))
    def sampled(case_matrices):
        totals = np.concatenate([case_matrices[chunk].sum(1) for chunk in np.array_split(indices, 40)])
        return scores(totals)

    f1_new, acc_new, pair_new, _ = sampled(now)
    f1_old, acc_old, pair_old, _ = sampled(before)
    return {"macro_f1": np.quantile(f1_new - f1_old, [.025, .975]).tolist(),
            "accuracy": np.quantile(acc_new - acc_old, [.025, .975]).tolist(),
            "none_reasoning_errors": np.quantile(pair_new - pair_old, [.025, .975]).tolist()}


def patch(base, coeff):
    return {**base, "matrix": base["matrix"] + coeff[:-1], "bias": base["bias"] + coeff[-1]}


def make_search(plan, recipe):
    return {
        "case_plan_sha256": sha(PLAN), "groups": plan["groups"], "base_head_sha256": recipe["existing_head_sha256"],
        "threshold": recipe["rule"], "seed": SEED, "ridges": REGULARIZATION, "gains": GAINS,
        "class_weights": FOCUS, "accuracy_tolerance": ACCURACY_TOLERANCE, "gradient_steps": 0,
        "arms": {
            "pair-boundary": "Two mirrored reviewer output columns. Ridge target is the smallest signed raw change that clears the existing normalized margin and veto for a misclassified pair example.",
            "pair-residual": "Two mirrored columns targeting the signed class margin. Other gold roles use a zero correction target.",
            "pair-bias": "Same objective as pair-residual fitted to the two biases only.",
            "pair-auxiliary": "Same objective as pair-residual fitted to structural and author-score features. Backbone pooled coordinates receive a zero edit.",
            "all-role": "All seven columns receive centered raw residual targets through the same ridge and gain grid.",
            "gap-only": "Unchanged head with candidate review thresholds and original-label veto comparison.",
        },
        "selection": "Maximize seven-role macro-F1 subject to pair errors decreasing and accuracy loss not exceeding 0.10 points. Identity is eligible. Include ungated native predictions and a guard preserving all decisions outside the None-Reasoning pair.",
        "controls": "Permute pair gold labels among routed pair examples, and all gold labels for the all-role arm. Match overall weight/bias edit norm. Reuse the selected gain, ridge and guard of the real edit.",
        "bootstrap": "5,000 paired case-cluster resamples within each cohort.",
        "data": "Official-train cases for solving, 48 validation cases for selection, 72 and 96 disjoint validation cases for replay; these validation cohorts have historical project exposure. No official test data read.",
    }


def choose(train, valid, raw_fit, design_fit, raw_valid, design_valid, base, rule):
    baseline = summary(valid, valid.reference)
    stats = (baseline["macro_f1"], baseline["accuracy"], -baseline["none_reasoning_errors"], 0)
    z = np.column_stack((design_fit, np.ones(len(design_fit))))
    zz = np.column_stack((design_valid, np.ones(len(design_valid))))
    selections, grid = {}, {}
    for arm in ("pair-boundary", "pair-residual", "pair-bias", "pair-auxiliary", "all-role", "gap-only"):
        record, best, nonzero, audit = None, stats, None, []
        if arm == "gap-only":
            trials = [((), np.zeros_like(np.vstack((base["matrix"], base["bias"]))), (rule[0], gap, veto), guard)
                      for gap, veto, guard in product((.04, .09, .16, .23, .32), (False, True), (False, True))]
        else:
            trials = []
            for focus, ridge in product(FOCUS, REGULARIZATION):
                coeff, support = weights_for(arm, train, z, raw_fit, rule, focus, ridge)
                trials.extend(((focus, ridge, gain, support), gain * coeff, rule, guard)
                              for gain, guard in product(GAINS, (False, True)))
        raw_cache = {}
        for args, coeff, gate, guard in trials:
            if args not in raw_cache:
                raw_cache[args] = (raw_valid + zz @ coeff) if args else raw_valid
            answer = apply(valid, raw_cache[args], gate, guard)
            current = summary(valid, answer)
            key = (current["macro_f1"], current["accuracy"], -current["none_reasoning_errors"], -current["changed"])
            audit.append({"settings": args, "rule": gate, "guard": guard, "rank": key})
            acceptable = key[1] >= stats[1] - ACCURACY_TOLERANCE and key[2] > stats[2]
            candidate = (args, coeff, gate, guard, key)
            if current["changed"] and (nonzero is None or key > nonzero[-1]):
                nonzero = candidate
            if acceptable and key > best:
                best, record = key, candidate
        chosen = record if record is not None else nonzero
        if chosen is None:
            selections[arm], grid[arm] = None, audit
            continue
        args, coeff, gate, guard, key = chosen
        path = HERE / (arm + ".safetensors")
        save_file(patch(base, coeff), path)
        restored = load_file(path)
        for name, value in patch(base, coeff).items():
            np.testing.assert_array_equal(value, restored[name])
        if arm.startswith("pair-"):
            others = [i for i in range(len(ROLES)) if i not in (NONE, REASONING)]
            np.testing.assert_array_equal(restored["matrix"][:, others], base["matrix"][:, others])
            np.testing.assert_array_equal(restored["bias"][others], base["bias"][others])
        result = {"selected_over_identity": record is not None, "settings": args, "rule": gate,
                  "guard": guard, "rank": key, "weights_sha256": sha(path),
                  "matrix_change_count": int(np.count_nonzero(restored["matrix"] - base["matrix"])),
                  "bias_change_count": int(np.count_nonzero(restored["bias"] - base["bias"])),
                  "delta_frobenius": float(np.linalg.norm(coeff))}
        if args:
            null, _ = weights_for(arm, train, z, raw_fit, rule, *args[:2], scrambled=True)
            norm = np.linalg.norm(null)
            null = null * (np.linalg.norm(coeff) / max(norm, 1e-12))
            null_path = HERE / (arm + "-shuffled.safetensors")
            save_file(patch(base, null), null_path)
            result["shuffled_weights_sha256"] = sha(null_path)
            result["shuffled_delta_frobenius"] = float(np.linalg.norm(null))
        selections[arm], grid[arm] = result, audit
        print("selected", arm, result["selected_over_identity"], key, args, gate, guard, flush=True)
    return selections, grid


def main(action):
    plan = read_json(PLAN)
    recipe = read_json(OLD / "review.json")
    origin = OLD / "head.safetensors"
    assert sha(origin) == recipe["existing_head_sha256"]
    process = make_search(plan, recipe)
    if action == "protocol":
        write_json(HERE / "protocol.json", process)
        return
    assert read_json(HERE / "protocol.json") == json.loads(json.dumps(process))
    groups = ["fit", "select", "confirm", "fresh"]
    all_ids = sum([plan["groups"][group] for group in groups], [])
    assert len(set(all_ids)) == len(all_ids)
    assert len(set(plan["case_text_sha256"][key] for key in all_ids)) == len(all_ids)
    with np.load(OLD / "donor-classifier.npz", allow_pickle=False) as f:
        donor = {name: f[name] for name in f.files}
    base = load_file(origin)
    receipts = read_json(POOLS / "pool-receipt.json")
    rule = recipe["rule"]

    def load(group):
        bank = read_bank(group, plan, receipts)
        archived = check_archived(group, bank)
        design = standard_design(bank, base, donor)
        raw = design @ base["matrix"] + base["bias"]
        np.testing.assert_allclose(normalized(raw), bank.reference_scores, atol=2e-5, rtol=0.)
        np.testing.assert_array_equal(apply(bank, raw, rule, False), bank.reference)
        print(group, "baseline parity", len(bank.keys), len(bank.gold), len(bank.positions), flush=True)
        return bank, design, raw, archived

    freeze_path = HERE / "freeze.json"
    if action == "fit":
        train, design_fit, raw_fit, _ = load("fit")
        valid, design_valid, raw_valid, _ = load("select")
        selections, grids = choose(train, valid, raw_fit, design_fit, raw_valid, design_valid, base, rule)
        write_json(freeze_path, {"protocol_sha256": sha(HERE / "protocol.json"), "pool_receipt_sha256": sha(POOLS / "pool-receipt.json"),
                                 "selections": selections, "selector_grids": grids})
        return
    freeze = read_json(freeze_path)
    assert freeze["protocol_sha256"] == sha(HERE / "protocol.json")
    assert freeze["pool_receipt_sha256"] == sha(POOLS / "pool-receipt.json")
    record = {"freeze_sha256": sha(freeze_path), "device": receipts["device"], "groups": {}}
    for group in groups[1:]:
        bank, design, _, archived = load(group)
        results, preds = {"incumbent": summary(bank, bank.reference)}, {}
        for arm, item in freeze["selections"].items():
            if item is None:
                continue
            candidates = [(arm, item["weights_sha256"])]
            if "shuffled_weights_sha256" in item:
                candidates.append((arm + "-shuffled", item["shuffled_weights_sha256"]))
            for name, digest in candidates:
                path = HERE / (name + ".safetensors")
                assert sha(path) == digest
                p = load_file(path)
                raw = design @ p["matrix"] + p["bias"]
                answer = apply(bank, raw, item["rule"], item["guard"])
                result = summary(bank, answer)
                if group == "select" and name == arm:
                    assert list((result["macro_f1"], result["accuracy"], -result["none_reasoning_errors"], -result["changed"])) == item["rank"]
                result["paired_95"] = paired(bank, answer, bank.reference)
                results[name], preds[name] = result, answer
                print(group, name, result["macro_f1"], result["accuracy"], result["none_reasoning_errors"], flush=True)
        for arm in freeze["selections"]:
            if arm in preds and arm + "-shuffled" in preds:
                results[arm]["paired_vs_shuffled_95"] = paired(bank, preds[arm], preds[arm + "-shuffled"])
        record["groups"][group] = {"judgments": len(bank.keys), "sentences": len(bank.gold),
                                    "reviewed": len(bank.positions), "archived_reference": archived,
                                    "methods": results}
        np.savez_compressed(HERE / (group + "-predictions.npz"), case_ids=bank.keys, offsets=bank.offsets,
                            gold=bank.gold, incumbent=bank.reference, review=bank.positions, **preds)
    write_json(HERE / "results.json", record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("protocol", "fit", "evaluate"))
    main(parser.parse_args().action)
