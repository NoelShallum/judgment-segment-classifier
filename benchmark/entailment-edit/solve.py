"""Fit, lock and test direct weight edits to the native NLI classifier."""

import argparse
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from itertools import product
import json
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from safetensors.numpy import load_file, save_file
from scipy.linalg import solve
from scipy.special import softmax

from prepare import ARCHIVE, HERE, baseline, design
from protocol import PAIRS, ROLES, digest

CAPTURES = HERE / "entailment-captures.zip"
PREP = HERE / "preparation.json"
FROZEN = HERE / "freeze.json"
WEIGHTS = HERE / "edited-readouts.safetensors"
TEMPERED_STEP = .6
RIDGES = (.025, .15)
TARGETS = ("source-residual", "target-contrast", "document-residual")
COMPOSITIONS = ("edited", "delta-only")
GATES = ("hard-pair", "top-two", "top-three", "all")
GAINS = (.10, .28, .56, .9, 1.35)
COST = .004
SEED = 29519
ENTAILMENT, CONTRADICTION = 0, 2


@dataclass
class Scored:
    bank: baseline.Bank
    rows: np.ndarray
    hidden: np.ndarray
    source: np.ndarray
    validated_delta: float


def preregister():
    spec = {
        "fit_captures": "192 official-train judgments, fixed routed positions from a separate train-only fit",
        "edit_rows": {"entailment": "+ half of closed-form contrast edit", "neutral": "unchanged", "contradiction": "- half of edit"},
        "targets": dict(zip(TARGETS, (
            "Gold minus inferred role probability from centered source entailment–contradiction margin",
            "Centered full retarget to official gold preference with limited ridge shrinkage",
            "Gold minus probability from jointly interpreting the frozen document and source NLI scores",
        ))),
        "ridges": list(RIDGES), "sources": list(COMPOSITIONS), "gates": list(GATES), "relative_gains": list(GAINS),
        "plain_donor_controls": "Tune unedited donor evidence on the same selector and also compare at candidate gate and gain",
        "shuffled_controls": "Repeat the fit with case-permuted and globally permuted training labels, independently matching the L2 norm of the actual classifier edit. Same gates, composition and scale as selected edit.",
        "selection": "0.5 times the sum of all and no-exact-key 7-role macro-F1, then accuracy and pair controls; retain current system when no edit beats it jointly. Keep a selector-locked diagnostic as a negative control if no candidate wins, using the exact same confidence checks.",
        "guard": "Require no growth in either all and no-key None–Reasoning swap counts and no loss of over 0.4 accuracy points on all rows of selector",
        "official_evaluation": "Out of scope absent positive paired confirmation on both complete-case cohorts. No test file access in this script.",
        "seed": SEED, "gpu_training_steps": 0, "cpu_iterative_training_steps": 0,
    }
    path = HERE / "fit-plan.json"
    if path.exists():
        assert baseline.read(path) == spec
    else:
        baseline.write(path, spec)


def verified(z, name, signature):
    content = z.read(name)
    if sha256(content).hexdigest() != signature:
        raise ValueError(f"Capture member {name} does not match its signature")
    return content


def capture_metadata(z):
    info = json.loads(z.read("capture-summary.json"))
    with ZipFile(HERE / "entailment-input.zip") as original:
        assert verified(z, "input-manifest.json", info["input_sha256"]) == original.read("input-manifest.json")
    w = np.load(BytesIO(verified(z, "native-head.npy", info["native_head_sha256"])), allow_pickle=False)
    donor = baseline.read(HERE / "donor.json")
    assert info["donor_revision"] == donor["revision"] and info["donor_sha256"] == donor["remote_weights_sha256"]
    assert "T4" in info["gpu"].split(",")[0] and info["dtype_backbone"] == "float16" and info["dtype_classifier"] == "float32"
    assert w.shape == (3, 768) and w.dtype == np.float32 and np.isfinite(w).all()
    assert info["effective_classifier_bias"] is None or info["effective_classifier_bias"] == info["checkpoint_classifier_bias"]
    return info, w


def cohort(z, group, info, w):
    declared = info["captures"][group]
    assert info["input_sha256"] == declared["input_sha256"] and declared["native_linear_max_diff"] <= 5e-5
    receipt = baseline.read(PREP)
    route_path = HERE / f"{group}-route.npy"
    assert digest(route_path) == receipt["cohorts"][group]["route_sha256"]
    data = BytesIO(verified(z, f"capture-{group}.npz", declared["output_sha256"]))
    with np.load(data, allow_pickle=False) as package:
        rows = package["rows"]
        hidden_raw = package["hidden"]
        logits = package["logits"]
    np.testing.assert_array_equal(rows, np.load(route_path, allow_pickle=False))
    assert hidden_raw.dtype == np.float16 and hidden_raw.shape == (len(rows), len(ROLES), w.shape[-1])
    assert logits.dtype == np.float32 and logits.shape == hidden_raw.shape[:2] + (3,)
    hidden_float = hidden_raw.astype(np.float32)
    expected = hidden_float @ w.T
    if info["effective_classifier_bias"] is not None:
        expected += np.asarray(info["effective_classifier_bias"], dtype=np.float32)
    d = float(np.abs(expected - logits).max())
    assert d <= .00008 and np.isfinite(expected).all()
    bank = baseline.load_bank(ARCHIVE, group)
    assert bank.signature == receipt["input_archive_hashes_by_group"][group]
    centered = hidden_float - hidden_float.mean(1, keepdims=True)
    raw = logits[:, :, ENTAILMENT] - logits[:, :, CONTRADICTION]
    return Scored(bank, rows, centered, raw - raw.mean(1, keepdims=True), d)


def candidates(scored, temperature):
    bank = scored.bank
    _, pair_mask, _ = design(bank, temperature)
    row = scored.rows
    top = np.argsort(-bank.scores[row], axis=1, kind="stable")
    pair_gate = baseline.EYE[bank.base[row]].astype(bool)
    for j, (a, b) in enumerate(PAIRS):
        chosen = pair_mask[row, j]
        pair_gate[chosen, a] = True
        pair_gate[chosen, b] = True
    return {
        "hard-pair": pair_gate,
        "top-two": baseline.EYE[top[:, :2]].any(1),
        "top-three": baseline.EYE[top[:, :3]].any(1),
        "all": np.ones((len(row), len(ROLES)), dtype=bool),
    }


def source_scale(train):
    return float(max(np.sqrt(np.mean(train.source ** 2)), .25))


def half_rows(train, y, native_scale, doc_scale):
    gold = baseline.EYE[y]
    doc_raw = train.bank.scores[train.rows] / doc_scale
    native_logit = train.source / native_scale
    return {
        "source-residual": native_scale * (gold - softmax(native_logit, axis=1)),
        "target-contrast": 1.8 * native_scale * (gold - 1 / 7.) - train.source,
        "document-residual": native_scale * (gold - softmax(doc_raw + TEMPERED_STEP * native_logit, axis=1)) / TEMPERED_STEP,
    }


def sufficient(train, y, native_scale, doc_scale):
    rows = train.rows
    bank = train.bank
    memberships = np.searchsorted(bank.ends[1:], rows, side="right")
    row_counts = np.bincount(memberships, minlength=len(bank.keys))
    prior = np.maximum(np.bincount(y, minlength=7) / len(y), .001)
    weight = prior[y] ** -.34 / np.sqrt(row_counts[memberships])
    weight /= np.mean(weight)
    fitting = (train.hidden.astype(np.float64) * np.sqrt(weight)[:, None, None]).reshape(-1, 768)
    targets = half_rows(train, y, native_scale, doc_scale)
    rhs = fitting.T @ np.stack([(v * np.sqrt(weight)[:, None]).reshape(-1) for v in targets.values()], axis=1)
    gram = fitting.T @ fitting
    return gram, rhs, memberships


def regression(gram, rhs, ridge):
    diag = np.trace(gram) / len(gram)
    adjusted = gram.copy()
    adjusted.flat[::len(gram) + 1] += ridge * max(diag, 1e-6)
    return solve(adjusted, rhs, assume_a="pos").astype(np.float32)


def predict(cohort, available, value, delta, temp, scale):
    if value is None:
        return cohort.bank.base.copy()
    if value["composition"] == "edited":
        evidence = cohort.source + delta
    elif value["composition"] == "delta-only":
        evidence = delta
    else:
        raise ValueError(value["composition"])
    merged = cohort.bank.scores[cohort.rows] + value["gain"] * temp * np.clip(evidence / scale, -6., 6.)
    prediction = cohort.bank.base.copy()
    prediction[cohort.rows] = np.where(available[value["scope"]], merged, -np.inf).argmax(1)
    return prediction


def working_score(m):
    return {part: {field: m[part][field] for field in ("macro_f1", "accuracy", "pair_errors")} for part in ("all", "novel")}


def candidates_for_selection(select, possible, delta, descriptor, temperature, scale, inherited, trials):
    best, best_r = None, baseline.choice_rank(inherited)
    largest, largest_r = None, None
    compositions = COMPOSITIONS if descriptor.get("mode") != "source" else ("edited",)
    for comp, scope, gain in product(compositions, GATES, GAINS):
        descriptor2 = dict(descriptor, composition=comp, scope=scope, gain=gain)
        m = baseline.stats(select.bank, predict(select, possible, descriptor2, delta, temperature, scale))
        ranking = baseline.choice_rank(m)
        guarded = baseline.rate_allowed(m, inherited)
        trials.append(dict(descriptor2, guard=guarded, selection_statistics=working_score(m)))
        if guarded and ranking > best_r:
            best, best_r = descriptor2, ranking
        if largest_r is None or ranking > largest_r:
            largest, largest_r = descriptor2, ranking
    return best, best_r, largest, largest_r


def patch(weight, delta):
    transformed = weight.copy()
    transformed[ENTAILMENT] += .5 * delta
    transformed[CONTRADICTION] -= .5 * delta
    np.testing.assert_array_equal(transformed[1], weight[1])
    return transformed


def fit():
    started = perf_counter()
    preregister()
    if FROZEN.exists():
        raise FileExistsError("Use the frozen readout for confirmation")
    with ZipFile(CAPTURES) as z:
        info, w = capture_metadata(z)
        train = cohort(z, "fit", info, w)
        select = cohort(z, "select", info, w)
    temp = baseline.baseline_scale(train.bank)
    scale = source_scale(train)
    available = candidates(select, temp)
    inherited = baseline.stats(select.bank, select.bank.base)
    y = train.bank.gold[train.rows]
    gram, rhs, memberships = sufficient(train, y, scale, temp)
    trials, deltas = [], {}
    best, best_r = None, baseline.choice_rank(inherited)
    diagnostic, diagnostic_r = None, None
    starter, _, largest_source, _ = candidates_for_selection(select, available, 0., {"mode": "source"}, temp, scale, inherited, trials)
    for ridge in RIDGES:
        proposals = regression(gram, rhs, ridge).T
        for mode, delta in zip(TARGETS, proposals):
            deltas[mode, ridge] = delta
            chosen, rank, largest, largest_rank = candidates_for_selection(select, available, select.hidden @ delta,
                                                                           {"mode": mode, "ridge": ridge}, temp, scale, inherited, trials)
            if chosen is not None and rank > best_r:
                best, best_r = chosen, rank
            if diagnostic_r is None or largest_rank > diagnostic_r:
                diagnostic, diagnostic_r = largest, largest_rank
    candidate = best or diagnostic
    mode, ridge = candidate["mode"], candidate["ridge"]
    original_edit = deltas[mode, ridge]
    norm = np.linalg.norm(original_edit)
    control_y = {"global-shuffle": np.random.default_rng(SEED).permutation(y), "case-shuffle": y.copy()}
    rng = np.random.default_rng(SEED)
    for case in np.unique(memberships):
        selected = memberships == case
        control_y["case-shuffle"][selected] = rng.permutation(y[selected])
    physical = {"source": w, "classifier.weight": patch(w, original_edit)}
    fractions = {}
    for name, labels in control_y.items():
        gram, rhs, _ = sufficient(train, labels, scale, temp)
        d = regression(gram, rhs, ridge)[:, TARGETS.index(mode)]
        d *= norm / np.linalg.norm(d)
        physical[name] = patch(w, d)
        fractions[name] = float(np.mean(y == labels))
    norms = {k: float(np.linalg.norm(v - w)) for k, v in physical.items()}
    np.testing.assert_allclose([norms["case-shuffle"], norms["global-shuffle"]], norms["classifier.weight"], rtol=2e-6)
    if info["effective_classifier_bias"] is not None:
        physical["classifier.bias"] = np.asarray(info["effective_classifier_bias"], dtype=np.float32)
    out = WEIGHTS.with_suffix(".part")
    save_file({k: np.ascontiguousarray(v) for k, v in physical.items()}, str(out))
    out.replace(WEIGHTS)
    baseline.write(HERE / "selection-trials.json", trials)
    config = {
        "fit_plan_sha256": digest(HERE / "fit-plan.json"), "preparation_sha256": digest(PREP),
        "capture_sha256": digest(CAPTURES), "weights_sha256": digest(WEIGHTS), "trials_sha256": digest(HERE / "selection-trials.json"),
        "native_scale": scale, "document_scale": temp, "active": best, "evaluated_edit": candidate,
        "source_active": starter, "source_diagnostic": largest_source, "selection_zero": working_score(inherited),
        "weight_edit_l2_norms": norms, "fraction_shuffled_role_unchanged": fractions, "delta_rank": 1,
        "fit_native_logit_parity": train.validated_delta, "select_native_logit_parity": select.validated_delta,
        "fit_standalone_nli": working_score({"all": baseline.score(y, train.source.argmax(1)), "novel": baseline.score(y, train.source.argmax(1))})["all"],
        "capture_device": info["gpu"], "fit_seconds": round(perf_counter() - started, 2),
        "native_preserved_bias": info["effective_classifier_bias"],
        "inspected_official_rows": 0,
    }
    baseline.write(FROZEN, config)
    print(json.dumps(config, indent=2), flush=True)


def exact_delta(capture, weights, source):
    # Neutral stays byte-identical and delta is measured from the serialized edited rows.
    np.testing.assert_array_equal(weights[1], source[1])
    diff = weights - source
    return capture.hidden @ (diff[ENTAILMENT] - diff[CONTRADICTION])


def pair_outcomes(m):
    return {"–".join((ROLES[i], ROLES[j])): [m[i][j], m[j][i]] for i, j in PAIRS}


def outcomes(bank, p, route):
    result = baseline.stats(bank, p)
    for k, measured in result.items():
        measured["pair_directions"] = pair_outcomes(measured["confusion"])
    result["routed"] = baseline.score(bank.gold[route], p[route])
    result["changes"] = int((p != bank.base).sum())
    result["corrected"] = int(((p == bank.gold) & (bank.base != bank.gold)).sum())
    result["damaged"] = int(((p != bank.gold) & (bank.base == bank.gold)).sum())
    return result


def infer(cohort, w, lock):
    temp, scale = lock["document_scale"], lock["native_scale"]
    available = candidates(cohort, temp)
    edit_config = lock["evaluated_edit"]
    source = lock["source_active"] or lock["source_diagnostic"]
    delta_edit = exact_delta(cohort, w["classifier.weight"], w["source"])
    matched = dict(edit_config, composition="edited")
    return {"zero": cohort.bank.base, "edited-lock": predict(cohort, available, edit_config, delta_edit, temp, scale),
            "unmodified-donor-its-best": predict(cohort, available, source, 0., temp, scale),
            "unmodified-donor-matched": predict(cohort, available, matched, 0., temp, scale),
            **{key: predict(cohort, available, edit_config, exact_delta(cohort, w[key], w["source"]), temp, scale)
               for key in ("case-shuffle", "global-shuffle")}}


def route_headroom(cohort, lock):
    bank, rows = cohort.bank, cohort.rows
    chosen = candidates(cohort, lock["document_scale"])[lock["evaluated_edit"]["scope"]]
    base_correct = bank.base == bank.gold
    truth_in_gate = chosen[np.arange(len(rows)), bank.gold[rows]]
    upper_any = bank.base.copy()
    upper_gate = bank.base.copy()
    upper_any[rows] = bank.gold[rows]
    upper_gate[rows[truth_in_gate]] = bank.gold[rows[truth_in_gate]]
    counts = {}
    for a, b in PAIRS:
        global_errors = ((bank.gold == a) & (bank.base == b)) | ((bank.gold == b) & (bank.base == a))
        counts[f"{ROLES[a]}–{ROLES[b]}"] = {"whole_cohort": int(global_errors.sum()), "routed": int(global_errors[rows].sum()),
                                           "eligible_under_selected_gate": int(global_errors[rows[truth_in_gate]].sum())}
    return {"incorrect_rows_whole_cohort": int((~base_correct).sum()), "incorrect_rows_routed": int((~base_correct[rows]).sum()),
            "pair_error_coverage": counts, "perfect_routed_hypothesis_score": working_score(baseline.stats(bank, upper_any)),
            "perfect_routed_and_selected_gate_score": working_score(baseline.stats(bank, upper_gate))}


def combine(cohorts):
    lengths = np.concatenate([np.diff(item.bank.ends) for item in cohorts])
    params = {key: np.concatenate([getattr(item.bank, key) for item in cohorts]) for key in ("keys", "gold", "scores", "experts", "old", "novel")}
    return baseline.Bank(group="pooled-confirmation", ends=np.r_[0, np.cumsum(lengths)], signature="", **params)


def confirmation():
    started = perf_counter()
    lock = baseline.read(FROZEN)
    assert lock["fit_plan_sha256"] == digest(HERE / "fit-plan.json")
    assert lock["capture_sha256"] == digest(CAPTURES) and lock["weights_sha256"] == digest(WEIGHTS)
    assert lock["preparation_sha256"] == digest(PREP)
    reports, arrays, confirm_cohorts, relevant = {}, {}, [], []
    w = load_file(str(WEIGHTS))
    with ZipFile(CAPTURES) as z:
        info, original = capture_metadata(z)
        np.testing.assert_array_equal(w["source"], original)
        if info["effective_classifier_bias"] is not None:
            np.testing.assert_array_equal(w["classifier.bias"], np.asarray(info["effective_classifier_bias"], dtype=np.float32))
        for group in ("select", "confirm", "fresh"):
            c = cohort(z, group, info, original)
            y = infer(c, w, lock)
            reports[group] = {key: outcomes(c.bank, p, c.rows) for key, p in y.items()}
            reports[group]["cohort"] = {"case_count": len(c.bank.keys), "total_rows": len(c.bank.gold), "routed_rows": len(c.rows),
                                        "novel_rows": int(c.bank.novel.sum()), "standalone_source_on_route": baseline.score(c.bank.gold[c.rows], c.source.argmax(1)),
                                        "verified_capture_logits_max_abs_difference": c.validated_delta}
            reports[group]["post_lock_route_headroom"] = route_headroom(c, lock)
            arrays[group] = y
            if group in ("confirm", "fresh"):
                relevant.append(c.rows + sum(len(prev.bank.gold) for prev in confirm_cohorts))
                confirm_cohorts.append(c)
            result = HERE / f"{group}-predictions.npz"
            np.savez_compressed(result, case_ids=c.bank.keys, offsets=c.bank.ends, gold=c.bank.gold, novel=c.bank.novel, routed=c.rows, **y)
    b = combine(confirm_cohorts)
    decisions = {name: np.concatenate([arrays[g][name] for g in ("confirm", "fresh")]) for name in arrays["confirm"]}
    idx = np.concatenate(relevant)
    reports["pooled"] = {key: outcomes(b, values, idx) for key, values in decisions.items()}
    reports["pooled"]["bootstrap_edit_minus_zero"] = {
        "all": baseline.interval(b, decisions["zero"], decisions["edited-lock"], np.full(len(b.gold), True)),
        "novel": baseline.interval(b, decisions["zero"], decisions["edited-lock"], b.novel),
    }
    reports["pooled"]["bootstrap_edit_minus_unmodified_donor"] = {
        "all": baseline.interval(b, decisions["unmodified-donor-matched"], decisions["edited-lock"], np.full(len(b.gold), True)),
        "novel": baseline.interval(b, decisions["unmodified-donor-matched"], decisions["edited-lock"], b.novel),
    }
    improves = {group: (reports[group]["edited-lock"]["all"]["macro_f1"] > reports[group]["zero"]["all"]["macro_f1"] and
                        reports[group]["edited-lock"]["all"]["pair_errors"] < reports[group]["zero"]["all"]["pair_errors"])
                for group in ("confirm", "fresh")}
    reports["joint_improvement_by_group"] = improves
    reports["meets_paired_confirmation"] = lock["active"] is not None and all(improves.values())
    reports["official_evaluated"] = False
    reports["cpu_verification_wall_seconds"] = round(perf_counter() - started, 2)
    baseline.write(HERE / "confirmation.json", reports)
    summary = {key: {name: working_score(m) for name, m in reports[key].items() if name in decisions}
               for key in ("select", "confirm", "fresh", "pooled")}
    baseline.write(HERE / "summary.json", dict(summary, joint_improvement_by_group=improves, meets_paired_confirmation=reports["meets_paired_confirmation"]))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("preregister", "fit", "confirmation"))
    action = p.parse_args().stage
    {"preregister": preregister, "fit": fit, "confirmation": confirmation}[action]()
