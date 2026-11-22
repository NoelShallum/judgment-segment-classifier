"""Write seven entry costs into an existing LegalSeg CRF, then score raw judgments."""
from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
import sys

import numpy as np
from safetensors.numpy import load_file, save_file
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT))
from discourse_span_edit import text_features
from hier_full import CACHE as TEST_LOGITS, gold_cases
from hier_model import CRF, HierModel, LABELS, path_labels
from hier_trial import DONOR, OUT as CHECKPOINTS
from preserved_emitter_edit import CACHE as TRAIN_ACTIVATIONS, edit_documents
from program_trial import TO_ID, save_json, fold
from span_eval import counts, document, summarize
from weight_edit import file_hash


PARENTS = ("exact", "balanced")
GAINS = (0., .25, .5, 1., 2., 4.)
MODES = ("entry", "pooled_entry", "diagonal")
PSEUDOCOUNT = 8.
PER_ROLE_CLIP = 2.
VAL_CASES = 64
BOOTSTRAPS = 3000


def run_counts(labels):
    labels = np.asarray(labels, dtype=np.int64)
    first = np.r_[True, labels[1:] != labels[:-1]]
    return np.bincount(labels[first], minlength=len(LABELS))


def write_delta(transition, delta, mode):
    """Use only existing transition entries. Start is 1; actual role IDs are 3 to 9."""
    delta = np.asarray(delta, dtype=transition.dtype)
    written = transition.copy()
    if mode == "diagonal":
        diagonal = np.arange(3, 10)
        written[diagonal, diagonal] += delta
        return written
    if mode not in ("entry", "pooled_entry"):
        raise ValueError(mode)
    crossing = ~np.eye(7, dtype=bool)
    written[3:, 3:] -= crossing * delta[None, :]
    written[1, 3:] -= delta
    return written


def verify_edit(transition):
    """Exercise the exact path-score identity, including singletons and document edges."""
    rng = np.random.default_rng(41882)
    delta = rng.normal(size=7).astype(np.float32)
    written = write_delta(transition, delta, "entry")
    diagonal = write_delta(transition, delta, "diagonal")
    def transition_score(t, labels):
        nodes = labels + 3
        return float(t[1, nodes[0]] + t[nodes[-1], 2]) + t[nodes[:-1], nodes[1:]].sum(dtype=np.float64)
    for length in (1, 2, 15, 40):
        for _ in range(60):
            y = rng.integers(7, size=length)
            rhs = -run_counts(y) @ delta
            np.testing.assert_allclose(transition_score(written, y) - transition_score(transition, y), rhs, atol=2e-5)
            np.testing.assert_allclose(transition_score(diagonal, y) - transition_score(transition, y) - delta[y].sum(dtype=np.float64), rhs, atol=2e-5)


def signature(d):
    return hashlib.sha256(d["text"].encode("utf-8")).hexdigest()


def val_sets(blocked):
    rows = defaultdict(list)
    with (ROOT / "references/official/val.csv").open(encoding="utf-8-sig", newline="") as source:
        for i, row in enumerate(csv.DictReader(source)):
            rows[row["Index"]].append((i, row["Text"], TO_ID[row["Label"]]))
    # Exclude the 72 + 72 judged during the previous discourse/transition selection.
    exclude = set()
    for confirm, seed in ((False, 6520), (True, 6521)):
        prior = sorted(k for k, v in rows.items() if (fold(k) == 0) == confirm and 40 <= len(v) <= 750)
        exclude.update(random.Random(seed).sample(prior, 72))
    documents = {}
    for split, confirm, seed in (("select", False, 131772), ("confirm", True, 783956)):
        pool = sorted(k for k, v in rows.items() if (fold(k) == 0) == confirm and k not in exclude and 40 <= len(v) <= 750)
        random.Random(seed).shuffle(pool)
        cohort = []
        for case in pool:
            d = document(case, rows[case])
            key = signature(d)
            if key in blocked:
                continue
            cohort.append(d)
            blocked.add(key)
            if len(cohort) == VAL_CASES:
                break
        assert len(cohort) == VAL_CASES
        documents[split] = cohort
    assert not {d["id"] for d in documents["select"]} & {d["id"] for d in documents["confirm"]}
    return documents


def augmented(logits, cue, position, profile):
    return logits + cue @ profile["cue_head"] + position @ profile["position_head"]


def direction_from_train(train, model, parents):
    predicted = {name: np.zeros(7, dtype=np.int64) for name in parents}
    gold = np.zeros(7, dtype=np.int64)
    n_units = 0
    for d in train:
        labels = np.asarray(d["reference_roles"])
        n_units += len(labels)
        gold += run_counts(labels)
        cache = TRAIN_ACTIVATIONS / f"aligned-source-train-{d['id']}.npy"
        raw = model.logits(np.load(cache, allow_pickle=False))
        assert len(raw) == len(labels)
        cue, position = text_features({**d, "units": d["reference_units"]})
        for name, profile in parents.items():
            z = augmented(raw, cue, position, profile)
            predicted[name] += run_counts(path_labels(z, profile[CRF]))
    results = {}
    for name, pred in predicted.items():
        ratio = np.log((pred + PSEUDOCOUNT) / (gold + PSEUDOCOUNT))
        pooled = float(np.log((pred.sum() + PSEUDOCOUNT) / (gold.sum() + PSEUDOCOUNT)))
        results[name] = {
            "gold_runs": gold.tolist(), "predicted_runs": pred.tolist(),
            "direction": np.clip(ratio, -PER_ROLE_CLIP, PER_ROLE_CLIP).tolist(),
            "pooled_direction": float(np.clip(pooled, -PER_ROLE_CLIP, PER_ROLE_CLIP)),
        }
    return results, n_units


@dataclass
class ScoreCase:
    case: str
    logits: dict
    token_role_counts: np.ndarray
    offsets: np.ndarray
    gold_end: np.ndarray
    gold_label: np.ndarray
    truth_run_counts: np.ndarray
    raw: np.ndarray

    def score(self, labels):
        starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
        edges = self.offsets[starts]
        right = np.r_[edges[1:], self.offsets[-1]]
        roles = labels[starts]
        matches = (right == self.gold_end[starts]) & (roles == self.gold_label[starts])
        role_matches = np.bincount(roles[matches], minlength=7)
        predicted = np.bincount(roles, minlength=7)
        confusion = self.token_role_counts.T @ np.eye(7, dtype=np.int64)[labels]
        nb = (self.gold_label[starts[1:]] >= 0).sum()
        n_gold = self.truth_run_counts.sum()
        stats = np.r_[confusion.ravel(), matches.sum(), len(starts), n_gold,
                      nb, len(starts) - 1, n_gold - 1]
        return np.r_[stats, role_matches, predicted, self.truth_run_counts]


def prepare_score(d, raw, parents):
    offsets = np.asarray([a for a, _ in d["units"]] + [len(d["text"])])
    gs = d["gold_spans"]
    gstarts = np.asarray([s["start"] for s in gs])
    gends = np.asarray([s["end"] for s in gs])
    groles = np.asarray([LABELS.index(s["label"]) for s in gs])
    words = np.fromiter((m.start() for m in re.finditer(r"\S+", d["text"])), dtype=np.int64)
    tid = np.searchsorted(offsets, words, side="right") - 1
    truth = groles[np.searchsorted(gstarts, words, side="right") - 1]
    token_role_counts = np.bincount(tid * 7 + truth, minlength=len(raw) * 7).reshape(-1, 7)
    end_by_unit = np.full(len(raw), -1, dtype=np.int64)
    label_by_unit = np.full(len(raw), -1, dtype=np.int64)
    _, idx, gj = np.intersect1d(offsets[:-1], gstarts, return_indices=True, assume_unique=True)
    end_by_unit[idx] = gends[gj]
    label_by_unit[idx] = groles[gj]
    cue, pos = text_features(d)
    inputs = {name: augmented(raw, cue, pos, patch) for name, patch in parents.items()}
    return ScoreCase(d["id"], inputs, token_role_counts, offsets, end_by_unit, label_by_unit,
                     np.bincount(groles, minlength=7), raw)


def raw_logits(d, split, model):
    if split == "test":
        with np.load(TEST_LOGITS / (d["id"] + ".npz")) as source:
            return source["raw"]
    cache = OUT / "validation-cache"
    cache.mkdir(exist_ok=True)
    p = cache / (signature(d) + ".npy")
    if p.exists():
        return np.load(p, allow_pickle=False)
    pieces = [d["text"][a:b].strip() for a, b in d["units"]]
    enc = model.logits(model.contextualize(model.words(pieces)))
    with p.with_suffix(".part").open("wb") as tmp:
        np.save(tmp, enc, allow_pickle=False)
    p.with_suffix(".part").replace(p)
    return enc


def prepare_cases(docs, split, model, parents):
    dataset = []
    for i, d in enumerate(docs):
        raw = raw_logits(d, split, model)
        assert len(raw) == len(d["units"])
        case = prepare_score(d, raw, parents)
        if i < 3:
            for parent in PARENTS:
                y = path_labels(case.logits[parent], parents[parent][CRF])
                np.testing.assert_array_equal(case.score(y)[:55], counts(d, d["units"], y))
        dataset.append(case)
        if (i + 1) % 100 == 0:
            print(split, i + 1, flush=True)
    return dataset


def options(case_data, transition, name):
    return np.stack([x.score(path_labels(x.logits[name], transition)) for x in case_data])


def metric(a):
    return float(2 * a[49] / (a[50] + a[51])), float(a[:49].reshape(7, 7).trace() / a[:49].sum())


def full_score(a):
    sums = a.sum(axis=0)
    result = summarize(sums[:55])
    matched, pred, gold = sums[55:62], sums[62:69], sums[69:76]
    result["roles"] = {name: {"matched": int(m), "predicted": int(p), "gold": int(g),
                              "span_f1": float(2 * m / (p + g)) if p + g else None}
                       for name, m, p, g in zip(LABELS, matched, pred, gold)}
    return result


def bootstrap(candidate, control):
    rng = np.random.default_rng(852311)
    n = len(control)
    deltas = []
    for _ in range(BOOTSTRAPS):
        indices = rng.integers(n, size=n)
        deltas.append(np.subtract(metric(candidate[indices].sum(0)), metric(control[indices].sum(0))))
    lo, hi = np.quantile(deltas, [.025, .975], axis=0)
    return {"exact_span_f1": [float(lo[0]), float(hi[0])], "token_accuracy": [float(lo[1]), float(hi[1])]}


def read_profiles():
    return {name: load_file(str(CHECKPOINTS / f"whole-discourse-{name}.safetensors")) for name in PARENTS}


def prove_selected(study, parents, direction):
    frozen = {}
    grids = {}
    for name, parent in parents.items():
        grid = []
        for mode in MODES:
            for gain in GAINS:
                base_delta = direction[name]["pooled_direction"] if mode == "pooled_entry" else direction[name]["direction"]
                delta = gain * np.broadcast_to(base_delta, 7)
                written = write_delta(parent[CRF], delta, mode)
                values = options(study, written, name)
                span, token = metric(values.sum(0))
                grid.append({"mode": mode, "gain": gain, "span": span, "token": token, "delta": delta.tolist()})
        base = next(row for row in grid if row["mode"] == "entry" and row["gain"] == 0)
        role = [row for row in grid if row["mode"] == "entry"]
        # The practical parent sacrifices at most one point on the same selection cases.
        if name == "balanced":
            role = [row for row in role if row["token"] >= base["token"] - .01]
        choice = max(role, key=lambda x: (x["span"], x["token"], -x["gain"]))
        comparisons = {row["mode"]: row for row in grid if row["gain"] == choice["gain"]}
        comparisons["parent"] = base
        frozen[name] = {}
        for mode, row in comparisons.items():
            written = write_delta(parent[CRF], row["delta"], row["mode"])
            target = OUT / f"whole-{name}-{mode}.safetensors"
            if mode != "parent":
                save_file({**parent, CRF: written}, str(target))
            else:
                target = CHECKPOINTS / f"whole-discourse-{name}.safetensors"
            frozen[name][mode] = {**row, "file": target.name, "sha256": file_hash(target)}
        grids[name] = grid
    return frozen, grids


def evaluate(dataset, split, parents, frozen):
    results = {}
    for name in parents:
        scores = {}
        for mode, profile in frozen[name].items():
            path = (CHECKPOINTS if mode == "parent" else OUT) / profile["file"]
            assert file_hash(path) == profile["sha256"]
            patch = load_file(str(path))
            scores[mode] = options(dataset, patch[CRF], name)
        control = scores["parent"]
        result = {}
        for mode, value in scores.items():
            result[mode] = {"scores": full_score(value), "case_bootstrap_delta_ci95": bootstrap(value, control)}
        result["entry_minus_pooled_ci95"] = bootstrap(scores["entry"], scores["pooled_entry"])
        result["entry_minus_diagonal_ci95"] = bootstrap(scores["entry"], scores["diagonal"])
        results[name] = result
        print(split, name, [(k, tuple(round(v, 6) for v in metric(a.sum(0)))) for k, a in scores.items()], flush=True)
        np.savez_compressed(OUT / f"{split}-{name}-paired-counts.npz",
                            cases=np.array([d.case for d in dataset]), **scores)
    return results


def main():
    torch.set_num_threads(1)
    model = HierModel(DONOR)
    parents = read_profiles()
    for parent in parents.values():
        verify_edit(parent[CRF])
    train = edit_documents()
    train_keys = {signature(d) for d in train}
    assert len(train) == 144 and len(train_keys) == len(train)
    blocked = train_keys.copy()
    groups = val_sets(blocked)
    directions, train_n = direction_from_train(train, model, parents)
    print("Train aligned source rows", train_n, directions, flush=True)
    select = prepare_cases(groups["select"], "select", model, parents)
    chosen, grids = prove_selected(select, parents, directions)
    protocol = {
        "scope": "Seven-role raw full judgments. Authored discourse profiles, then physical CRF arrival cost. Gold original source-row cuts are used only for the train count estimates. Validation and test infer cuts from raw text; archived test logits are reused. Case IDs are split-local. The official test has been used for earlier benchmark comparisons.",
        "formula": "delta_k = clip(log((predicted_train_runs_k+8)/(gold_train_runs_k+8)), -2, 2) times selected gain. Subtract delta_k at START->k and each j->k with j != k. Total change for a path is minus sum_k delta_k times its number of runs of k. The diagonal ablation instead adds delta_k to k->k. The pooled ablation uses the pooled train run ratio for all roles at the same gain.",
        "selection": "Both parents choose the gain for the seven-direction entry method on 64 selection cases by exact-span F1, then token accuracy, then smaller gain. The balanced parent requires token accuracy >= its baseline minus .01. Both ablations use the same gain as the entry method. Confirmation is case-disjoint and does not select a second gain. Full test artifacts and labels are loaded only after the candidates are frozen.",
        "dataset_provenance": json.loads((ROOT / "references/official/verified-local.json").read_text()),
        "parameters": {"pseudocount": PSEUDOCOUNT, "direction_clip": PER_ROLE_CLIP, "gains": GAINS,
                       "validation_filter": "40 to 750 source rows inclusive, selected separately from SHA-256-based folds. The prior two 72-case transition/discourse cohorts are excluded.",
                       "bootstrap": "3000 paired whole-judgment percentile replicates, seed 852311"},
        "model_sha256": file_hash(DONOR / "hier.safetensors"), "script_sha256": file_hash(Path(__file__)),
        "role_order": LABELS, "train_case_ids": [d["id"] for d in train], "train_source_rows": train_n,
        "selection_case_ids": [d["id"] for d in groups["select"]],
        "confirmation_case_ids": [d["id"] for d in groups["confirm"]],
        "train_estimates": directions, "frozen": chosen, "selection_grid": grids,
    }
    path = OUT / "frozen-selection.json"
    save_json(path, protocol)
    frozen_hash = file_hash(path)
    print("FROZEN", frozen_hash, {k: u["entry"]["gain"] for k, u in chosen.items()}, flush=True)
    result = {"freeze_sha256": frozen_hash, "selection": evaluate(select, "select", parents, chosen)}
    del select
    confirmation = prepare_cases(groups["confirm"], "confirm", model, parents)
    result["confirmation"] = evaluate(confirmation, "confirm", parents, chosen)
    del confirmation
    case_rows = gold_cases()
    docs = [document(case, rows) for case, rows in case_rows.items()]
    test_duplicate_signatures = [d["id"] for d in docs if signature(d) in blocked]
    result["exact_text_overlap_test_with_fit_or_development"] = test_duplicate_signatures
    test = prepare_cases(docs, "test", None, parents)
    assert len(test) == 712
    result["test_cases"] = len(test)
    result["test"] = evaluate(test, "test", parents, chosen)
    save_json(OUT / "results.json", result)


if __name__ == "__main__":
    main()
