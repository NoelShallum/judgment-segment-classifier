"""Write an expanded document readout while ablating the original model's role prior."""

import argparse
from dataclasses import dataclass
from hashlib import file_digest
from itertools import product
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from safetensors.numpy import load_file, save_file
from scipy.linalg import cho_factor, cho_solve


HERE = Path(__file__).resolve().parent
EYE = np.eye(7, dtype=np.float32)
RARITIES = (0., .40)
RIDGES = (.035, .20)
MIXES = (.22, .45, .70, 1.)
GROUPS = ("select", "confirm", "fresh")


@dataclass
class Panel:
    keys: np.ndarray
    offsets: np.ndarray
    gold: np.ndarray
    compact: np.ndarray
    pre: np.ndarray
    reviewed: np.ndarray | None = None
    score: np.ndarray | None = None
    novel: np.ndarray | None = None


def hash_file(path):
    with path.open("rb") as inp:
        return file_digest(inp, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, item):
    part = path.with_suffix(".part")
    part.write_text(json.dumps(item, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    part.replace(path)


def checked_npz(file):
    with np.load(file, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def load_fit(scale, block, reviewed, protocol):
    compact = checked_npz(scale / f"train-{block:02d}.npz")
    arr = checked_npz(HERE / f"anchor-{block:02d}.npz")
    source = HERE / f"anchor-{block:02d}.npz"
    assert hash_file(source) == read(source.with_suffix(".json"))["file_sha256"]
    np.testing.assert_array_equal(compact["case"], protocol["case_ids"][64 * block:64 * (block + 1)])
    for name, expected in (("case_ids", compact["case"]), ("gold", compact["y"]), ("offsets", compact["offsets"])):
        np.testing.assert_array_equal(arr[name], expected)
    final = None
    if reviewed is not None:
        f = reviewed / f"reviewed-{block:02d}.npz"
        receipt = read(reviewed / "runtime-receipt.json")["shards"][f.name]
        assert receipt["input_sha256"] == hash_file(source) and receipt["file_sha256"] == hash_file(f)
        active = checked_npz(f)
        for name in ("case_ids", "offsets", "gold", "base", "positions"):
            np.testing.assert_array_equal(active[name], arr[name])
        final = active["reference"]
    return Panel(compact["case"], compact["offsets"], compact["y"], compact["x"], arr["base"], final)


def load_validation(scale, group):
    assert group in GROUPS
    a = checked_npz(scale / (group + "-panel.npz"))
    b = checked_npz(HERE / ("validation-" + group + ".npz"))
    for name, match in (("case_ids", "case"), ("offsets", "offsets"), ("gold", "y")):
        np.testing.assert_array_equal(b[name], a[match])
    return Panel(a["case"], a["offsets"], a["y"], a["x"], b["pre"], b["reviewed"], b["score"], b["novel"])


def anchored_features(doc, labels, p):
    output = []
    for a, b in zip(doc.offsets, doc.offsets[1:]):
        n = b - a
        rows = np.arange(n)
        frac = (rows[:, None] + .5) / n
        here = EYE[labels[a:b]]
        totals = np.vstack((np.zeros((1, 7)), np.cumsum(here, axis=0, dtype=np.float64)))
        doc_hist = np.broadcast_to(totals[-1] / n, here.shape)
        pieces = [here, here * p[a:b], p[a:b] - here, here * (frac - .5), doc_hist, here * doc_hist]
        for size in (1, 6, 24):
            low = np.maximum(rows - size, 0)
            high = np.minimum(rows + size + 1, n)
            pieces.extend(((totals[rows] - totals[low]) / np.maximum(rows - low, 1)[:, None],
                           (totals[high] - totals[rows + 1]) / np.maximum(high - rows - 1, 1)[:, None]))
        output.append(np.column_stack(pieces).astype(np.float32))
    return np.concatenate(output)


def build_design(doc, feature_plan, scale_design, docbasis, probabilities, include_reviewed):
    base = docbasis(scale_design(doc.compact, doc.offsets), feature_plan).astype(np.float32)
    donor = .5 * (probabilities(doc.compact[:, :7] * .65) + probabilities(doc.compact[:, 62:69] * .65))
    pieces = [base, anchored_features(doc, doc.pre, donor)]
    if include_reviewed:
        if doc.reviewed is None:
            raise ValueError("An exact reviewed anchor is required for every record")
        pieces.append(anchored_features(doc, doc.reviewed, donor))
    full = np.column_stack(pieces)
    groups = {"no-anchor": np.arange(base.shape[1]),
              "pre-review": np.arange(base.shape[1] + pieces[1].shape[1])}
    if include_reviewed:
        groups["exact-reviewed"] = np.r_[groups["no-anchor"], np.arange(len(groups["pre-review"]), full.shape[1])]
    return full, groups


def solve(gram, rhs, ridge, intercept):
    scale = np.maximum(np.diag(gram), 1.)
    scale[intercept] *= .001
    return cho_solve(cho_factor(gram + np.diag(ridge * scale), check_finite=False), rhs, check_finite=False).astype(np.float32)


def metrics(y, p):
    m = np.bincount(7 * y.astype(np.int64) + p, minlength=49).reshape(7, 7)
    n = m.diagonal()
    f = (2 * n / (m.sum(0) + m.sum(1)).clip(min=1)).mean()
    return {"macro_f1": float(f), "accuracy": float(n.sum() / max(1, m.sum())), "pair_errors": int(m[0, 5] + m[5, 0]),
            "none_to_reasoning": int(m[0, 5]), "reasoning_to_none": int(m[5, 0]), "confusion": m.tolist()}


def report(panel, p):
    mask = panel.novel
    return {"all": metrics(panel.gold, p), "novel": metrics(panel.gold[mask], p[mask])}


def rank(scores):
    full, novel = scores["all"], scores["novel"]
    return ((full["macro_f1"] + novel["macro_f1"]) / 2, -full["pair_errors"] - novel["pair_errors"], full["accuracy"])


def run(scale, out, blocks, reviewed):
    # The pinned source modules provide the compact feature layout. Their CUDA
    # training functions are never invoked by this CPU linear solve.
    import run_scale
    from solve import docbasis, probabilities, delta_ci

    out.mkdir(parents=True, exist_ok=True)
    manifest = read(HERE / "manifest.json")
    for name, digest in manifest["files"].items():
        assert digest == hash_file(HERE / name)
    protocol = read(HERE / "protocol.json")
    basis = scale / "basis.safetensors"
    assert hash_file(basis) == read(HERE / "fit-source.json")["scale_basis_sha256"]
    feature_plan = load_file(basis)
    include_reviewed = reviewed is not None
    original_fit = checked_npz(scale / "fit-panel.npz")
    fit_doc_keys = set(original_fit["case"])
    role_counts = np.bincount(original_fit["y"], minlength=7) + 1
    mass = role_counts / role_counts.sum()
    inputs = {i: hash_file(scale / f"train-{i:02d}.npz") for i in range(blocks)}
    assert not fit_doc_keys & set(protocol["case_ids"][:64 * blocks])
    design = lambda p: build_design(p, feature_plan, run_scale.design, docbasis, probabilities, include_reviewed)
    selector = load_validation(scale, "select")
    assert not set(selector.keys) & set(protocol["case_ids"][:64 * blocks])
    selected_X, columns = design(selector)
    baseline = report(selector, selector.score.argmax(1))
    gram, rhs = {}, {}
    total_cases = total_rows = 0
    for i in range(blocks):
        start = perf_counter()
        fit = load_fit(scale, i, reviewed, protocol)
        if hash_file(scale / f"train-{i:02d}.npz") != inputs[i]:
            raise ValueError("Compact input changed after lock")
        x, idx = design(fit)
        for k, v in columns.items():
            np.testing.assert_array_equal(v, idx[k])
        for rare in RARITIES:
            n = (mass[fit.gold] ** -rare).astype(np.float32)
            weighted = x * n[:, None]
            g = (x.T @ weighted).astype(np.float64)
            b = (weighted.T @ EYE[fit.gold]).astype(np.float64)
            gram[rare] = gram.get(rare, 0.) + (g + g.T) * .5
            rhs[rare] = rhs.get(rare, 0.) + b
        total_cases += len(fit.keys)
        total_rows += len(fit.gold)
        checkpoint = out / f"moments-{total_cases:03d}.npz"
        part = checkpoint.with_suffix(".part")
        with part.open("wb") as handle:
            np.savez(handle, **{name + str(k): x for name, coll in (("gram_", gram), ("rhs_", rhs)) for k, x in coll.items()})
        part.replace(checkpoint)
        write(out / "progress.json", {"cases": total_cases, "rows": total_rows, "block": i,
                                      "moments_sha256": hash_file(checkpoint), "seconds": perf_counter() - start})
        print("anchor moments", total_cases, total_rows, f"{perf_counter() - start:.2f}s", flush=True)
    tensors, records = {}, {}
    intercept = columns["no-anchor"][-1]
    for name, col in columns.items():
        # All arms see the same compact features and the same rows, differing only
        # in which incumbent label stream supplies their additional features.
        best = None
        grid = []
        for rare, ridge in product(RARITIES, RIDGES):
            coef = solve(gram[rare][np.ix_(col, col)], rhs[rare][col], ridge, intercept)
            response = selected_X[:, col] @ coef
            for mix in MIXES:
                new = ((1. - mix) * selector.score + mix * response).argmax(1)
                stats = report(selector, new)
                quality = rank(stats)
                grid.append({"parameters": [rare, ridge, mix], "rank": quality})
                keep = (quality, (rare, ridge, mix), stats, coef)
                if best is None or keep[0] > best[0]:
                    best = keep
        key, rule, results, coef = best
        tensors[name + ".coef"] = np.ascontiguousarray(coef)
        tensors[name + ".columns"] = col.astype(np.int64)
        records[name] = {"rank": key, "selection": results, "settings": rule, "grid": grid}
    weights = out / "expanded-anchors.safetensors"
    part = weights.with_suffix(".part")
    save_file(tensors, part)
    part.replace(weights)
    for name, value in load_file(weights).items():
        np.testing.assert_array_equal(value, tensors[name])
    choice, chosen = max(records, key=lambda name: records[name]["rank"]), "doc-semantic"
    if records[choice]["rank"] > rank(baseline):
        chosen = choice
    freeze = {"selection": chosen, "ridges": records, "incumbent_selector": baseline, "weights_sha256": hash_file(weights),
              "basis_sha256": hash_file(basis), "compact_fit_shards": inputs, "cases": total_cases, "rows": total_rows,
              "protocol_sha256": hash_file(HERE / "protocol.json"), "official_test_read": False, "device": "CPU",
              "selection_objective": "Equal weight for full and no-key seven-role macro-F1, then pair error tally and accuracy."}
    freeze_path = out / "freeze.json"
    write(freeze_path, freeze)

    # Confirmation labels are fetched only after the weights and combination rule
    # have been frozen. The 48-case check confirms the recorded winner and scores.
    rows = {}
    for g in GROUPS:
        panel = selector if g == "select" else load_validation(scale, g)
        assert not set(panel.keys) & set(protocol["case_ids"][:64 * blocks])
        z, shape = (selected_X, columns) if g == "select" else design(panel)
        base_p = panel.score.argmax(1)
        predictions = {"doc-semantic": base_p}
        for name, item in records.items():
            _, _, mix = item["settings"]
            exact = z[:, shape[name]] @ tensors[name + ".coef"]
            predictions[name] = ((1 - mix)*panel.score + mix*exact).argmax(1)
            if g == "select":
                assert report(panel, predictions[name]) == item["selection"]
        rows[g] = {"judgments": len(panel.keys), "rows": len(base_p), "no_key_rows": int(panel.novel.sum()),
                   "scores": {name: report(panel, value) for name, value in predictions.items()},
                   "paired": {name: delta_ci(panel, value, base_p) for name, value in predictions.items() if name != "doc-semantic"}}
        np.savez_compressed(out / (g + "-predictions.npz"), case_ids=panel.keys, offsets=panel.offsets, gold=panel.gold, novel=panel.novel, **predictions)
        print(g, {name: (round(item["all"]["macro_f1"]*100, 4), round(item["novel"]["macro_f1"]*100, 4), item["all"]["pair_errors"])
                  for name, item in rows[g]["scores"].items()}, flush=True)
    write(out / "results.json", {"freeze_sha256": hash_file(freeze_path), "selection": chosen, "groups": rows})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=Path, default=Path("/content/reviewer2/scale"))
    parser.add_argument("--reviewed", type=Path, help="Path containing the exact T4 incumbent predictions")
    parser.add_argument("--output", type=Path, default=HERE / "head-fit")
    parser.add_argument("--blocks", type=int, required=True, choices=range(1, 9))
    args = parser.parse_args()
    for directory in (args.scale.resolve(), args.scale.resolve().parent / "legal", args.scale.resolve().parent / "context"):
        sys.path.insert(0, str(directory))
    run(args.scale, args.output, args.blocks, args.reviewed)
