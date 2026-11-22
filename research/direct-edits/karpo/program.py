"""Write seven Karpo readout rows from frozen states and evaluate matched controls."""
import argparse
from hashlib import sha256
import os
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
from safetensors.numpy import load_file, save_file

from assets import HERE, MODEL, READOUT_SHA, WEIGHT_SHA, checksum, read, write


CHECKPOINT = "readout.safetensors"
BASE = MODEL / CHECKPOINT
N = 7


def load_base():
    assert checksum(BASE) == READOUT_SHA
    base = load_file(str(BASE))["weight"]
    assert base.shape == (255, 1024) and base.dtype == np.float32
    return base


def labels(plan, split):
    f = HERE / "truth.json"
    assert checksum(f) == plan["truth_sha256"]
    return {k: np.asarray(y, np.int8) for k, y in read(f)[split].items()}


def backend(root):
    verified = read(root / "backend.json")
    assert verified["device_capability"] == [7, 5] and verified["compute_dtype"] == "torch.float32"
    assert verified["protocol_sha256"] == checksum(HERE / "protocol.json")
    assert len(verified["nvidia_smi"]) == 1 and "T4" in verified["nvidia_smi"][0]
    return verified


def load_block(path, facts):
    receipt = read(path.with_suffix(".json"))
    assert receipt["sha256"] == checksum(path) and receipt["backend"] == facts and receipt["parameter_source"] == WEIGHT_SHA
    with np.load(path, allow_pickle=False) as arrays:
        data = {k: arrays[k] for k in arrays.files}
    np.testing.assert_array_equal(data["targets"], receipt["target_rows"])
    np.testing.assert_array_equal(data["option_order"], receipt["option_order"])
    assert data["hidden"].shape == (len(data["targets"]), 1024) and data["hidden"].dtype == np.float32
    assert data["logits"].shape == (len(data["targets"]), 7) and data["logits"].dtype == np.float32
    assert np.isfinite(data["hidden"]).all() and np.isfinite(data["logits"]).all()
    return data, receipt


def cases(root, split, ids, plan, base):
    facts = backend(root)
    result = {}
    for key in ids:
        folder = root / "states" / split / key
        receipt = read(folder / "complete.json")
        files = [f for f in sorted(folder.glob("*.npz")) if f.stem.isdecimal()]
        assert len(files) == receipt["blocks"]
        pieces, receipts = zip(*(load_block(f, facts) for f in files), strict=True)
        pos = np.concatenate([chunk["targets"] for chunk in pieces])
        np.testing.assert_array_equal(pos, np.arange(plan["rows"][key]))
        for chunk in pieces:
            np.testing.assert_array_equal(chunk["option_order"], np.arange(7))
        x = np.concatenate([a["hidden"] for a in pieces])
        z = np.concatenate([a["logits"] for a in pieces])
        replay = x @ base[:7].T
        np.testing.assert_allclose(replay, z, atol=.02, rtol=1e-4)
        result[key] = {"x": x, "z": z, "shards": {p.name: v["sha256"] for p, v in zip(files, receipts)},
                       "cpu_replay_max_error": float(np.max(np.abs(replay - z))),
                       "cpu_native_argmax_differences": int(np.count_nonzero(replay.argmax(1) != z.argmax(1))),
                       "clipped_rows": [v["target_rows"][0] for v in receipts if v["clip"]]}
    return result


def join(data, gold, keys):
    return tuple(np.concatenate([q[k] for k in keys]) for q in ({k: v["x"] for k, v in data.items()},
                                                                 {k: v["z"] for k, v in data.items()}, gold))


def write_delta(x, native, y, ridge, balance, margin):
    # Weighted dual ridge remains small because the fit has 357 rows and 1024 features.
    counts = np.maximum(np.bincount(y, minlength=N), 1)
    w = (len(y) / counts[y]) ** balance
    root_weight = np.sqrt(w / w.mean())
    target = margin * (np.eye(N)[y] - 1. / N) - (native - native.mean(1, keepdims=True))
    f = x.astype(np.float64) * root_weight[:, None]
    gram = f @ f.T
    regular = ridge * gram.trace() / len(gram)
    gram.flat[::len(gram) + 1] += regular
    delta = (f.T @ np.linalg.solve(gram, target * root_weight[:, None])).T
    return delta - delta.mean(0, keepdims=True)


def confusion(y, p):
    return np.bincount(N * y.astype(np.int64) + p, minlength=N*N).reshape(N, N)


def div(a, b):
    return np.divide(a, b, where=b > 0, out=np.zeros_like(a, dtype=float))


def sentence_metrics(c):
    hit = np.diagonal(c, axis1=-2, axis2=-1)
    f1 = div(2 * hit, c.sum(-1) + c.sum(-2))
    return f1.mean(-1), hit.sum(-1) / c.sum((-1, -2)), f1


def fit(root, plan, base):
    folder = root / "fit"
    folder.mkdir(exist_ok=True)
    if (folder / "freeze.json").exists():
        raise ValueError("This training write is already frozen")
    train = cases(root, "train", plan["fit_cases"], plan, base)
    gold = labels(plan, "train")
    folds = plan["training_cross_validation"]
    assert sorted(sum(folds, [])) == sorted(train)
    partitions = [(join(train, gold, [k for k in train if k not in group]), join(train, gold, group)) for group in folds]
    grid, chosen, best = [], None, None
    margin = plan["target_margin"]
    for strength in plan["ridge"]:
        for balance in plan["balance"]:
            probed = [(write_delta(x, z, y, strength, balance, margin), partial) for (x, z, y), partial in partitions]
            for gain in plan["gains"]:
                counts = sum(confusion(y, (z + gain * (x @ d.T)).argmax(1)) for d, (x, z, y) in probed)
                f1, acc, _ = sentence_metrics(counts)
                settings = [strength, balance, gain, float(f1), float(acc)]
                grid.append(settings)
                order = (f1, acc, -gain, strength, -balance)
                if best is None or order > best:
                    best, chosen = order, settings
    strength, balance, gain, f1, accuracy = chosen
    x, z, y = join(train, gold, plan["fit_cases"])
    edit = gain * write_delta(x, z, y, strength, balance, margin)
    shuffle = gain * write_delta(x, z, np.random.default_rng(plan["shuffle_seed"]).permutation(y), strength, balance, margin)
    shuffle *= np.linalg.norm(edit) / np.linalg.norm(shuffle)
    physical = {}
    for arm, patch in (("edited", edit), ("shuffled", shuffle)):
        home = folder / arm
        home.mkdir(exist_ok=True)
        w = base.copy()
        w[:7] = (base[:7].astype(np.float64) + patch).astype(np.float32)
        destination = home / CHECKPOINT
        partial = destination.with_suffix(".part")
        save_file({"weight": np.ascontiguousarray(w)}, str(partial))
        loaded = load_file(str(partial))["weight"]
        np.testing.assert_array_equal(loaded, w)
        np.testing.assert_array_equal(loaded[7:], base[7:])
        partial.replace(destination)
        physical[arm] = {"sha256": checksum(destination), "shape": list(w.shape), "changed_rows": list(range(7)),
                         "changed_values": int(np.count_nonzero(w != base)), "solve_rank": int(np.linalg.matrix_rank(patch, tol=1e-9)),
                         "additive_frobenius": float(np.linalg.norm(w.astype(np.float64) - base)),
                         "nonrole_rows_sha256": sha256(w[7:].tobytes()).hexdigest()}
    f1_base, acc_base, _ = sentence_metrics(confusion(y, z.argmax(1)))
    result = {"protocol_sha256": checksum(HERE / "protocol.json"), "program_sha256": checksum(Path(__file__)),
              "base_readout_sha256": READOUT_SHA, "base_checkpoint_sha256": WEIGHT_SHA, "judgments": plan["fit_cases"],
              "rows": len(y), "role_support": np.bincount(y, minlength=7).tolist(), "selection": {"ridge": strength, "balance": balance, "gain": gain},
              "heldout_training_macro_f1": f1, "heldout_training_accuracy": accuracy, "native_training_macro_f1": float(f1_base),
              "native_training_accuracy": float(acc_base), "grid": grid, "moves": physical,
              "training_shards": {k: v["shards"] for k, v in train.items()}, "gradient_steps": 0}
    write(folder / "freeze.json", result)
    print("Frozen native decision rows", result["selection"], "training CV macro F1", f1, physical, flush=True)


def spans(y):
    cuts = np.flatnonzero(np.r_[True, y[1:] != y[:-1], True])
    return {(int(a), int(b), int(y[a])) for a, b in zip(cuts, cuts[1:])}


def counts(y, p):
    a, b = spans(y), spans(p)
    return len(a & b), len(a), len(b)


def metrics(gold, predict, keys, draws):
    cm = np.stack([confusion(gold[k], predict[k]) for k in keys])
    sc = np.stack([counts(gold[k], predict[k]) for k in keys])
    base = cm.sum(0)
    f1, acc, roles = sentence_metrics(base)
    good, total, proposed = sc.sum(0)
    f1s, accs, _ = sentence_metrics(cm[draws].sum(1))
    d = sc[draws].sum(1)
    sample = np.column_stack((f1s, accs, div(2*d[:, 0], d[:, 1] + d[:, 2])))
    return {"macro_f1": float(f1), "accuracy": float(acc), "per_role_f1": roles.tolist(), "confusion": base.tolist(),
            "span_f1": float(2*good/(total + proposed)), "span_recall": float(good/total), "span_precision": float(good/proposed),
            "exact_spans": int(good), "gold_spans": int(total), "predicted_spans": int(proposed)}, sample


def sensitivity(root, keys, heads, gold):
    facts = backend(root)
    report = {}
    edit = heads["edited"][:7]
    original = heads["native"][:7]
    for case in keys:
        place = root / "states" / "val" / case
        if not (place / "sensitivity.json").exists():
            continue
        canon, memo = load_block(place / "0000.npz", facts)
        rows = canon["targets"]
        prior0 = canon["logits"].argmax(1)
        prior1 = (canon["hidden"] @ edit.T).argmax(1)
        v = report[case] = {}
        for mode in ("rotated-options", "reversed-questions"):
            data, stamp = load_block(place / (mode + ".npz"), facts)
            assert set(data["targets"]) == set(rows)
            p = {a: i for i, a in enumerate(data["targets"])}
            order = [p[t] for t in rows]
            role = data["option_order"]
            h = data["hidden"][order]
            native = role[data["logits"][order].argmax(1)]
            fixed = role[(h @ edit.T).argmax(1)]
            moved = role[(h @ (original + (edit - original)[role]).T).argmax(1)]
            arms = {"native": native, "fixed_code_rows": fixed, "role_transported_rows": moved}
            v[mode] = {"targets": rows.tolist(), "displayed_role_order": role.tolist(), "tokens": [memo["tokens"], stamp["tokens"]],
                       "accuracy": {k: float(np.mean(a == gold[case][rows])) for k, a in arms.items()},
                       "agreement": {k: float(np.mean(a == (prior0 if k == "native" else prior1))) for k, a in arms.items()},
                       "predictions": {k: a.tolist() for k, a in arms.items()}, "original_native": prior0.tolist(),
                       "original_edited": prior1.tolist()}
    return report


def score(root, phase, plan, base):
    receipt = root / "fit" / "freeze.json"
    lock = read(receipt)
    assert lock["protocol_sha256"] == checksum(HERE / "protocol.json") and lock["program_sha256"] == checksum(Path(__file__))
    keys = plan["fallback_validation_first_four_cases"] if phase == "four" else plan["validation_cases"]
    holdout = cases(root, "val", keys, plan, base)
    gold = labels(plan, "val")
    source = HERE / "incumbent.json"
    assert checksum(source) == plan["incumbent_sha256"]
    old = read(source)
    heads = {"native": base}
    pred = {"native": {k: r["z"].argmax(1) for k, r in holdout.items()}, "incumbent": {k: np.array(old[k]) for k in keys}}
    for arm in ("edited", "shuffled"):
        file = root / "fit" / arm / CHECKPOINT
        assert checksum(file) == lock["moves"][arm]["sha256"]
        w = heads[arm] = load_file(str(file))["weight"]
        np.testing.assert_array_equal(w[7:], base[7:])
        pred[arm] = {k: (v["x"] @ w[:7].T).argmax(1) for k, v in holdout.items()}
    n = len(keys)
    ix = np.random.default_rng(plan["bootstrap_seed"]).integers(n, size=(plan["bootstrap_draws"], n))
    scores = {arm: metrics(gold, p, keys, ix) for arm, p in pred.items()}
    paired = {k: dict(zip(("macro_f1", "accuracy", "span_f1"), np.quantile(scores["edited"][1] - scores[k][1], [.025, .975], axis=0).T.tolist()))
              for k in ("native", "shuffled", "incumbent")}
    predictions = root / (phase + "-predictions.json")
    write(predictions, {name: {k: y.tolist() for k, y in items.items()} for name, items in pred.items()})
    result = {"split": phase, "cases": keys, "rows": sum(len(gold[k]) for k in keys),
              "fit_freeze_sha256": checksum(receipt), "protocol_sha256": checksum(HERE / "protocol.json"),
              "predictions_sha256": checksum(predictions), "scores": {k: s[0] for k, s in scores.items()},
              "paired_95_edited_minus": paired, "native_cpu_vs_gpu_argmax_differences": sum(v["cpu_native_argmax_differences"] for v in holdout.values()),
              "native_replay_max_error": max(v["cpu_replay_max_error"] for v in holdout.values()),
              "clipped_rows": {k: v["clipped_rows"] for k, v in holdout.items() if v["clipped_rows"]},
              "input_sensitivity": sensitivity(root, keys, heads, gold), "official_test_opened": False}
    write(root / (phase + ".json"), result)
    print(phase, result["rows"], {k: (v["macro_f1"], v["span_f1"]) for k, v in result["scores"].items()}, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("fit", "four", "validation"))
    ap.add_argument("--out", type=Path, default=HERE / "run")
    args = ap.parse_args()
    plan = read(HERE / "protocol.json")
    base = load_base()
    if args.action == "fit":
        fit(args.out, plan, base)
    else:
        score(args.out, args.action, plan, base)


if __name__ == "__main__":
    main()
