"""Count train-only role and boundary associations for inspectable discourse cues."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from hashlib import file_digest, sha256
from itertools import groupby, islice
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from safetensors.numpy import save_file

import cues

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CSV = ROOT / "references/official"
SOURCE = CSV / "train.csv"
PINNED = "60028fc085e3cb018e8d49a8879ba796dd03a0838367891f3312f9e576921794"
LABELS = ("None", "Facts", "Issue", "Arguments of Petitioner", "Arguments of Respondent", "Reasoning", "Decision")
TO_ID = {x: i for i, x in enumerate(LABELS)}
FORMS = ("local", "context", "edge")
FEATURES = cues.feature_names()
CLASSES = {"local": 7, "context": 7, "edge": 49}
EYE = np.eye(7, dtype=np.float32)
EDGE_EYE = np.eye(49, dtype=np.float32)
TRAIN_COUNTS = (1122376, 4984)
SAMPLE_COUNT = 1536
SAMPLE_SEED = "attoflow-discourse-reference-cues-v1|"


def digest(path):
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def cases(split):
    if split not in ("train", "val"):
        raise ValueError("This experiment uses the public train and validation splits")
    seen = set()
    with (CSV / (split + ".csv")).open(encoding="utf-8-sig", newline="") as f:
        for key, group in groupby(csv.DictReader(f), key=lambda row: row["Index"]):
            if key in seen:
                raise ValueError(f"Noncontiguous judgment {key}")
            seen.add(key)
            values = [(row["Text"], TO_ID[row["Label"]]) for row in group if row["Text"].strip()]
            if values:
                texts, roles = zip(*values)
                yield key, texts, np.asarray(roles, dtype=np.int8)


def blocks(stream, size=32):
    while items := list(islice(stream, size)):
        yield items


def zero():
    return {f"{mode}.{kind}.{name}": np.zeros((cues.PHASES, *shape), np.float64)
            for mode in ("real", "shuffled") for kind in FORMS
            for name, shape in (("present", (len(FEATURES[kind]), CLASSES[kind])), ("marginal", (CLASSES[kind],)))}


def aggregate(items):
    vals = zero()
    total = np.zeros(7, dtype=np.int64)
    start = perf_counter()
    for key, texts, y in items:
        f = cues.extract(texts)
        n = len(y)
        total += np.bincount(y, minlength=7)
        salt = int.from_bytes(sha256(("discourse-within-case-rotation|" + key).encode()).digest()[:4], "big")
        rotates = np.roll(y, 1 + salt % (n - 1)) if n > 1 else y
        mass = (np.float32(230) / n) ** np.float32(.23)
        for mode, label in (("real", y), ("shuffled", rotates)):
            node = EYE[label] * mass
            transitions = np.zeros((n, 49), dtype=np.float32)
            transitions[1:] = EDGE_EYE[7 * label[:-1].astype(np.int32) + label[1:]] * mass
            for k in range(cues.PHASES):
                p = f.phase == k
                for kind, response in (("local", node), ("context", node), ("edge", transitions)):
                    x = getattr(f, kind)[p]
                    y_p = response[p]
                    base = f"{mode}.{kind}."
                    vals[base + "present"][k] += x.T @ y_p
                    vals[base + "marginal"][k] += y_p.sum(0)
    return vals, total, len(items), perf_counter() - start


def log_ratio(v, m, smooth, edge):
    marginal = (m + 1.) / (m + 1.).sum(axis=-1, keepdims=True)
    eta = v + smooth * marginal[:, None, :]
    q = eta / eta.sum(axis=-1, keepdims=True)
    shrink = (v.sum(axis=-1, keepdims=True) / (v.sum(axis=-1, keepdims=True) + smooth)) ** .35
    if edge:
        def pmi(x):
            x = x.reshape(x.shape[:-1] + (7, 7))
            unit = np.log(x) - np.log(x.sum(-1, keepdims=True)) - np.log(x.sum(-2, keepdims=True))
            return unit.reshape(unit.shape[:-2] + (49,))
        log = pmi(q) - pmi(marginal[:, None, :])
    else:
        log = np.log(q) - np.log(marginal[:, None, :])
        log -= log.mean(-1, keepdims=True)
    return (np.clip(log, -2.5, 2.5) * shrink).astype(np.float32)


def weights(stats):
    tensors = {}
    for kind in FORMS:
        for mode in ("real", "shuffled"):
            base = f"{mode}.{kind}."
            x = log_ratio(stats[base + "present"], stats[base + "marginal"], 90. if kind == "edge" else 45., kind == "edge")
            if mode == "shuffled":
                x *= np.linalg.norm(tensors[kind + ".real"]) / max(np.linalg.norm(x), .000001)
            tensors[kind + "." + mode] = np.ascontiguousarray(x)
    return tensors


def save_moments(path, **data):
    tmp = path.with_suffix(".part")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **data)
    tmp.replace(path)


def checkpointed(job):
    folder, revision, rows = job
    keys = [x[0] for x in rows]
    path = folder / (sha256(("\0".join(keys)).encode()).hexdigest() + ".npz")
    if path.is_file():
        with np.load(path, allow_pickle=False) as v:
            assert v["identity"].item() == revision and v["keys"].tolist() == keys
            return {k: v[k] for k in zero()}, v["support"], len(keys), float(v["seconds"])
    result, support, docs, elapsed = aggregate(rows)
    save_moments(path, identity=revision, keys=keys, support=support, seconds=elapsed, **result)
    return result, support, docs, elapsed


def build(workers, count):
    if digest(SOURCE) != PINNED:
        raise ValueError("Official train CSV checksum does not match the prior experiment")
    start = perf_counter()
    total, n = np.zeros(7, dtype=np.int64), 0
    s = zero()
    worker_time = 0.
    assert 0 < count <= TRAIN_COUNTS[1]
    chosen = sorted(map(str, range(TRAIN_COUNTS[1])), key=lambda k: sha256((SAMPLE_SEED + k).encode()).digest())[:count]
    selection = set(chosen)
    revision = sha256((PINNED + digest(Path(__file__)) + digest(HERE / "cues.py") +
                        digest(cues.ATTRIBUTION / "signals.py") + "|".join(chosen)).encode()).hexdigest()
    folder = HERE / ("blocks-" + str(count))
    folder.mkdir(exist_ok=True)
    groups = ((folder, revision, q) for q in blocks((row for row in cases("train") if row[0] in selection)))
    with ProcessPoolExecutor(workers) as pool:
        for part, support, docs, dt in pool.map(checkpointed, groups):
            for key, v in part.items():
                s[key] += v
            worker_time += dt
            total += support
            n += docs
            if n % 384 == 0 or n == count:
                print("Discourse train", n, int(total.sum()), round(perf_counter() - start, 1), flush=True)
    assert n == count
    if count == TRAIN_COUNTS[1]:
        assert int(total.sum()) == TRAIN_COUNTS[0]
    path = HERE / "association.safetensors"
    saved = weights(s)
    stage = path.with_suffix(".part")
    save_file(saved, str(stage))
    stage.replace(path)
    save_moments(HERE / "sufficient-statistics.npz", **s)
    write(HERE / "training.json", {
        "train_csv_sha256": PINNED,
        "train_judgments": n, "train_sentences": int(total.sum()), "role_counts": total.tolist(),
        "selected_training_keys": chosen, "sampling": "Take a fixed hash-order sample of official training judgment IDs.",
        "sampling_seed": SAMPLE_SEED,
        "recovery": "An earlier full-split process reached 3,840 judgments before interruption. This run is checkpointed by case block.",
        "weights_sha256": digest(path), "moments_sha256": digest(HERE / "sufficient-statistics.npz"),
        "feature_definitions_sha256": digest(HERE / "cues.py"),
        "source_attribution_sha256": digest(cues.ATTRIBUTION / "signals.py"),
        "feature_names": FEATURES, "phase_bounds": cues.PHASE_CUTS,
        "trace": {
            "training": "Occurrence-weighted role label contrasts relative to per-phase marginal role frequency.",
            "transitions": "Per-feature adjacent-label pointwise mutual information relative to the corresponding phase's overall transitions.",
            "control": "Circular rotate labels inside training judgments and match parameter norm. This retains judgment level counts and nearly all neighboring role pairs.",
            "case_normalization": "Each cue decays with within-judgment cue frequency. Large judgments are also downweighted in accumulation.",
        },
        "wall_seconds": perf_counter() - start, "worker_seconds": worker_time, "workers": workers,
        "gradient_steps": 0, "device": "CPU", "official_test_accessed": False,
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", default=4, type=int)
    ap.add_argument("--judgments", default=SAMPLE_COUNT, type=int)
    args = ap.parse_args()
    build(args.workers, args.judgments)
