"""Write closed-form case-aware calibrations for locked document role scores."""

import argparse
from dataclasses import dataclass
from hashlib import file_digest, sha256
from io import BytesIO
from itertools import product
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from safetensors.numpy import load_file, save_file
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit, softmax

HERE = Path(__file__).resolve().parent
CONT = HERE.parent
LEGAL = CONT / "legal-context"
MAIN_SOURCE = LEGAL / "legal-results.zip"
GROUPS = ("fit", "select", "confirm", "fresh")
PLAN = HERE / "plan.json"
FROZEN = HERE / "freeze.json"
ROLES = ("None", "Facts", "Issue", "Arguments of Petitioner", "Arguments of Respondent", "Reasoning", "Decision")
NONE, REASONING = 0, 5
PAIR = np.asarray([NONE, REASONING])
EYE = np.eye(7, dtype=np.float32)

ARMS = {
    "bias": ("bias", "all"),
    "score-affine": ("score", "all"),
    "case-affine": ("case", "all"),
    "local-experts": ("experts", "all"),
    "case-experts": ("case-experts", "all"),
    "pair-affine": ("score", "pair"),
    "pair-case": ("case-experts", "pair"),
}
RARITY = (0., .35, .65)
RIDGES = (.025, .16)
GAINS = (.08, .22, .45, .75)
GUARD = .004
SEED = 285761


@dataclass
class Bank:
    group: str
    keys: np.ndarray
    ends: np.ndarray
    gold: np.ndarray
    scores: np.ndarray
    experts: np.ndarray
    old: np.ndarray
    novel: np.ndarray
    signature: str

    @property
    def base(self):
        return self.scores.argmax(1).astype(np.int8)


def digest(path):
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    stage = path.with_suffix(".tmp")
    stage.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    stage.replace(path)


def preregister():
    with ZipFile(MAIN_SOURCE) as z:
        doc_lock = sha256(z.read("freeze.json")).hexdigest()
        assert json.loads(z.read("freeze.json"))["selected"] == "doc-semantic"
    plan = {
        "parent_frozen_sha256": doc_lock,
        "validation_parent_archive_sha256": digest(MAIN_SOURCE),
        "training": "192 official train cases, exact frozen doc-semantic scores; upstream trained on these cases. No new end-to-end training.",
        "selection": "48 previously used selector judgments. Rank by the mean of full-set and no-exact-key macro-F1, then combined pair error change and accuracy. Require that pair errors do not grow on either set and overall accuracy stays within 0.4 points in selection. Keep zero edit when it is stronger.",
        "confirmation": "Replay frozen weights on 72 and 96 validation judgments. Earlier project evaluation exposure is possible. Report only label-derived outcomes on the two already-declared sets. Confidence intervals resample whole judgments.",
        "targets": {
            "all": "Class-balanced difference between one-hot human roles and temperature-calibrated frozen scores.",
            "pair": "Signed None–Reasoning preference residual, emphasizing incumbent mistakes within this pair and protecting the remaining positions. At inference only currents already in the pair can switch.",
        },
        "model_families": ARMS, "ridges": RIDGES, "gain": GAINS, "rarity": RARITY,
        "controls": "Static bias and score-only heads, current donor-expert evidence without case features, length-binned shuffled case summaries matched to learned weight norm, shuffled gold targets matched to the learned head norm.",
        "novel": "Published train exact key unavailable at 1, 3, 5 and 9 sentence widths.",
        "accuracy_selection_tolerance": GUARD, "random_seed": SEED, "gradient_steps": 0,
        "device": "CPU", "official_labels_opened_for_validation_experiments": False,
    }
    if PLAN.exists():
        assert read(PLAN) == json.loads(json.dumps(plan))
    else:
        write(PLAN, plan)
    return plan


def payload(path, group):
    name = f"scorebank-{group}.npz"
    if path.is_dir():
        source = (path / name).read_bytes()
    else:
        with ZipFile(path) as z:
            source = z.read(name)
            manifest = json.loads(z.read("manifest.json"))
            assert sha256(source).hexdigest() == manifest[name]
    return source


def load_bank(path, group):
    assert group in GROUPS
    raw = payload(path, group)
    with np.load(BytesIO(raw), allow_pickle=False) as a:
        keys, ends, y, old = (a[k] for k in ("case_ids", "offsets", "gold", "old"))
        score, expert = a["score"].astype(np.float32), a["experts"].astype(np.float32)
    assert len(set(keys.tolist())) == len(keys)
    assert ends.shape == (len(keys) + 1,) and ends[-1] == len(y) and np.all(np.diff(ends) > 0)
    assert score.shape == (len(y), 7) and expert.shape == (len(y), 44)
    assert old.shape == (len(y),) and np.isfinite(score).all() and np.isfinite(expert).all()
    assert np.isin(y, np.arange(7)).all() and np.isin(old, np.arange(7)).all()
    novel = np.ones(len(y), dtype=bool)
    if group != "fit":
        with ZipFile(MAIN_SOURCE) as package:
            payload_name = group + "-predictions.npz"
            source = package.read(payload_name)
            assert sha256(source).hexdigest() == json.loads(package.read("SHA256SUMS.json"))[payload_name]
        with np.load(BytesIO(source), allow_pickle=False) as a:
            for expected, actual in (("case_ids", keys), ("offsets", ends), ("gold", y), ("incumbent", old),
                                     ("doc-semantic", score.argmax(1))):
                np.testing.assert_array_equal(a[expected], actual)
        with np.load(CONT / "scale" / (group + "-mask.npz"), allow_pickle=False) as p:
            np.testing.assert_array_equal(p["case_ids"], keys)
            np.testing.assert_array_equal(p["offsets"], ends)
            novel = p["novel"]
    return Bank(group, keys, ends, y, score, expert, old, novel, sha256(raw).hexdigest())


def baseline_scale(b):
    # Anchor temperature and score scale to the fitting judgments only.
    sort = np.sort(b.scores, axis=1)
    return float(np.clip(np.median(sort[:, -1] - sort[:, -2]), .045, .4))


def probabilities(scores, temp):
    return softmax(scores / temp, axis=1).astype(np.float32)


def margins(s):
    n = len(s)
    largest = s.argmax(1)
    sorted_ = np.sort(s, axis=1)
    gap = sorted_[:, -1] - sorted_[:, -2]
    i = np.arange(n)
    return np.column_stack((s[i, largest], gap, s[:, REASONING] - s[:, NONE])).astype(np.float32)


def expert_inputs(b):
    parts = []
    for a in (0, 22):
        parts.extend((probabilities(b.experts[:, a:a + 7], 1. / .65),
                      probabilities(b.experts[:, a + 7:a + 14], 1. / .8),
                      np.clip(b.experts[:, a + 14:a + 22], -4., 4.)))
    return np.column_stack(parts)


def summary(p, base, s):
    n = len(p)
    a = max(1, n // 4)
    soft = np.mean(p, 0, dtype=np.float32)
    return np.column_stack((
        soft[None, :], EYE[base].mean(0, keepdims=True),
        (p[:a].mean(0) - soft)[None, :], (p[-a:].mean(0) - soft)[None, :],
        np.asarray([[np.log1p(n), margins(s)[:, 1].mean()]], dtype=np.float32),
    ))[0]


def neighbor_means(p, width):
    n = len(p)
    i = np.arange(n)
    pre = np.vstack((np.zeros(p.shape[1], np.float64), np.cumsum(p, axis=0, dtype=np.float64)))
    left = np.maximum(i - width, 0)
    right = np.minimum(i + 1 + width, n)
    return ((pre[i] - pre[left]) / np.maximum(i - left, 1)[:, None],
            (pre[right] - pre[i + 1]) / np.maximum(right - i - 1, 1)[:, None])


def design(b, temp, kind, *, shuffled_cases=False):
    if kind == "bias":
        return np.zeros((len(b.gold), 0), dtype=np.float32)
    p = probabilities(b.scores, temp)
    basis = np.column_stack((p, b.scores - b.scores.mean(1, keepdims=True), EYE[b.base], margins(b.scores)))
    if kind == "score":
        return basis
    expert = expert_inputs(b) if "experts" in kind else None
    if kind == "experts":
        return np.column_stack((basis, expert))
    summaries = np.stack([summary(p[a:z], b.base[a:z], b.scores[a:z]) for a, z in zip(b.ends, b.ends[1:])])
    if shuffled_cases:
        # Rotation within judgment-length bands preserves the observed length mix.
        length_order = np.argsort(np.diff(b.ends), kind="stable")
        rng = np.random.default_rng(SEED)
        for batch in np.array_split(length_order, max(1, len(length_order) // 8)):
            order = rng.permutation(batch)
            summaries[order] = summaries[np.roll(order, 1)]
    out = []
    for j, (a, z) in enumerate(zip(b.ends, b.ends[1:])):
        n = z - a
        local = p[a:z]
        positions = ((np.arange(n, dtype=np.float32) + .5) / n)[:, None]
        signal = summaries[j]
        history = neighbor_means(local, 4) + neighbor_means(local, 16)
        vals = [basis[a:z], np.broadcast_to(signal, (n, len(signal))), local * signal[:7],
                (local - signal[:7]), (positions - .5) * signal[:7], positions, positions ** 2, *history]
        if expert is not None:
            vals.append(expert[a:z])
            parts = expert[a:z, [0, 5, 7, 12, 22, 27, 29, 34]]
            vals.append(parts * (local[:, REASONING] - local[:, NONE])[:, None])
        out.append(np.column_stack(vals).astype(np.float32))
    return np.concatenate(out)


def normalizer(x):
    return x.mean(0, dtype=np.float64).astype(np.float32), (
        1. / np.maximum(x.std(0, dtype=np.float64), .04)).astype(np.float32)


def input_features(x, center, scale):
    return np.column_stack((np.clip((x - center) * scale, -5, 5), np.ones(len(x), dtype=np.float32)))


def targets(b, temp, scope, shuffle=False):
    y = b.gold if not shuffle else b.gold[np.random.default_rng(SEED + 1).permutation(len(b.gold))]
    if scope == "all":
        return (EYE[y] - probabilities(b.scores, temp)), y
    pair = np.isin(y, PAIR)
    old_margin = (b.scores[:, REASONING] - b.scores[:, NONE]) / temp
    desired = np.where(y == REASONING, 1., -1.)
    target = np.where(pair & np.isin(b.base, PAIR), desired - (2 * expit(old_margin) - 1), 0)
    return target[:, None].astype(np.float32), y


def fit_equations(x, target, y, b, rarity, scope):
    n = np.repeat(np.diff(b.ends), np.diff(b.ends))
    role_mass = np.bincount(y, minlength=7).clip(min=1) / len(y)
    mass = (role_mass[y] ** -rarity) * (n / np.median(n)) ** -.2
    if scope == "pair":
        pair = np.isin(y, PAIR) & np.isin(b.base, PAIR)
        mass *= np.where(pair & (b.base != y), 2., np.where(pair, 1., .7))
    mass = (mass / mass.mean()).astype(np.float32)
    wx = x * mass[:, None]
    gram = (x.T @ wx).astype(np.float64)
    return .5 * (gram + gram.T), (wx.T @ target).astype(np.float64)


def solve(gram, target, ridge):
    shape = len(gram)
    x = gram.copy()
    lam = ridge * gram.diagonal().mean()
    x.flat[::shape + 1] += lam
    x[-1, -1] -= .95 * lam
    return cho_solve(cho_factor(x, check_finite=False), target, check_finite=False).astype(np.float32)


def decide(b, delta, temp, gain, scope):
    old = b.base
    if scope == "all":
        return (b.scores + (gain * temp) * delta).argmax(1).astype(np.int8)
    margin = b.scores[:, REASONING] - b.scores[:, NONE] + (gain * temp) * delta[:, 0]
    return np.where(np.isin(old, PAIR), np.where(margin > 0, REASONING, NONE), old).astype(np.int8)


def matrix(y, pred):
    return np.bincount(y.astype(np.int64) * 7 + pred, minlength=49).reshape(7, 7)


def measure(m):
    hits = np.diagonal(m, axis1=-2, axis2=-1)
    role = (2 * hits) / (m.sum(-2) + m.sum(-1)).clip(min=1)
    return role.mean(-1), hits.sum(-1) / m.sum((-1, -2)).clip(min=1), role


def score(y, prediction):
    m = matrix(y, prediction)
    f1, accuracy, role = measure(m)
    return {"macro_f1": float(f1), "accuracy": float(accuracy), "per_role_f1": dict(zip(ROLES, role.tolist())),
            "none_to_reasoning": int(m[NONE, REASONING]), "reasoning_to_none": int(m[REASONING, NONE]),
            "pair_errors": int(m[NONE, REASONING] + m[REASONING, NONE]), "confusion": m.tolist()}


def stats(b, p):
    return {"all": score(b.gold, p), "novel": score(b.gold[b.novel], p[b.novel])}


def choice_rank(value):
    full, novel = value["all"], value["novel"]
    return (.5 * (full["macro_f1"] + novel["macro_f1"]), -full["pair_errors"] - novel["pair_errors"], full["accuracy"])


def rate_allowed(trial, base):
    return (trial["all"]["accuracy"] >= base["all"]["accuracy"] - GUARD and
            trial["all"]["pair_errors"] <= base["all"]["pair_errors"] and
            trial["novel"]["pair_errors"] <= base["novel"]["pair_errors"])


def save_weights(name, p):
    path = HERE / (name + ".safetensors")
    tmp = path.with_suffix(".part")
    save_file({key: np.ascontiguousarray(v) for key, v in p.items()}, str(tmp))
    tmp.replace(path)
    return digest(path)


def learn_one(train, select, temp, name, kind, scope):
    a = design(train, temp, kind)
    center, scale = normalizer(a)
    fx = input_features(a, center, scale)
    sx = input_features(design(select, temp, kind), center, scale)
    del a
    tgt, y = targets(train, temp, scope)
    base = stats(select, select.base)
    all_variants = []
    keep = None
    for rarity, ridge in product(RARITY, RIDGES):
        g, t = fit_equations(fx, tgt, y, train, rarity, scope)
        coef = solve(g, t, ridge)
        correction = sx @ coef
        for gain in GAINS:
            p = decide(select, correction, temp, gain, scope)
            item = stats(select, p)
            opt = (rarity, ridge, gain)
            all_variants.append({"parameters": opt, "macro_f1": item["all"]["macro_f1"],
                                 "novel_macro_f1": item["novel"]["macro_f1"], "pair": item["all"]["pair_errors"],
                                 "novel_pair": item["novel"]["pair_errors"], "allowed": rate_allowed(item, base)})
            if keep is None or (rate_allowed(item, base), choice_rank(item)) > keep[0]:
                keep = (rate_allowed(item, base), choice_rank(item)), opt, coef, item
    _, opt, coef, results = keep
    p = {"center": center, "scale": scale, "output.weight": coef[:-1], "output.bias": coef[-1]}
    settings = {"kind": kind, "scope": scope, "params": opt, "weights_sha256": save_weights(name, p),
                "selection": results, "identity_higher": choice_rank(base) >= choice_rank(results),
                "eligible": rate_allowed(results, base), "grid": all_variants}
    # Keep same physical architecture and output norm for two destructive controls.
    if name in ("case-affine", "case-experts", "pair-case"):
        rarity, ridge, gain = opt
        variants = {
            "shuffled-labels": (fx, *targets(train, temp, scope, shuffle=True)),
            "shuffled-cases": (input_features(design(train, temp, kind, shuffled_cases=True), center, scale), tgt, y),
        }
        for suffix, (dx, value, roles) in variants.items():
            g, t = fit_equations(dx, value, roles, train, rarity, scope)
            sham = solve(g, t, ridge)
            sham *= np.linalg.norm(coef) / max(1e-8, np.linalg.norm(sham))
            pp = {"center": center, "scale": scale, "output.weight": sham[:-1], "output.bias": sham[-1]}
            control = name + "-" + suffix
            settings.setdefault("controls", {})[control] = save_weights(control, pp)
    print(name, opt, f'{results["all"]["macro_f1"] * 100:.3f}', f'{results["novel"]["macro_f1"] * 100:.3f}',
          results["all"]["pair_errors"], settings["eligible"], flush=True)
    return settings


def learn(archive):
    plan = preregister()
    train, select = (load_bank(archive, g) for g in ("fit", "select"))
    assert not set(train.keys.tolist()) & set(select.keys.tolist())
    temp = baseline_scale(train)
    arms = {name: learn_one(train, select, temp, name, *cfg) for name, cfg in ARMS.items()}
    advance = [name for name, v in arms.items() if v["eligible"] and not v["identity_higher"]]
    name = max(advance, key=lambda k: choice_rank(arms[k]["selection"])) if advance else "base"
    frozen = {"protocol_sha256": digest(PLAN), "temperature": temp, "arms": arms, "selected": name,
              "fit_score_bank_sha256": train.signature, "select_score_bank_sha256": select.signature,
              "fit_cases": train.keys.tolist(), "select_cases": select.keys.tolist(),
              "fit_rows": len(train.gold), "selector_rows": len(select.gold), "selector_novel_rows": int(select.novel.sum()),
              "selector_baseline": stats(select, select.base), "configuration": plan["selection"], "official_test_opened": False}
    write(FROZEN, frozen)
    print("frozen", name, flush=True)


def interval(b, old, new, keep):
    a, t = [], []
    for lo, hi in zip(b.ends, b.ends[1:]):
        g, filter_ = b.gold[lo:hi], keep[lo:hi]
        a.append(matrix(g[filter_], old[lo:hi][filter_]))
        t.append(matrix(g[filter_], new[lo:hi][filter_]))
    a, t = np.asarray(a), np.asarray(t)
    picks = np.random.default_rng(SEED).integers(len(a), size=(4000, len(a)))
    af, aa, _ = measure(a[picks].sum(1))
    tf, ta, _ = measure(t[picks].sum(1))
    n2r = (t[:, NONE, REASONING] - a[:, NONE, REASONING])[picks].sum(1)
    r2n = (t[:, REASONING, NONE] - a[:, REASONING, NONE])[picks].sum(1)
    return {name: np.quantile(value, [.025, .975]).tolist()
            for name, value in zip(("macro_f1", "accuracy", "none_to_reasoning", "reasoning_to_none"),
                                   (tf - af, ta - aa, n2r, r2n))}


def evaluate(archive):
    proof = read(FROZEN)
    assert proof["protocol_sha256"] == digest(PLAN)
    temp = proof["temperature"]
    cols = {}
    for group in ("select", "confirm", "fresh"):
        b = load_bank(archive, group)
        if group == "select":
            assert b.signature == proof["select_score_bank_sha256"]
        else:
            fitted = set(proof["fit_cases"]) | set(proof["select_cases"])
            assert not fitted & set(b.keys.tolist())
        pred = {"base": b.base}
        tests = {}
        x = {}
        for name, opt in proof["arms"].items():
            kind, scope = opt["kind"], opt["scope"]
            if kind not in x:
                x[kind] = design(b, temp, kind)
            for tag, sh in ((name, opt["weights_sha256"]), *opt.get("controls", {}).items()):
                path = HERE / (tag + ".safetensors")
                assert digest(path) == sh
                p = load_file(str(path))
                source = input_features(x[kind], p["center"], p["scale"])
                delta = source[:, :-1] @ p["output.weight"] + p["output.bias"]
                pred[tag] = decide(b, delta, temp, opt["params"][-1], scope)
                if group == "select" and tag == name:
                    assert stats(b, pred[tag])["all"]["confusion"] == opt["selection"]["all"]["confusion"]
        for name, p in pred.items():
            item = stats(b, p)
            item["paired_all"] = interval(b, pred["base"], p, np.ones(len(p), bool))
            item["paired_novel"] = interval(b, pred["base"], p, b.novel)
            item["changes"] = int(np.count_nonzero(p != pred["base"]))
            item["fixes"] = int(np.count_nonzero((p == b.gold) & (pred["base"] != b.gold)))
            item["breaks"] = int(np.count_nonzero((p != b.gold) & (pred["base"] == b.gold)))
            tests[name] = item
        cols[group] = {"cases": len(b.keys), "rows": len(b.gold), "novel_rows": int(b.novel.sum()),
                       "bank_sha256": b.signature, "arms": tests}
        np.savez_compressed(HERE / (group + "-predictions.npz"), case_ids=b.keys, offsets=b.ends, gold=b.gold,
                            novel=b.novel, **pred)
        print(group, {key: (round(z["all"]["macro_f1"] * 100, 3), round(z["novel"]["macro_f1"] * 100, 3),
                             z["all"]["pair_errors"]) for key, z in tests.items()}, flush=True)
    write(HERE / "results.json", {"freeze_sha256": digest(FROZEN), "selection": proof["selected"], "groups": cols,
                                   "official_test_accessed": False})


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=("plan", "fit", "score"))
    cli.add_argument("--archive", type=Path, default=LEGAL / "calibration-banks.zip")
    args = cli.parse_args()
    if args.mode == "plan":
        preregister()
    elif args.mode == "fit":
        learn(args.archive)
    else:
        evaluate(args.archive)


if __name__ == "__main__":
    main()
