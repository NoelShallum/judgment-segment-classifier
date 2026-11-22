"""Apply frozen discourse emissions and transitions to the public document classifier."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import ctypes
from itertools import product
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
from safetensors.numpy import load_file, save_file

import associations as a
import cues

HERE = Path(__file__).resolve().parent
CONT = HERE.parent
sys.path.insert(0, str(CONT / "doc-calibration"))
import calibrate as base

GOLD_PLAN = HERE / "protocol-recovery.json"
FREEZE = HERE / "freeze.json"
ARCHIVE = CONT / "legal-context/calibration-banks.zip"
WEIGHTS = HERE / "association.safetensors"
CODEC = a.ROOT / "packed_role_decode.cpp"
COMPILER = HERE / "decode.so"
RIDGES = (.06, .28)
UNARY = (.12, .30, .60, 1.10)
EDGES = (.04, .12, .26, .55)
SCOPES = ("all", "pair")
ROLE_MODES = ("local-association", "context-association", "local-residual", "context-residual", "edge", "joint")


def register():
    p = {
        "input_scores": "Frozen doc-semantic scores exported on the existing train and validation case banks.",
        "base_config_sha256": a.digest(base.MAIN_SOURCE),
        "association_scope": "A fixed 1,536-judgment hash sample from public official train. The original full scan was interrupted before selection.",
        "residual_scope": "192 official training judgments already fitted by the underlying score provider.",
        "selection": "48 selection judgments. Maximize the mean of full-judgment and no-exact-key macro-F1, then reduce pair errors. Keep close to baseline accuracy and avoid higher pair errors on either subset. Freeze each arm before confirmation.",
        "confirmation": "72 and 96 complete validation judgments, plus no-exact-training-key slices. Prior project validation exposure applies.",
        "controls": "Identity, paired local-only variants, and amplitude-matched label rotations within training judgments.",
        "cues": cues.feature_names(), "positions": cues.PHASE_CUTS,
        "association_strength": UNARY, "transition_strength": EDGES, "residual_ridge": RIDGES,
        "topics": ROLE_MODES, "scopes": SCOPES, "train_gradient_steps": 0,
        "decode_source_sha256": a.digest(CODEC), "device": "CPU", "official_test_accessed": False,
    }
    if GOLD_PLAN.exists():
        import json
        assert base.read(GOLD_PLAN) == json.loads(json.dumps(p))
    else:
        a.write(GOLD_PLAN, p)


def capture_one(item):
    item = item[0], item[1], item[2], cues.extract(item[1])
    return item[0], item[2], item[3]


def gather(bank):
    name = bank.group
    dest = HERE / (name + "-cues.npz")
    attr = a.digest(cues.ATTRIBUTION / "signals.py")
    fs = a.digest(HERE / "cues.py")
    path = a.CSV / ("train.csv" if name == "fit" else "val.csv")
    source = a.digest(path)
    signature = [bank.signature, attr, fs, source]
    start = perf_counter()
    if dest.is_file():
        with np.load(dest, allow_pickle=False) as f:
            assert f["signature"].tolist() == signature
            return cues.Features(*(f[k] for k in ("local", "context", "edge", "phase")))
    wanted = set(bank.keys.tolist())
    with ProcessPoolExecutor(3) as pool:
        data = {k: (y, seq) for k, y, seq in pool.map(
            capture_one, (row for row in a.cases("train" if name == "fit" else "val") if row[0] in wanted),
            chunksize=16)}
    items = [data[k] for k in bank.keys]
    np.testing.assert_array_equal(np.concatenate([y for y, _ in items]), bank.gold)
    np.testing.assert_array_equal(np.diff(bank.ends), [len(y) for y, _ in items])
    payload = {k: np.concatenate([getattr(seq, k) for _, seq in items]) for k in ("local", "context", "edge", "phase")}
    with dest.with_suffix(".part").open("wb") as f:
        np.savez_compressed(f, signature=signature, **payload)
    dest.with_suffix(".part").replace(dest)
    print("Captured cues", bank.group, len(data), len(bank.gold), round(perf_counter()-start, 2), flush=True)
    return cues.Features(**payload)


def decode_loader():
    if not COMPILER.exists():
        tmp = COMPILER.with_suffix(".part")
        subprocess.run(["g++", "-shared", "-fPIC", "-std=c++17", "-O3", "-ffp-contract=off", str(CODEC),
                        "-o", str(tmp)], check=True)
        tmp.replace(COMPILER)
    p = np.ctypeslib.ndpointer
    func = ctypes.CDLL(str(COMPILER)).decode_judgments
    func.argtypes = (p(np.float64, flags="C_CONTIGUOUS"), p(np.float32, flags="C_CONTIGUOUS"),
                     p(np.int64, flags="C_CONTIGUOUS"), ctypes.c_size_t,
                     p(np.float32, flags="C_CONTIGUOUS"), p(np.float32, flags="C_CONTIGUOUS"),
                     p(np.int8, flags="C_CONTIGUOUS"))
    func.restype = None

    def decode(emission, edge, offsets):
        e = np.ascontiguousarray(emission, dtype=np.float64)
        z = np.ascontiguousarray(edge, dtype=np.float32)
        off = np.ascontiguousarray(offsets, dtype=np.int64)
        p = np.empty(len(e), dtype=np.int8)
        func(e, z, off, len(off) - 1, np.zeros((10, 10), dtype=np.float32), np.ones(49, np.float32), p)
        return p

    rng = np.random.default_rng(21)
    e = rng.normal(size=(3, 7))
    z = rng.normal(size=(3, 7, 7)).astype(np.float32)
    paths = np.asarray(list(product(range(7), repeat=3)))
    subtotal = e[np.arange(3), paths].sum(1) + z[np.arange(1, 3), paths[:, :-1], paths[:, 1:]].sum(1, dtype=np.float64)
    np.testing.assert_array_equal(decode(e, z.reshape(3, 49), [0, 3]), paths[subtotal.argmax()])
    return decode


def association(f, w, kind, profile):
    x = getattr(f, kind)
    output = np.zeros((len(x), w[kind + "." + profile].shape[-1]), np.float32)
    for i in range(cues.PHASES):
        rows = f.phase == i
        seen = x[rows]
        response = seen @ w[kind + "." + profile][i]
        output[rows] = response / np.sqrt(np.maximum(seen.sum(1, keepdims=True), 1.))
    return output


def residual_input(b, f, temperature, w, kind, profile):
    memory = association(f, w, kind, profile)
    cue = getattr(f, kind)
    probs = base.probabilities(b.scores, temperature)
    margin = b.scores[:, base.REASONING] - b.scores[:, base.NONE]
    delta = np.clip(margin / max(temperature, 1e-5), -4, 4)
    strength = np.clip(cue.sum(1), 0, 1)
    phase = np.eye(cues.PHASES, dtype=np.float32)[f.phase]
    return np.column_stack((cue, memory, cue * delta[:, None], memory * probs,
                            strength[:, None] * phase * (probs[:, 5:6] - probs[:, 0:1]))).astype(np.float32)


def rotation(b):
    y = b.gold.copy()
    for key, start, end in zip(b.keys, b.ends, b.ends[1:]):
        n = end - start
        if n > 1:
            k = int.from_bytes(a.sha256(("discourse-residual-control|" + key).encode()).digest()[:4], "big")
            y[start:end] = np.roll(y[start:end], 1 + k % (n - 1))
    return y


def residual_weights(b, f, w, temp, *, kind, mode, ridge):
    raw = residual_input(b, f, temp, w, kind, mode)
    scale = 1 / np.maximum(.06, raw.std(0, dtype=np.float64)).astype(np.float32)
    x = np.clip(raw * scale, -5, 5)
    chosen = b.gold if mode == "real" else rotation(b)
    freq = np.bincount(chosen, minlength=7).clip(min=1) / len(chosen)
    doclen = np.repeat(np.diff(b.ends), np.diff(b.ends))
    row_mass = (freq[chosen] ** -.2) * (250 / doclen) ** .23
    row_mass = (row_mass / row_mass.mean()).astype(np.float32)
    expected = base.probabilities(b.scores, temp)
    target = a.EYE[chosen] - expected
    importance = np.where(b.base == chosen, 1.0, 1.5).astype(np.float32)
    wx = x * (row_mass * importance)[:, None]
    gram = (x.T @ wx).astype(np.float64)
    rhs = (wx.T @ target).astype(np.float64)
    pen = ridge * gram.diagonal().mean()
    g = (gram + gram.T) * .5 + pen * np.eye(len(gram))
    beta = np.linalg.solve(g, rhs)
    return {"scale": scale, "weight": beta.astype(np.float32)}


def residual_response(b, f, w, temp, p, kind, mode):
    x = residual_input(b, f, temp, w, kind, mode)
    out = np.clip(x * p["scale"], -5, 5) @ p["weight"]
    return np.clip(out, -.8, .8)


def fit_residual(w, t, training, temp):
    output = {}
    for kind, ridge in product(("local", "context"), RIDGES):
        real, sham = [residual_weights(t, training, w, temp, kind=kind, mode=mode, ridge=ridge)
                      for mode in ("real", "shuffled")]
        norm = np.linalg.norm(sham["weight"])
        sham["weight"] *= np.linalg.norm(real["weight"]) / max(norm, 1.e-6)
        for mode, p in (("real", real), ("shuffled", sham)):
            output.update({f"{kind}/{ridge:g}/{mode}.{k}": v for k, v in p.items()})
    save_file({k: np.ascontiguousarray(v) for k, v in output.items()}, str(HERE / "residual.safetensors"))
    return output


def streams(b, f, w, residual, temp, phase):
    keys = ("real", "shuffled") if phase == "score" else ("real",)
    out = {}
    for mode in keys:
        for kind in a.FORMS:
            out[kind + "-association", mode, 0.] = association(f, w, kind, mode)
        for kind, r in product(("local", "context"), RIDGES):
            p = {k: residual[f"{kind}/{r:g}/{mode}.{k}"] for k in ("scale", "weight")}
            out[kind + "-residual", mode, r] = residual_response(b, f, w, temp, p, kind, mode)
    return out


def predict(b, values, temp, opt, engine, mode="real"):
    name, ridge, u, e, scope = opt
    output = b.scores.copy()
    if name not in ("edge", "zero"):
        kind = "context-residual" if name == "joint" else name
        z = values[kind, mode, ridge]
        output += (u * temp) * z
    if scope == "pair":
        pair = np.isin(b.base, base.PAIR)
        other = np.arange(7)[None, :] == b.base[:, None]
        supported = np.isin(np.arange(7), base.PAIR)[None, :]
        output[~(other | (pair[:, None] & supported))] = -10000.
    if e:
        return engine(output, (e * temp) * values["edge-association", mode, 0.], b.ends)
    return output.argmax(1).astype(np.int8)


def read_inputs(require_residual=True):
    train = base.read(HERE / "training.json")
    assert train["weights_sha256"] == a.digest(WEIGHTS)
    assert train["feature_definitions_sha256"] == a.digest(HERE / "cues.py")
    assert train["source_attribution_sha256"] == a.digest(cues.ATTRIBUTION / "signals.py")
    w = load_file(WEIGHTS)
    r = load_file(HERE / "residual.safetensors") if require_residual else None
    return w, r


def fit():
    t0 = perf_counter()
    register()
    train, select = (base.load_bank(ARCHIVE, g) for g in ("fit", "select"))
    assert not set(train.keys.tolist()) & set(select.keys.tolist())
    left, right = gather(train), gather(select)
    w, _ = read_inputs(False)
    temp = base.baseline_scale(train)
    residual = fit_residual(w, train, left, temp)
    engine = decode_loader()
    v = streams(select, right, w, residual, temp, "tune")
    baseline = base.stats(select, select.base)
    zeros = engine(select.scores, np.zeros((len(select.gold), 49), np.float32), select.ends)
    np.testing.assert_array_equal(zeros, select.base)

    best, sweeps = {}, {}
    for name in ROLE_MODES:
        search = []
        if name == "edge":
            grid = product((name,), (0.,), (0.,), EDGES, SCOPES)
        elif name == "joint":
            grid = product((name,), RIDGES, (.30, .60), (.08, .20), SCOPES)
        else:
            grid = product((name,), RIDGES if name.endswith("-residual") else (0.,), UNARY, (0.,), SCOPES)
        for cfg in grid:
            pred = predict(select, v, temp, cfg, engine)
            score = base.stats(select, pred)
            search.append({"configuration": cfg, "eligible": base.rate_allowed(score, baseline),
                           "rank": base.choice_rank(score), "scores": score})
        pick = max(search, key=lambda x: (x["eligible"], x["rank"]))
        best[name] = pick
        sweeps[name] = [(x["configuration"], x["eligible"], x["rank"]) for x in search]
        print("Selected discourse", name, pick["configuration"], pick["rank"], pick["eligible"], flush=True)
    eligible = [k for k, v in best.items() if v["eligible"] and v["rank"] > base.choice_rank(baseline)]
    selected = max(eligible, key=lambda k: best[k]["rank"]) if eligible else "zero"
    a.write(FREEZE, {"protocol_sha256": a.digest(GOLD_PLAN), "association_sha256": a.digest(WEIGHTS),
                     "residual_sha256": a.digest(HERE / "residual.safetensors"), "decoder_sha256": a.digest(COMPILER),
                     "fit_bank_sha256": train.signature, "selector_bank_sha256": select.signature,
                     "fit_keys": train.keys.tolist(), "selector_keys": select.keys.tolist(),
                     "fit_rows": len(train.gold), "selector_rows": len(select.gold), "selector": baseline,
                     "settings": best, "all_candidates": sweeps, "selected": selected, "temperature": temp,
                     "wall_seconds": perf_counter() - t0, "official_test_opened": False})
    print("Frozen", selected, flush=True)


def span_boundary(y, pred, ends):
    first = np.zeros(len(y), dtype=bool)
    first[ends[:-1]] = 1
    g, p = (y != np.roll(y, 1)) & ~first, (pred != np.roll(pred, 1)) & ~first
    hit = int((g & p).sum())
    return {"f1": 2 * hit / max(1, int(g.sum() + p.sum())), "hits": hit,
            "labeled_boundaries": int(g.sum()), "predicted_boundaries": int(p.sum())}


def evaluate():
    t0 = perf_counter()
    p = base.read(FREEZE)
    assert p["protocol_sha256"] == a.digest(GOLD_PLAN) and p["association_sha256"] == a.digest(WEIGHTS)
    assert p["residual_sha256"] == a.digest(HERE / "residual.safetensors")
    w, r = read_inputs()
    engine = decode_loader()
    assert p["decoder_sha256"] == a.digest(COMPILER)
    temp = p["temperature"]
    all_scores = {}
    for name in ("select", "confirm", "fresh"):
        b = base.load_bank(ARCHIVE, name)
        if name == "select":
            assert p["selector_bank_sha256"] == b.signature
        else:
            assert not (set(p["fit_keys"]) | set(p["selector_keys"])) & set(b.keys.tolist())
        z = gather(b)
        start = perf_counter()
        s = streams(b, z, w, r, temp, "score")
        pred = {"zero": predict(b, s, temp, ("zero", 0., 0., 0., "all"), engine)}
        np.testing.assert_array_equal(pred["zero"], b.base)
        for key, t in p["settings"].items():
            for mode in ("real", "shuffled"):
                label = key if mode == "real" else key + "-shuffled"
                pred[label] = predict(b, s, temp, t["configuration"], engine, mode)
                if name == "select" and mode == "real":
                    assert base.stats(b, pred[label])["all"]["confusion"] == t["scores"]["all"]["confusion"]
        compute_seconds = perf_counter() - start
        quote = (z.local[:, 11] > .15) | (z.local[:, 12] > 0) | (z.local[:, 13] > 0)
        attributed = z.context[:, len(a.FEATURES["local"]):len(a.FEATURES["local"]) + 12].max(1) > .1
        scores = {}
        for key, y in pred.items():
            values = base.stats(b, y)
            values.update(paired_all=base.interval(b, b.base, y, np.ones(len(y), bool)),
                          paired_novel=base.interval(b, b.base, y, b.novel),
                          boundary=span_boundary(b.gold, y, b.ends),
                          quote=base.score(b.gold[quote], y[quote]),
                          attributed=base.score(b.gold[attributed], y[attributed]),
                          changed=int(np.count_nonzero(y != b.base)),
                          fixed=int(np.count_nonzero((y == b.gold) & (b.base != b.gold))),
                          broken=int(np.count_nonzero((y != b.gold) & (b.base == b.gold))))
            scores[key] = values
        all_scores[name] = {"cases": len(b.keys), "rows": len(b.gold), "no_train_key_rows": int(b.novel.sum()),
                            "quote_rows": int(quote.sum()), "attributed_rows": int(attributed.sum()), "scores": scores,
                            "inference_seconds_all_arms_reuse_cues": compute_seconds}
        np.savez_compressed(HERE / (name + "-predictions.npz"), case_ids=b.keys, offsets=b.ends, gold=b.gold,
                            novel=b.novel, **pred)
        print(name, {k: (round(v["all"]["macro_f1"] * 100, 3), round(v["novel"]["macro_f1"] * 100, 3),
                         v["all"]["pair_errors"]) for k, v in scores.items()}, flush=True)
    a.write(HERE / "results.json", {"freeze_sha256": a.digest(FREEZE), "selected": p["selected"], "cohorts": all_scores,
                                    "wall_seconds": perf_counter() - t0, "official_test_accessed": False})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task", choices=("plan", "cache", "fit", "score"))
    op = ap.parse_args().task
    if op == "plan":
        register()
    elif op == "cache":
        for group in ("fit", "select"):
            gather(base.load_bank(ARCHIVE, group))
    elif op == "fit":
        fit()
    else:
        evaluate()
