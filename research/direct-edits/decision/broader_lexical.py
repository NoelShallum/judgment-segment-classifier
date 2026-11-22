"""Directly write GLiDE's native decision rows from increasingly broad case-disjoint public training sets."""
import csv
import hashlib
import json
import re
import time
from collections import Counter, defaultdict

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer

from cohort import LABEL_ID, OFF, case_rows
from compare import scores, stats
from fetch import MODEL, ROOT, sha
from readout_study import raw_mask
from weights import arr, ridge, savej, source, wtproof

OUT = ROOT / "broader-static-lexical"
PLAN = ROOT / "broader-lexical-plan.json"


def freeze_cases():
    csv_path = OFF / "train.csv"
    ids = set()
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["Text"].strip():
                ids.add(row["Index"])
    selected = sorted(ids, key=lambda x: hashlib.sha256(("decision-lexical-scale-2026-11-22|" + x).encode()).digest())[:256]
    validation = json.loads((ROOT / "packed-full-plan.json").read_text())["fixed_validation_ids"]
    assert not (set(selected) & set(validation))
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"plan": sha(PLAN), "all_train_ids": len(ids), "selected_ids": selected,
           "validation_ids": validation, "official_train_sha256": sha(csv_path)}
    savej(OUT / "cases.json", out)
    return selected, validation


def scaled_token_table():
    with safe_open(str(MODEL / "model-00001-of-00001.safetensors"), "pt", device="cpu") as src:
        t = src.get_tensor("model.embeddings.tok_embeddings.weight")
        norm = arr(src.get_tensor("model.embeddings.norm.weight"))
        table = np.empty(tuple(t.shape), dtype=np.float32)
        for i in range(0, len(table), 1024):
            b = arr(t[i:i + 1024])
            b -= b.mean(1, keepdims=True)
            b /= np.sqrt((b * b).mean(1, keepdims=True) + np.float32(1.e-5))
            b *= norm
            table[i:i + len(b)] = b
        return table


def norm_text(t):
    return re.sub(r"\s+", " ", t.strip()).casefold()


def feature_bank(ids, docs, tokenizer, table):
    rows = sum(len(docs[c]) for c in ids)
    x = np.empty((rows, table.shape[1]), np.float32)
    y, cases, positions, text_hash = [], [], [], []
    i = 0
    for cid in ids:
        for pos, (t, label) in enumerate(docs[cid]):
            tokens, selected = raw_mask(tokenizer, t)
            x[i] = table[tokens[selected]].mean(0)
            y.append(LABEL_ID[label])
            cases.append(cid)
            positions.append(pos)
            text_hash.append(hashlib.sha256(norm_text(t).encode()).digest())
            i += 1
    return x, np.asarray(y), np.asarray(cases), np.asarray(positions), text_hash


def primal(x, y, w):
    n, width = x.shape
    freq = np.maximum(np.bincount(y, minlength=7), 1)
    sample = np.float64(1.) / np.sqrt(freq[y])
    sample /= sample.mean()
    ss = sum(float(np.square(x[i:i + 256], dtype=np.float64).sum()) for i in range(0, n, 256))
    scale = np.sqrt(ss / n)
    gram = .15 * np.eye(width, dtype=np.float64)
    cross = np.zeros((width, 7), dtype=np.float64)
    hot = np.eye(7)
    for i in range(0, n, 256):
        chunk = x[i:i + 256]
        z = chunk.astype(np.float64) / scale
        original = chunk @ w.T
        residual = 4. * (hot[y[i:i + len(z)]] - 1. / 7.) - (original - original.mean(1, keepdims=True))
        weighted = sample[i:i + len(z), None] * z
        gram += z.T @ weighted
        cross += weighted.T @ residual
    return (np.linalg.solve(gram, cross) / scale).T


def run():
    started = time.monotonic()
    train_ids, val_ids = freeze_cases()
    tokenizer = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    table = scaled_token_table()
    base = source()[0]
    w = arr(base)[:7]
    # Confirm both linear systems before any broader fit or validation scoring.
    canon = json.loads((ROOT / "packed-full-plan.json").read_text())["fixed_fit_ids"]
    small, gold, *_ = feature_bank(canon, case_rows("train", set(canon)), tokenizer, table)
    pred = small @ w.T
    target = 4 * (np.eye(7)[gold] - 1 / 7) - (pred - pred.mean(1, keepdims=True))
    delta = primal(small, gold, w)
    error = float(np.abs(delta - ridge(small, target, gold)).max())
    check = base.clone()
    check[:7] = torch.from_numpy(w + delta).to(base.dtype)
    existing = load_file(str(ROOT / "readout-study/static-tokens/roles-readout.safetensors"))["weight"]
    max_saved = float(torch.max(torch.abs(check.float() - existing.float())))
    assert error < 2.e-6 and max_saved < 1.e-4, (error, max_saved)
    print("validated primal against dual on 357 examples", error, max_saved, flush=True)
    data = case_rows("train", set(train_ids))
    sizes = [len(data[k]) for k in train_ids]
    x, y, c, _, train_text = feature_bank(train_ids, data, tokenizer, table)
    bank_ref = {"shape": list(table.shape), "kept_input_parameters": table.size + table.shape[1] + 7 * table.shape[1]}
    print("static training rows", len(y), "seconds", round(time.monotonic() - started, 1), flush=True)
    fits = {}
    for name, count, shuffle in [("12-cases", 12, False), ("64-cases", 64, False),
                                  ("256-cases", 256, False), ("256-shuffled", 256, True)]:
        n = sum(sizes[:count])
        labels = y[:n]
        if shuffle:
            labels = np.random.default_rng(221126).permutation(labels)
        written = base.clone()
        written[:7] = torch.from_numpy(w + primal(x[:n], labels, w)).to(written.dtype)
        assert torch.equal(written[7:], base[7:])
        path = OUT / (name + "-readout.safetensors")
        save_file({"weight": written}, str(path))
        fits[name] = {"n": n, "case_count": count, "support": np.bincount(labels, minlength=7).tolist(),
                      "sha256": sha(path), "tensor": wtproof(base, written)}
        print("frozen", name, n, "seconds", round(time.monotonic() - started, 1), flush=True)
    savej(OUT / "fit.json", {"manifest": sha(OUT / "cases.json"), "plan": sha(PLAN), "native_embedding": bank_ref,
          "source_counts": {cid: n for cid, n in zip(train_ids, sizes)}, "dual_primal_max_error": error,
          "stored_reference_357_max_abs": max_saved, "gradient_steps": 0, "methods": fits})
    v, vy, vc, pos, hashes = feature_bank(val_ids, case_rows("val", set(val_ids)), tokenizer, table)
    del table, x, data
    incumbent = {cid: json.loads((ROOT.parent / "pooler/shared-cases" / (cid + ".json")).read_text())["predictions"]["unchanged"] for cid in val_ids}
    inc = np.asarray([incumbent[cid][i] for cid, i in zip(vc, pos)])
    all_train = set(train_text)
    novel = np.array([h not in all_train for h in hashes], dtype=bool)
    baseline = (v @ arr(existing)[:7].T).argmax(1)
    estimates = {"12-cases-357-earlier": baseline, "unchanged": (v @ w.T).argmax(1)}
    for name, item in fits.items():
        p = OUT / (name + "-readout.safetensors")
        assert sha(p) == item["sha256"]
        estimates[name] = (v @ arr(load_file(str(p))["weight"])[:7].T).argmax(1)
    estimates["matched_incumbent"] = inc
    res = {}
    for scope, mask in [("all", np.ones(len(v), dtype=bool)), ("exclude_exact_train_text", novel)]:
        bc = stats(vy[mask], baseline[mask], vc[mask])[0]
        ic = stats(vy[mask], inc[mask], vc[mask])[0]
        rng = np.random.default_rng(760230)
        bs = rng.integers(0, len(bc), (4000, len(bc)))
        br = scores(bc[bs].sum(1))
        ir = scores(ic[bs].sum(1))
        res[scope] = {"rows": int(mask.sum()), "judgments": len(set(vc[mask])), "scores": {}}
        for name, p in estimates.items():
            cm, metric = stats(vy[mask], p[mask], vc[mask])
            boot = scores(cm[bs].sum(1))
            res[scope]["scores"][name] = {"metrics": metric, "paired_delta_vs_357_static_95": np.quantile(boot-br, [.025,.975]).tolist(),
                                           "paired_delta_vs_incumbent_95": np.quantile(boot-ir, [.025,.975]).tolist()}
    savej(OUT / "validation.json", {"preregistered": sha(PLAN), "fit": sha(OUT / "fit.json"), "cohorts": res})
    savej(OUT / "predictions.json", {"case": vc.tolist(), "position": pos.tolist(), "gold": vy.tolist(),
                                      "novel_text": novel.tolist(), "predictions": {k: z.tolist() for k, z in estimates.items()}})
    print("broader public static lexical", {k: round(q["metrics"]["macro_f1"], 6) for k,q in res["all"]["scores"].items()}, flush=True)


if __name__ == "__main__":
    run()
