"""Capture the pinned incumbent review inputs for ordered extra-training judgments."""

import argparse
from functools import lru_cache
from hashlib import file_digest, sha256
import gzip
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


HERE = Path(__file__).resolve().parent
CONT = HERE.parent
ROOT = CONT.parent
OLD = ROOT / "live-nov22/bert-t4"
ROUND2 = ROOT / "live-nov22/reviewer-round2"
SCALE = CONT / "scale"
WORD_DIR = ROOT / "live-nov22/hierarchy/scale/words"
WORD_PLAN = WORD_DIR.parent / "plan.json"
OLD_POOLS = CONT / "t4-return"
CASE_DIR = HERE / "cases"
SHARDS = HERE / "shards"
FIELDS = ("ids", "mask", "positions", "base", "original", "aux", "prior")


def sha(path):
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def save(path, **arrays):
    tmp = path.with_suffix(".part")
    with tmp.open("wb") as output:
        np.savez_compressed(output, **arrays)
    tmp.replace(path)


def snapshot():
    old = read(OLD / "protocol.json")
    scale = read(SCALE / "protocol.json")
    words = read(WORD_PLAN)
    assert words["source"] == old["csv_sha256"]["train"]
    assert words["donor"] == sha(ROOT / "references/hier/hier.safetensors")
    assert scale["train_only"] and scale["shard_size"] == 64
    prefix = scale["case_ids"][:512]
    assert len(prefix) == len(set(prefix)) == 512 and set(prefix).isdisjoint(old["groups"]["fit"])
    state = {
        "source_scale_sha256": sha(SCALE / "protocol.json"), "source_train_sha256": old["csv_sha256"]["train"],
        "source_records_sha256": sha(SCALE / "fit.jsonl.gz"), "old_protocol_sha256": sha(OLD / "protocol.json"),
        "word_cache_plan_sha256": sha(WORD_PLAN), "hierarchy_sha256": words["donor"],
        "capture_source_sha256": sha(OLD / "prepare.py"), "route_source_sha256": sha(ROOT / "predict_wide_bert_review.py"),
        "prior_donor_sha256": read(OLD / "review.json")["model"]["sha256"],
        "prior_head_sha256": sha(OLD / "head.safetensors"), "case_ids": prefix, "shard_size": 64,
        "capture": "Frozen 20% confidence route including original score features, prior, tokenizer IDs, and review positions. Ordered prefixes of 64 from the independent expanded public training cohort.",
        "raw_word_cache": "Per-document raw sentence embeddings available for most public training cases, pinned original HierModel and same official train CSV. Verify source row order and independent word checks. A prefixed parity sample compares full original and accelerated capture. Readout and routing use the original code.",
        "official_test_read": False,
    }
    dest = HERE / "protocol.json"
    if dest.exists():
        assert state == read(dest)
    else:
        write(dest, state)
    return state


@lru_cache(maxsize=1)
def original():
    name = "incumbent_route_capture"
    spec = importlib.util.spec_from_file_location(name, OLD / "prepare.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.sequence.phrase.load_predictor = lru_cache(maxsize=1)(module.sequence.phrase.load_predictor)
    module.sequence.load_decoder = lru_cache(maxsize=1)(module.sequence.load_decoder)
    tokenizer = module.BertTokenizerFast.from_pretrained(ROOT / "references/tokenizer", local_files_only=True)
    return module, tokenizer, module.pair.predictor("direct")


def training(spec, limit):
    helper = original()[0]
    train_file = ROOT / "references/official/train.csv"
    assert sha(train_file) == spec["source_train_sha256"]
    keys = spec["case_ids"][:limit]
    data = helper.cases("train", set(keys))
    fingerprints = {}
    with gzip.open(SCALE / "fit.jsonl.gz", "rt", encoding="utf-8") as source:
        for j, key in enumerate(keys):
            reference = json.loads(next(source))
            rows = data[key]
            assert key == reference["case"] and j // 64 == reference["shard"]
            assert [r[1] for r in rows] == reference["text"] and [r[2] for r in rows] == reference["gold"]
            fingerprints[key] = helper.fingerprint(rows)
    assert len(set(fingerprints.values())) == len(keys)
    return data, fingerprints


def capture(rows, key, use_words=True):
    import torch

    helper, tokenizer, parent = original()
    model = helper.sequence.phrase.load_predictor("target phrases")[0]
    torch.set_num_threads(2)
    texts = [r[1] for r in rows]
    cache = WORD_DIR / (key + ".npy")
    start = perf_counter()
    mode, max_error, used_digest = "canonical", None, None
    if use_words and cache.is_file():
        cached = np.load(cache, allow_pickle=False)
        assert cached.shape == (len(texts), 200) and cached.dtype == np.float32 and np.isfinite(cached).all()
        indices = sorted({0, len(texts) // 2, len(texts) - 1})
        method = model.words
        confirmed = method([texts[i].strip() for i in indices]).numpy()
        max_error = float(np.max(np.abs(confirmed - cached[indices])))
        if np.allclose(confirmed, cached[indices], atol=4e-5, rtol=0.):
            exact_input = [t.strip() for t in texts]

            def reuse(batch, *args, **kwargs):
                assert list(batch) == exact_input, "Unexpected sentence boundary before document encoder"
                return torch.from_numpy(cached)

            model.words = reuse
            try:
                encoded = helper.route_capture(texts, tokenizer, parent)
            finally:
                model.words = method
            mode, used_digest = "verified-cached-word-input", sha(cache)
        else:
            encoded = helper.route_capture(texts, tokenizer, parent)
    else:
        encoded = helper.route_capture(texts, tokenizer, parent)
    elapsed = perf_counter() - start
    assert len(encoded["base"]) == len(rows) and len(encoded["positions"]) == int(np.ceil(len(rows)*.20))
    return encoded, {"mode": mode, "sample_word_max_abs_error": max_error, "source_words_sha256": used_digest, "seconds": elapsed}


def verify_cached_route(spec, n=8):
    helper = original()[0]
    exact = spec["case_ids"][:n]
    data, fps = training(spec, n)
    audits = {}
    for key in exact:
        raw, control = capture(data[key], key, use_words=False)
        reused, fast = capture(data[key], key, use_words=True)
        differences = {}
        for name in FIELDS:
            a, b = raw[name], reused[name]
            if name in ("aux", "prior"):
                np.testing.assert_allclose(a, b, atol=3e-5, rtol=0.)
                differences[name] = float(np.max(np.abs(a - b)))
            else:
                np.testing.assert_array_equal(a, b)
                differences[name] = 0
        assert fps[key] == helper.fingerprint(data[key])
        audits[key] = {"sentences": len(raw["base"]), "reviewed": len(raw["positions"]),
                       "reference_seconds": control["seconds"], "fast_seconds": fast["seconds"],
                       "cache_used": fast["mode"] != "canonical", "max_differences": differences}
        print("parity", key, len(raw["base"]), fast["mode"], f"{control['seconds']:.2f}s→{fast['seconds']:.2f}s", flush=True)
    write(HERE / "capture-parity.json", {"cases": audits, "protocol_sha256": sha(HERE / "protocol.json"),
                                         "exact": "Token windows, route positions, and both role-index arrays were identical to canonical immutable inference. Input floats compared at the specified tolerance."})


def materialize(spec, count, cached_words):
    CASE_DIR.mkdir(exist_ok=True)
    SHARDS.mkdir(exist_ok=True)
    data, fingerprints = training(spec, count)
    start = perf_counter()
    totals = {"rows": 0, "review_rows": 0, "word_cache": 0, "cases": 0}
    for i, key in enumerate(spec["case_ids"][:count]):
        output = CASE_DIR / (key + ".npz")
        receipt = CASE_DIR / (key + ".json")
        if output.exists() and receipt.exists():
            old = read(receipt)
            assert old["file_sha256"] == sha(output) and old["case_sha256"] == fingerprints[key]
            meta = old
        else:
            rows = data[key]
            values, meta = capture(rows, key, cached_words)
            indices, _, gold = zip(*rows)
            save(output, **values, original_row=np.asarray(indices, np.int64), gold=np.asarray(gold, np.int8),
                 case_sha256=fingerprints[key])
            meta.update(case_id=key, file_sha256=sha(output), case_sha256=fingerprints[key], rows=len(rows), reviewed=len(values["positions"]))
            write(receipt, meta)
        totals["cases"] += 1
        totals["rows"] += meta["rows"]
        totals["review_rows"] += meta["reviewed"]
        totals["word_cache"] += int(meta["mode"] == "verified-cached-word-input")
        if i == 0 or (i + 1) % 8 == 0:
            progress = {**totals, "seconds": perf_counter() - start}
            write(HERE / "progress.json", progress)
            print("capture", progress, flush=True)
        if (i + 1) % 64 == 0:
            pack_inputs(spec, (i + 1) // 64 - 1)


def pack_inputs(spec, block):
    keys = spec["case_ids"][64 * block:64 * (block + 1)]
    out = SHARDS / f"anchor-{block:02d}.npz"
    if out.exists() and out.with_suffix(".json").exists():
        current = read(out.with_suffix(".json"))
        assert current["case_ids"] == keys and current["protocol_sha256"] == sha(HERE / "protocol.json")
        assert current["file_sha256"] == sha(out)
        print("reused checkpoint", block, current["rows"], current["reviewed"], current["file_sha256"], flush=True)
        return
    rows = []
    receipts = []
    for key in keys:
        f = CASE_DIR / (key + ".npz")
        info = read(f.with_suffix(".json"))
        assert sha(f) == info["file_sha256"]
        with np.load(f, allow_pickle=False) as src:
            assert src["case_sha256"].item() == info["case_sha256"]
            rows.append({name: src[name] for name in (*FIELDS, "original_row", "gold")})
        receipts.append(info)
    offset = np.r_[0, np.cumsum([len(v["base"]) for v in rows])]
    route_offset = np.r_[0, np.cumsum([len(v["positions"]) for v in rows])]
    positions = np.concatenate([a + r["positions"] for a, r in zip(offset[:-1], rows)])
    inputs = {name: np.concatenate([r[name] for r in rows]) for name in rows[0] if name != "positions"}
    save(out, **inputs, positions=positions, case_ids=np.array(keys), offsets=offset, review_offsets=route_offset)
    write(out.with_suffix(".json"), {"file_sha256": sha(out), "protocol_sha256": sha(HERE / "protocol.json"), "rows": int(offset[-1]),
                                    "reviewed": int(route_offset[-1]), "case_ids": keys, "cases": receipts})
    print("checkpoint", block, len(keys), int(offset[-1]), int(route_offset[-1]), sha(out), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("plan", "audit", "capture"))
    ap.add_argument("--cases", type=int, default=256)
    ap.add_argument("--canonical", action="store_true", help="Capture every case without previously saved word inputs")
    args = ap.parse_args()
    spec = snapshot()
    if args.mode == "capture":
        if args.cases < 64 or args.cases > 512 or args.cases % 64:
            ap.error("Use an ordered 64-case increment from 64 through 512")
        if not args.canonical:
            assert read(HERE / "capture-parity.json")["protocol_sha256"] == sha(HERE / "protocol.json")
        materialize(spec, args.cases, not args.canonical)
    elif args.mode == "audit":
        verify_cached_route(spec, n=min(args.cases, 8))
