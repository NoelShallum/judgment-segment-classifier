"""Recompute native BERT review after the frozen whole-judgment CRF patch."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from safetensors.numpy import load_file
from transformers import BertTokenizerFast

import native_edits as e
import sentence_bert_referral_write as b

sys.path.insert(0, str(e.REVIEWER))
from fit import Readout, prob
from native import donor_at

HERE = e.HERE / "reroute"
SOURCE = e.HERE.parent / "t4-return"
GROUPS = ("confirm", "fresh")
MISSING = HERE / "new-windows.npz"
MODEL = e.ROOT / "references/checkpoint/InLegalBERT(i)model.safetensors"
FIELDS = ("ids", "mask", "positions", "base", "original", "aux", "prior")


def original_case(key, group):
    directory = e.ROUND2 / "new-cases" if group == "fresh" else e.REVIEWER / "cases"
    with np.load(directory / (key + ".npz"), allow_pickle=False) as data:
        return {name: data[name] for name in (*FIELDS, "gold", "case_sha256")}


def encode(tokenizer, case_view):
    windows = b.pack_ids(tokenizer, [case_view], b.MODES[1])
    ids = np.full((len(windows), b.LIMIT), tokenizer.pad_token_id, dtype=np.int32)
    mask = np.zeros(ids.shape, dtype=bool)
    for idx, values in enumerate(windows):
        ids[idx, :len(values)] = values
        mask[idx, :len(values)] = True
    return ids, mask


def prepare_shard(task):
    group, keys, rows, spec, patch = task
    pipeline = e.Pipeline(memo=True)
    tokenizer = BertTokenizerFast.from_pretrained(e.ROOT / "references/tokenizer", local_files_only=True)
    out = []
    smoke = {}
    for i, key in enumerate(keys, 1):
        original = original_case(key, group)
        text, gold = zip(*rows[key])
        assert spec["text_sha256"][key] == original["case_sha256"].item() == e.text_digest(text)
        np.testing.assert_array_equal(original["gold"], gold)
        item, h, lex = pipeline.features(key, text)
        def capture(change):
            pipeline.install_absolute(change)
            # Use the exact staging and route employed by predict_wide_bert_review.
            base, positions, original_discourse = pipeline.predict(item, h, lex, include_original=True)
            row = b.case_view(item, h, pipeline.model, pipeline.cfg, pipeline.vocab, positions)
            np.testing.assert_array_equal(row[2], original_discourse)
            ids, mask = encode(tokenizer, row)
            return dict(ids=ids, mask=mask, positions=positions, base=base, original=row[2][positions],
                        aux=row[4], prior=row[5])
        if i == 1:
            observed = capture((pipeline.w0, pipeline.b0, pipeline.t0))
            errors = {}
            for k, v in observed.items():
                if k in ("aux", "prior"):
                    # PyTorch and BLAS thread counts can change source floats in the seventh decimal.
                    np.testing.assert_allclose(v, original[k], atol=1.e-5, rtol=0, err_msg=(group, key, k))
                    errors[k] = float(np.max(np.abs(v - original[k])))
                else:
                    np.testing.assert_array_equal(v, original[k], err_msg=(group, key, k))
            smoke[key] = errors
        out.append((key, original["gold"], original, capture(patch)))
        if i % 12 == 0:
            print(group, "prepared", i, "/", len(keys), flush=True)
    return out, smoke


def combine(parts, k):
    return np.concatenate([row[k] for row in parts], axis=0)


def global_bank(rows, offsets):
    return dict(ids=combine(rows, "ids"), mask=combine(rows, "mask"),
                review=np.concatenate([lo + row["positions"] for lo, row in zip(offsets, rows)]),
                ro=np.r_[0, np.cumsum([len(row["positions"]) for row in rows])],
                **{name: combine(rows, name) for name in ("base", "original", "aux", "prior")})


def take(window, mask):
    return window.tobytes(), mask.tobytes()


def prepare():
    HERE.mkdir(exist_ok=True)
    plan = e.recipe()
    frozen = e.read(e.HERE / "freeze.json")
    path = e.HERE / "crf.safetensors"
    assert e.file_hash(path) == frozen["parameter_files"]["crf"]
    params = load_file(str(path))
    patch = (params[e.EMITTER + "weight"], params[e.EMITTER + "bias"], params[e.CRF])
    selected = [k for name in GROUPS for k in plan["groups"][name]]
    rows = e.load_rows("val", set(selected))
    ledger = {}
    new_splits = {}
    jobs = []
    for group in GROUPS:
        for i in range(2):
            keys = plan["groups"][group][i::2]
            jobs.append((group, keys, {k: rows[k] for k in keys}, plan, patch))
    with ProcessPoolExecutor(4) as pool:
        shards = list(pool.map(prepare_shard, jobs))
    source = {r[0]: r[1:] for shard, _ in shards for r in shard}
    smoke = {key: err for _, comparisons in shards for key, err in comparisons.items()}
    for group in GROUPS:
        keys = plan["groups"][group]
        ordered = [source[k] for k in keys]
        offsets = np.r_[0, np.cumsum([len(r[0]) for r in ordered])]
        gold = np.concatenate([r[0] for r in ordered])
        base, updated = [global_bank([r[i] for r in ordered], offsets) for i in (1, 2)]
        get = {take(v, m): i for i, (v, m) in enumerate(zip(base["ids"], base["mask"]))}
        link = np.array([get.get(take(v, m), -1) for v, m in zip(updated["ids"], updated["mask"])])
        q = link >= 0
        matched = np.flatnonzero(q)
        np.testing.assert_array_equal(updated["ids"][matched], base["ids"][link[q]])
        np.testing.assert_array_equal(updated["mask"][matched], base["mask"][link[q]])
        changed_routes = len(np.setdiff1d(updated["review"], base["review"]))
        with np.load(e.HERE / (group + "-predictions.npz"), allow_pickle=False) as old:
            np.testing.assert_array_equal(old["ids"], keys)
            np.testing.assert_array_equal(old["gold"], gold)
            np.testing.assert_array_equal(old["identity"], base["base"])
            np.testing.assert_array_equal(old["crf"], updated["base"])
            np.testing.assert_array_equal(old["positions"], base["review"])
        data_path = HERE / (group + "-banks.npz")
        e.stash(data_path, review_link=link, case_ids=keys, gold=gold, offsets=offsets,
                **{f"{tag}.{k}": v for tag, s in (("identity", base), ("crf", updated)) for k, v in s.items()})
        new_splits[f"{group}.ids"] = updated["ids"][~q]
        new_splits[f"{group}.mask"] = updated["mask"][~q]
        ledger[group] = {"cases": len(keys), "sentences": len(gold), "archived_route_size": len(base["review"]),
                         "edited_route_size": len(updated["review"]), "newly_routed_positions": changed_routes,
                         "exact_token_and_mask_matches": len(matched), "new_windows": int((~q).sum()),
                         "bank_sha256": e.file_hash(data_path)}
        print(group, ledger[group], flush=True)
    e.stash(MISSING, **new_splits)
    e.save_json(HERE / "prepare-receipt.json", {
        "frozen_crf_sha256": e.file_hash(path), "groups": ledger,
        "new_windows_sha256": e.file_hash(MISSING), "gpu_feature_matching": "Exact token IDs and masks, from same checkpoint.",
        "unchanged_route_inputs": "Original T4 case shards aligned to exact recomputed routes and parent labels on every row. Four fresh identity forwards spot-check auxiliaries, anchor, and token inputs.",
        "unchanged_route_forward_max_errors": smoke,
        "source_plan_sha256": e.file_hash(e.PLAN), "test_accessed": False,
    })


def capture_cpu():
    plan = e.read(HERE / "prepare-receipt.json")
    assert e.file_hash(MISSING) == plan["new_windows_sha256"]
    e.torch.set_num_threads(4)
    model = donor_at(e.REVIEWER, MODEL, "cpu")
    out = {}
    with np.load(MISSING, allow_pickle=False) as batch, e.torch.inference_mode():
        for name in GROUPS:
            ids, mask = batch[name + ".ids"], batch[name + ".mask"]
            pooled = []
            for i in range(0, len(ids), 32):
                x = e.torch.from_numpy(ids[i:i + 32]).long()
                m = e.torch.from_numpy(mask[i:i + 32])
                pooled.append(model(x, attention_mask=m).pooler_output.numpy())
            out[name] = np.concatenate(pooled) if pooled else np.empty((0, 768), dtype=np.float32)
            print(name, "native CPU windows", len(ids), flush=True)
    path = HERE / "new-pooled.npz"
    e.stash(path, **out)
    e.save_json(HERE / "new-pool-receipt.json", {
        "device": "CPU", "input_sha256": e.file_hash(MISSING), "model_sha256": e.file_hash(MODEL),
        "output_sha256": e.file_hash(path), "rows": {k: len(v) for k, v in out.items()},
    })


def evaluate(new_pools, new_receipt):
    plan = e.read(HERE / "prepare-receipt.json")
    assert e.file_hash(MISSING) == plan["new_windows_sha256"]
    r = e.read(SOURCE / "pool-receipt.json")
    update = e.read(new_receipt)
    assert update["input_sha256"] == plan["new_windows_sha256"]
    assert update["model_sha256"] == r["model_sha256"]
    assert update["output_sha256"] == e.file_hash(new_pools)
    if update["device"] != "CPU":
        assert update["device"]["devices"][0].split(",")[0].strip() in {"Tesla T4", "NVIDIA T4", "T4"}
    head = Readout(e.REVIEWER, "cpu")
    e.torch.set_num_threads(2)
    results = {"groups": {}, "input_receipt_sha256": e.file_hash(HERE / "prepare-receipt.json"),
               "new_pool_sha256": e.file_hash(new_pools), "fresh_feature_backend": "CPU" if update["device"] == "CPU" else "Tesla T4"}
    with np.load(new_pools, allow_pickle=False) as pool:
        for group in GROUPS:
            expected_path = SOURCE / (group + ".npz")
            path = HERE / (group + "-banks.npz")
            assert e.file_hash(expected_path) == r["groups"][group]["file_sha256"]
            assert e.file_hash(path) == plan["groups"][group]["bank_sha256"]
            with np.load(expected_path, allow_pickle=False) as f, np.load(path, allow_pickle=False) as d:
                ids, y, offsets, link = (d[k] for k in ("case_ids", "gold", "offsets", "review_link"))
                np.testing.assert_array_equal(ids, f["case_ids"])
                np.testing.assert_array_equal(offsets, f["offsets"])
                np.testing.assert_array_equal(d["identity.ro"], f["review_offsets"])
                baseline_pool = f["pool"]
                reference = f["reference"]
                mixed = np.empty((len(link), baseline_pool.shape[1]), dtype=np.float32)
                matched = link >= 0
                mixed[matched] = baseline_pool[link[matched]]
                assert pool[group].shape == ((~matched).sum(), 768)
                mixed[~matched] = pool[group]
                assert np.isfinite(mixed).all()
                bank_output = {}
                for tag, v in (("identity", baseline_pool), ("crf", mixed)):
                    bank = SimpleNamespace(parent=d[tag + ".base"], review=d[tag + ".review"],
                                           original=d[tag + ".original"], prior=d[tag + ".prior"])
                    a = e.torch.from_numpy(d[tag + ".aux"])
                    pooled = e.torch.from_numpy(v)
                    with e.torch.inference_mode():
                        response = prob(head.raw(a, pooled))
                    bank_output[tag] = head.review(bank, response)
                np.testing.assert_array_equal(bank_output["identity"], reference)
                with np.load(e.ROUND_PRED / (group + "-predictions.npz"), allow_pickle=False) as z:
                    np.testing.assert_array_equal(reference, z["incumbent"])
                candidate = bank_output["crf"]
                results["groups"][group] = {
                    **plan["groups"][group], "identity": e.scores(y, reference), "crf": e.scores(y, candidate),
                    "paired_ci95": e.interval(y, offsets, reference, candidate),
                    "changes": int(np.count_nonzero(candidate != reference)),
                    "corrected": int(np.sum((candidate == y) & (reference != y))),
                    "broken": int(np.sum((candidate != y) & (reference == y))),
                }
                e.stash(HERE / (group + "-predictions.npz"), case_ids=ids, offsets=offsets, gold=y,
                        identity=reference, crf=candidate, identity_review=d["identity.review"], crf_review=d["crf.review"])
                c = results["groups"][group]
                print(group, {k: (round(c[k]["macro_f1"] * 100, 4), c[k]["pair_errors"]) for k in ("identity", "crf")}, flush=True)
    e.save_json(HERE / "results.json", results)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("op", choices=("prepare", "cpu-pool", "evaluate"))
    cli.add_argument("--new-pools", type=Path, default=HERE / "new-pooled.npz")
    cli.add_argument("--new-receipt", type=Path, default=HERE / "new-pool-receipt.json")
    args = cli.parse_args()
    if args.op == "prepare":
        prepare()
    elif args.op == "cpu-pool":
        capture_cpu()
    else:
        evaluate(args.new_pools, args.new_receipt)


if __name__ == "__main__":
    main()
