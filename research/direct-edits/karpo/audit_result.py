"""Independently verify a Karpo archive against frozen LegalSeg rows and readouts."""
import argparse
import csv
from hashlib import file_digest, sha256
from io import BytesIO
from itertools import groupby, product
import json
import os
from pathlib import Path, PurePosixPath
import struct
from zipfile import ZipFile

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np


HERE = Path(__file__).resolve().parent
ROLE = ("None", "Facts", "Issue", "Arguments of Petitioner", "Arguments of Respondent", "Reasoning", "Decision")
ARMS = ("native", "edited", "shuffled", "incumbent")
READOUT_SHA = "e290da9edde55094aff5aa26c7b6d0f5a65b08eac590570af35a2c6fb0d6983e"
SHARD_SHA = "656bc586f18e9c26df29033ebd50036b33004de82b2c61e4a670124e88d3143f"
PLAN_SHA = "a571e4e5b78e46f712586670fffad146b8b1f8efcd1b0470ba5c7dc4e86ffd6e"
WORKER_SHA = "03f971f92957781594ec0ed1ffa2bed79a748092ce44a4561322db4f3734be70"
SHAPE = (255, 1024)
SEVEN = list(range(7))


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def hash_file(path):
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def hash_bytes(raw):
    return sha256(raw).hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def members(archive):
    names = archive.namelist()
    need(len(set(names)) == len(names), "Duplicate named archive members")
    need(all(not PurePosixPath(n).is_absolute() and ".." not in PurePosixPath(n).parts for n in names),
         "Unexpected path in results archive")
    need(archive.testzip() is None, "Archive CRC failed")
    return set(names)


def readout(raw):
    header_length, = struct.unpack_from("<Q", raw)
    end = 8 + header_length
    header = json.loads(raw[8:end])
    header.pop("__metadata__", None)
    wanted = {"weight": {"dtype": "F32", "shape": list(SHAPE), "data_offsets": [0, 4 * int(np.prod(SHAPE))]}}
    need(header == wanted and len(raw) - end == wanted["weight"]["data_offsets"][1], "Unexpected safetensors header/length")
    matrix = np.frombuffer(raw, dtype="<f4", offset=end).reshape(SHAPE, order="C")
    need(matrix.flags.c_contiguous and matrix.strides == (4096, 4) and matrix.tobytes(order="C") == raw[end:],
         "Safetensors buffer is not the expected physical row-major matrix")
    need(np.isfinite(matrix).all(), "Non-finite readout weight")
    return matrix, 4 * int(np.prod(SHAPE))


def local_inputs():
    plan_path = HERE / "protocol.json"
    worker_path = HERE / "karpo-t4-full-worker.zip"
    need(hash_file(plan_path) == PLAN_SHA and hash_file(worker_path) == WORKER_SHA, "Local preregistration or worker has changed")
    plan = read_json(plan_path)
    need(tuple(plan["roles"]) == ROLE, "Official seven-role mapping changed")
    frozen = {}
    for key in ("input", "truth", "incumbent"):
        path = HERE / (("inputs" if key == "input" else key) + ".json")
        need(hash_file(path) == plan[key + "_sha256"], f"Changed local {key} bytes")
        frozen[key] = read_json(path)
    canonical = HERE.parent / "decision" / "glide-content-colab.zip"
    need(hash_file(canonical) == plan["outer_archive_sha256"], "Changed comparison-model input archive")
    with ZipFile(canonical) as archive:
        sample_bytes = archive.read("sampled-texts.json")
        need(hash_bytes(sample_bytes) == plan["glide_sample_bytes_sha256"], "Changed original shared samples")
    source = json.loads(sample_bytes)
    manifest = HERE.parent / "shared-confirmation.json"
    need(hash_file(manifest) == plan["source_csv_sha256"]["shared_manifest"], "Changed held-out case manifest")
    need(source["source"] == plan["source_csv_sha256"], "Source digest sets disagree")
    need(read_json(manifest)["small_validation_judgment_ids"] == plan["validation_cases"], "Unexpected comparison judgments")
    for split, case_key, n_cases, n_rows in (("train", "fit_cases", 12, 357), ("val", "validation_cases", 16, 3558)):
        keys = plan[case_key]
        need(len(keys) == len(set(keys)) == n_cases, f"Invalid {split} case count")
        need(set(keys) == set(source[split]) == set(frozen["input"][split]) == set(frozen["truth"][split]),
             f"Missing or extra case in {split}")
        for case in keys:
            check = list(zip(frozen["input"][split][case], [ROLE[y] for y in frozen["truth"][split][case]], strict=True))
            need(check == list(map(tuple, source[split][case])) and len(check) == plan["rows"][case],
                 f"Order, label, or text changed for {split}/{case}")
        need(sum(plan["rows"][k] for k in keys) == n_rows, f"Invalid {split} row count")
    need(not set(plan["fit_cases"]) & set(plan["validation_cases"]), "Fit and validation document IDs overlap")
    val_path = HERE.parents[1] / "references" / "official" / "val.csv"
    need(hash_file(val_path) == plan["source_csv_sha256"]["val"], "Changed original official validation CSV")
    val = {k: [] for k in plan["validation_cases"]}
    with val_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            case = row["Index"]
            if case in val and (sentence := row["Text"].strip()):
                val[case].append([sentence, row["Label"]])
    need(val == source["val"], "Raw official CSV does not match all 3,558 retained text/role pairs")
    case_rows = [(key, index, sentence, role) for key in plan["validation_cases"]
                 for index, (sentence, role) in enumerate(val[key])]
    evidence = {"rows": len(case_rows), "fit_rows": 357, "judgments": plan["validation_cases"], "roles": ROLE,
                "heldout_sha256": hash_bytes(json.dumps(case_rows, ensure_ascii=False, separators=(",", ":")).encode()),
                "source_val_csv_sha256": plan["source_csv_sha256"]["val"],
                "shared_manifest_sha256": plan["source_csv_sha256"]["shared_manifest"],
                "shared_samples_sha256": plan["glide_sample_bytes_sha256"], "worker_sha256": WORKER_SHA}
    original = (HERE / "model" / "readout.safetensors").read_bytes()
    need(hash_bytes(original) == READOUT_SHA, "Wrong publisher readout")
    base, payload = readout(original)
    evidence["publisher_readout"] = {"sha256": READOUT_SHA, "payload_bytes": payload, "shape": SHAPE, "row_major": True}
    return plan, frozen, base, evidence


def check_training(archive, plan, frozen, trained, base):
    need(trained["protocol_sha256"] == PLAN_SHA and trained["program_sha256"] == hash_file(HERE / "program.py"),
         "Frozen ridge run has a different program or protocol")
    need(trained["base_readout_sha256"] == READOUT_SHA and trained["base_checkpoint_sha256"] == SHARD_SHA,
         "Fit did not declare pinned model")
    need(trained["judgments"] == plan["fit_cases"] and trained["rows"] == 357 and trained["gradient_steps"] == 0,
         "Fit corpus or weight-write mode differs")
    train = np.concatenate([frozen["truth"]["train"][k] for k in trained["judgments"]])
    np.testing.assert_array_equal(trained["role_support"], np.bincount(train, minlength=7))
    need(len(plan["training_cross_validation"]) == 3 and all(len(x) == 4 for x in plan["training_cross_validation"]),
         "Unexpected case folds")
    need(sorted(sum(plan["training_cross_validation"], [])) == sorted(trained["judgments"]), "Training case folds overlap or omit cases")
    grid = trained["grid"]
    need([tuple(row[:3]) for row in grid] == list(product(plan["ridge"], plan["balance"], plan["gains"])), "Unexpected ridge grid")
    ridge, balance, gain, f1, acc = max(grid, key=lambda v: (v[3], v[4], -v[2], v[0], -v[1]))
    need(dict(ridge=ridge, balance=balance, gain=gain) == trained["selection"], "Chosen result is not the frozen training CV winner")
    np.testing.assert_allclose([f1, acc], [trained["heldout_training_macro_f1"], trained["heldout_training_accuracy"]])
    need(set(trained["moves"]) == {"edited", "shuffled"}, "Unexpected programmed arms")
    physical = {}
    matrices = {"native": base}
    for arm in ("edited", "shuffled"):
        path = f"run/fit/{arm}/readout.safetensors"
        raw = archive.read(path)
        declared = trained["moves"][arm]
        w, payload = readout(raw)
        bitwise_changes = w.view("<u4") != base.view("<u4")
        row_ids = np.flatnonzero(bitwise_changes.any(1)).tolist()
        need(row_ids == declared["changed_rows"] == SEVEN, f"{arm} changed rows outside the seven roles or skipped a role")
        need(hash_bytes(raw) == declared["sha256"] and list(w.shape) == declared["shape"], f"{arm} weight bytes disagree with fit lock")
        need(hash_bytes(w[7:].tobytes(order="C")) == declared["nonrole_rows_sha256"], f"{arm} untouched tail hash mismatch")
        need(int((w != base).sum()) == declared["changed_values"], f"{arm} changed-value count mismatch")
        delta = w[:7].astype(np.float64) - base[:7]
        frobenius = float(np.linalg.norm(delta))
        singular = np.linalg.svd(delta, compute_uv=False)
        rank = int(np.count_nonzero(singular > max(singular[0] * 1e-6, 1e-8)))
        np.testing.assert_allclose(frobenius, declared["additive_frobenius"], rtol=1e-10, atol=1e-10)
        need(0 < declared["solve_rank"] <= 6 and 0 < rank <= 6, f"{arm} edit appears outside the centered rank-six family")
        matrices[arm] = w
        physical[arm] = {"archive_member": path, "sha256": hash_bytes(raw), "shape": SHAPE, "f32_payload_bytes": payload,
                         "row_major": True, "changed_rows": row_ids, "changed_coefficients": int(bitwise_changes.sum()),
                         "c_order_tail_sha256": hash_bytes(w[7:].tobytes(order="C")), "frobenius": frobenius, "numerical_rank": rank}
    np.testing.assert_allclose(physical["edited"]["frobenius"], physical["shuffled"]["frobenius"], rtol=2e-6)
    return matrices, physical


def exact_runs(labels):
    result, start = set(), 0
    for label, run in groupby(labels):
        end = start + sum(1 for _ in run)
        result.add((start, end, int(label)))
        start = end
    return result


def counts(gold, guesses):
    matrix = np.zeros((7, 7), dtype=np.int64)
    np.add.at(matrix, (gold, guesses), 1)
    observed, proposed = exact_runs(gold), exact_runs(guesses)
    return matrix, (len(observed & proposed), len(observed), len(proposed))


def micro_tables(matrix, runs):
    correct = np.diagonal(matrix, axis1=-2, axis2=-1)
    denom = matrix.sum(-1) + matrix.sum(-2)
    by_role = np.divide(2 * correct, denom, where=denom != 0, out=np.zeros_like(denom, dtype=float))
    summary = np.stack([by_role.mean(-1), correct.sum(-1) / matrix.sum((-1, -2)), 2 * runs[..., 0] / runs[..., 1:].sum(-1)], axis=-1)
    return summary, by_role


def check_scores(report, predictions, plan, frozen):
    keys = plan["validation_cases"]
    gold = frozen["truth"]["val"]
    need(report["split"] == "validation" and report["cases"] == keys and report["rows"] == 3558, "Invalid score cohort")
    need(report["official_test_opened"] is False, "Unexpected official test flag")
    need(set(predictions) == set(report["scores"]) == set(ARMS), "Missing analysis arm")
    need(predictions["incumbent"] == frozen["incumbent"], "Incumbent weakly matched or changed")
    draw = np.random.default_rng(plan["bootstrap_seed"]).integers(len(keys), size=(plan["bootstrap_draws"], len(keys)))
    totals, samples = {}, {}
    for arm in ARMS:
        need(set(predictions[arm]) == set(keys), f"Missing or extra case in {arm}")
        cases = []
        for case in keys:
            pred = predictions[arm][case]
            need(len(pred) == len(gold[case]) and all(type(v) is int and 0 <= v < 7 for v in pred),
                 f"Missing or invalid role in {arm}/{case}")
            cases.append(counts(gold[case], pred))
        confusion = np.stack([c[0] for c in cases])
        runs = np.array([c[1] for c in cases])
        cm, spans = confusion.sum(0), runs.sum(0)
        numbers, roles = micro_tables(cm, spans)
        samples[arm], _ = micro_tables(confusion[draw].sum(1), runs[draw].sum(1))
        audited = dict(zip(("macro_f1", "accuracy", "span_f1"), numbers.tolist(), strict=True))
        audited.update(confusion=cm.tolist(), per_role_f1=roles.tolist(), exact_spans=int(spans[0]),
                       gold_spans=int(spans[1]), predicted_spans=int(spans[2]), span_precision=float(spans[0] / spans[2]),
                       span_recall=float(spans[0] / spans[1]))
        for name, expected in audited.items():
            np.testing.assert_allclose(report["scores"][arm][name], expected, atol=1e-12, rtol=1e-12, err_msg=f"{arm}/{name}")
        totals[arm] = audited
    for arm, declared in report["paired_95_edited_minus"].items():
        expected = np.quantile(samples["edited"] - samples[arm], [.025, .975], axis=0).T
        np.testing.assert_allclose([declared[k] for k in ("macro_f1", "accuracy", "span_f1")], expected, atol=1e-12, rtol=1e-12)
    need(set(report["paired_95_edited_minus"]) == {"native", "shuffled", "incumbent"}, "Missing comparison interval")
    return totals


def check_blocks(archive, plan, tokens, fit):
    cases = {"train": plan["fit_cases"], "val": plan["validation_cases"]}
    need(set(fit["training_shards"]) == set(cases["train"]), "Wrong training shard case set")
    out = {}
    for split, keys in cases.items():
        blocks, total_tokens, rows = 0, 0, 0
        for case in keys:
            raw = (HERE / "run" / "token-layout" / split / f"{case}.json").read_bytes()
            expected = tokens[split][case]
            need(hash_bytes(raw) == expected["layout_sha256"], f"Changed input token layout {split}/{case}")
            normal = json.loads(raw)["normal"]
            need([i for group in normal for i in group["targets"]] == list(range(plan["rows"][case])), f"Missing row in {split}/{case}")
            need(all(group["option_order"] == SEVEN and len(group["markers"]) == len(group["targets"]) for group in normal),
                 f"Wrong class layout or marker count in {split}/{case}")
            receipt = json.loads(archive.read(f"run/states/{split}/{case}/complete.json"))
            need(receipt["blocks"] == expected["blocks"] == len(normal) and receipt["tokens"] == expected["total_tokens"]
                 and receipt["rows"] == plan["rows"][case], f"Run receipt disagrees with frozen token plan for {split}/{case}")
            if split == "train":
                wanted = {f"{i:04d}.npz" for i in range(len(normal))}
                need(set(fit["training_shards"][case]) == wanted, f"Missing or extra fit shard for {case}")
            blocks += receipt["blocks"]
            total_tokens += receipt["tokens"]
            rows += receipt["rows"]
        out[split] = {"blocks": blocks, "tokens": total_tokens, "rows": rows}
    need(out["val"] == {"blocks": 680, "tokens": 943001, "rows": 3558}, "Wrong full validation totals")
    need(out["train"] == {"blocks": 76, "tokens": 97097, "rows": 357}, "Wrong training totals")
    return out


def sensitivity(report, predictions, gold, plan):
    variants = report["input_sensitivity"]
    if not variants:
        return {"evaluated_cases": 0}
    need(set(variants) == set(plan["validation_cases"]), "Incomplete sensitivity group")
    result = {}
    for case, modes in variants.items():
        layout = read_json(HERE / "run" / "token-layout" / "val" / f"{case}.json")
        targets = layout["normal"][0]["targets"]
        need(set(modes) == set(layout["variants"]), f"Unexpected sensitivity variants for {case}")
        for name, trial in modes.items():
            changed = layout["variants"][name]
            need(trial["targets"] == targets and trial["displayed_role_order"] == changed["option_order"],
                 f"Unexpected sensitivity target or role order for {case}/{name}")
            for arm in ("native", "edited"):
                need(trial["original_" + arm] == [predictions[arm][case][i] for i in targets], f"Canonical {arm} differs in sensitivity control")
            y = np.array([gold[case][i] for i in targets])
            for arm, p in trial["predictions"].items():
                need(len(p) == len(y) and all(type(v) is int and 0 <= v < 7 for v in p), f"Invalid sensitivity predictions for {case}/{name}/{arm}")
                p = np.array(p)
                reference = np.array(trial["original_native" if arm == "native" else "original_edited"])
                same, hit = int(np.count_nonzero(p == reference)), int(np.count_nonzero(p == y))
                np.testing.assert_allclose([same / len(p), hit / len(p)], [trial["agreement"][arm], trial["accuracy"][arm]])
                tally = result.setdefault(name, {}).setdefault(arm, [0, 0, 0])
                tally[0] += same
                tally[1] += hit
                tally[2] += len(p)
    return {"evaluated_cases": len(variants), "by_variant": {k: {a: {"agreement": s / n, "accuracy": h / n, "rows": n}
             for a, (s, h, n) in v.items()} for k, v in result.items()}}


def replay_states(archive, names, backend, predictions, plan, matrices, trained, report):
    validation_diffs, max_error, checked = 0, 0., 0
    for split, keys in (("train", plan["fit_cases"]), ("val", plan["validation_cases"])):
        for case in keys:
            layout = read_json(HERE / "run" / "token-layout" / split / f"{case}.json")["normal"]
            blocks = [f"run/states/{split}/{case}/{i:04d}" for i in range(len(layout))]
            values, logits = [], []
            for name, expected in zip(blocks, layout, strict=True):
                needed = {name + ".npz", name + ".json"}
                need(needed <= names, f"Incomplete state archive for {name}")
                raw = archive.read(name + ".npz")
                receipt = json.loads(archive.read(name + ".json"))
                need(receipt["sha256"] == hash_bytes(raw) and receipt["backend"] == backend and
                     receipt["parameter_source"] == SHARD_SHA and receipt["digest"] == expected["digest"], f"State receipt mismatch for {name}")
                if split == "train":
                    need(receipt["sha256"] == trained["training_shards"][case][PurePosixPath(name).name + ".npz"],
                         f"Frozen training state digest mismatch for {name}")
                with np.load(BytesIO(raw), allow_pickle=False) as arrays:
                    for sql, saved in (("targets", "targets"), ("input_ids", "input_ids"), ("marker_positions", "markers"),
                                       ("option_order", "option_order")):
                        np.testing.assert_array_equal(arrays[sql], expected[saved], err_msg=name)
                    x, z = arrays["hidden"], arrays["logits"]
                n = len(expected["targets"])
                need(x.shape == (n, 1024) and x.dtype == np.float32 and np.isfinite(x).all(), f"Invalid state array for {name}")
                need(z.shape == (n, 7) and z.dtype == np.float32 and np.isfinite(z).all(), f"Invalid native logits for {name}")
                values.append(x)
                logits.append(z)
                checked += 1
            state, native = np.concatenate(values), np.concatenate(logits)
            native_replay = state @ matrices["native"][:7].T
            np.testing.assert_allclose(native_replay, native, atol=.02, rtol=1e-4, err_msg=case)
            if split == "val":
                validation_diffs += np.count_nonzero(native_replay.argmax(1) != native.argmax(1))
                max_error = max(max_error, float(np.max(np.abs(native_replay - native))))
                for arm in ("native", "edited", "shuffled"):
                    produced = native if arm == "native" else state @ matrices[arm][:7].T
                    np.testing.assert_array_equal(produced.argmax(1), predictions[arm][case], err_msg=f"{arm}/{case}")
    need(validation_diffs == report["native_cpu_vs_gpu_argmax_differences"], "Unmatched native replay decisions")
    return {"verified": True, "canonical_blocks": checked, "numpy_native_vs_returned_gpu_argmax_differences": int(validation_diffs),
            "numpy_native_max_absolute_logit_error": max_error}


def audit(archive_path, state_path, plan, frozen, base, out):
    with ZipFile(archive_path) as archive:
        names = members(archive)
        with ZipFile(HERE / "karpo-t4-full-worker.zip") as worker:
            for pinned in ("source.json", "protocol.json", "token-plan.json"):
                need(archive.read(pinned) == worker.read(pinned), f"Remote modified pinned payload {pinned}")
        report = json.loads(archive.read("run/validation.json"))
        prediction_raw = archive.read("run/validation-predictions.json")
        fit_raw = archive.read("run/fit/freeze.json")
        need(report["protocol_sha256"] == PLAN_SHA and report["predictions_sha256"] == hash_bytes(prediction_raw) and
             report["fit_freeze_sha256"] == hash_bytes(fit_raw), "Validation provenance checksums differ")
        backend = json.loads(archive.read("run/backend.json"))
        need(backend["nvidia_smi"] in [["Tesla T4"], ["NVIDIA T4"], ["T4"]] and backend["device_capability"] == [7, 5]
             and backend["compute_dtype"] == "torch.float32" and backend["protocol_sha256"] == PLAN_SHA
             and backend["transformers"] == "5.21.0" and backend["runner_sha256"] == hash_file(HERE / "runner.py"),
             "Execution receipt is inconsistent with the pinned T4 FP32 worker")
        smoke = json.loads(archive.read("run/smoke.json"))
        need(smoke["backend"] == backend and smoke["weight_sha256"] == SHARD_SHA and smoke["readout_sha256"] == READOUT_SHA,
             "Smoke runtime or model identity disagrees")
        need(not report["clipped_rows"], "Unexpected clipping in scored rows")
        fit, predictions = json.loads(fit_raw), json.loads(prediction_raw)
        matrices, physical = check_training(archive, plan, frozen, fit, base)
        coverage = check_blocks(archive, plan, json.loads(archive.read("token-plan.json")), fit)
        scores = check_scores(report, predictions, plan, frozen)
        if state_path:
            with ZipFile(state_path) as states:
                replay = replay_states(states, members(states), backend, predictions, plan, matrices, fit, report)
        elif any(n.startswith("run/states/") and n.endswith(".npz") for n in names):
            replay = replay_states(archive, names, backend, predictions, plan, matrices, fit, report)
        else:
            replay = {"verified": False, "reason": "Only case receipts and predictions supplied, without hidden-state shards"}
        return {**out, "archive_sha256": hash_file(archive_path), "archive_members": len(names), "backend": backend,
                "physical": physical, "scores": scores, "coverage": coverage, "fit_selection": fit["selection"],
                "paired_95_edited_minus": report["paired_95_edited_minus"], "states_replay": replay,
                "sensitivity": sensitivity(report, predictions, frozen["truth"]["val"], plan), "official_test_opened": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archive", type=Path, nargs="?")
    ap.add_argument("--states", type=Path, help="optional separate archive with all run/states shards and sidecars")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    plan, frozen, base, facts = local_inputs()
    result = audit(args.archive, args.states, plan, frozen, base, facts) if args.archive else facts
    encoded = json.dumps(result, indent=2) + "\n"
    if args.report:
        args.report.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
