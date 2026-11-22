"""Audit archived pre-review labels and build checksummed training-only T4 inputs."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from prepare_anchor import CONT, FIELDS, HERE, OLD, OLD_POOLS, ROUND2, SHARDS, read, save, sha, write


def group_load(group):
    proof = read(OLD_POOLS / "pool-receipt.json")
    assert all("T4" in name for name in proof["device"]["devices"])
    path = OLD_POOLS / (group + ".npz")
    assert sha(path) == proof["groups"][group]["file_sha256"]
    with np.load(path, allow_pickle=False) as arr:
        names, ends, review_ends, refs, scores = (arr[k] for k in ("case_ids", "offsets", "review_offsets", "reference", "scores"))
    cases_dir = ROUND2 / "new-cases" if group == "fresh" else OLD / "cases"
    plan = read(ROUND2 / "round2-plan.json")
    np.testing.assert_array_equal(names, plan["groups"][group])
    return names, ends, review_ends, refs, scores, cases_dir, plan


def metrics(y, p):
    matrix = np.bincount(7*y.astype(np.int64) + p, minlength=49).reshape(7, 7)
    hit = matrix.diagonal()
    per_role = 2 * hit / (matrix.sum(0) + matrix.sum(1)).clip(min=1)
    return {"macro_f1": float(per_role.mean()), "accuracy": float(hit.sum() / matrix.sum()),
            "none_to_reasoning": int(matrix[0, 5]), "reasoning_to_none": int(matrix[5, 0]),
            "pair_errors": int(matrix[0, 5] + matrix[5, 0]), "confusion": matrix.tolist()}


def proxy():
    outcomes = {}
    for group in ("fit", "select", "confirm", "fresh"):
        keys, offsets, _, final, _, folder, plan = group_load(group)
        pre, truth = [], []
        for key, a, b in zip(keys, offsets[:-1], offsets[1:]):
            with np.load(folder / (key + ".npz"), allow_pickle=False) as row:
                assert row["case_sha256"].item() == plan["case_text_sha256"][key]
                assert len(row["base"]) == len(row["gold"]) == b-a
                assert np.isin(np.flatnonzero(row["base"] != final[a:b]), row["positions"]).all()
                pre.append(row["base"])
                truth.append(row["gold"])
        naive, gold = np.concatenate(pre), np.concatenate(truth)
        outcomes[group] = {"judgments": len(keys), "rows": len(final), "label_changes": int(np.count_nonzero(naive != final)),
                           "same_label_fraction": float(np.mean(naive == final)), "pre_review": metrics(gold, naive),
                           "published_original_reviewer": metrics(gold, final)}
        print(group, outcomes[group]["same_label_fraction"], outcomes[group]["pre_review"]["macro_f1"],
              outcomes[group]["published_original_reviewer"]["macro_f1"], flush=True)
    write(HERE / "pre-review-proxy.json", {"cohorts": outcomes, "official_test_read": False})


def fixed_control():
    keys, offsets, review_offsets, refs, scores, folder, plan = group_load("select")
    refs_out, scores_out, inputs = [], [], []
    picks = (0, len(keys) // 2, len(keys) - 1)
    for i in picks:
        key = keys[i]
        with np.load(folder / (key + ".npz"), allow_pickle=False) as raw:
            assert raw["case_sha256"].item() == plan["case_text_sha256"][key]
            inputs.append({k: raw[k] for k in FIELDS})
        refs_out.append(refs[offsets[i]:offsets[i + 1]])
        scores_out.append(scores[review_offsets[i]:review_offsets[i + 1]])
    ends = np.r_[0, np.cumsum([len(x["base"]) for x in inputs])]
    positions = np.concatenate([start + d["positions"] for start, d in zip(ends[:-1], inputs)])
    joined = {k: np.concatenate([d[k] for d in inputs]) for k in inputs[0] if k != "positions"}
    save(HERE / "control.npz", **joined, positions=positions, reference=np.concatenate(refs_out),
         reference_scores=np.concatenate(scores_out), offsets=ends, case_ids=keys[list(picks)])


def validation():
    archive = CONT / "legal-context/calibration-banks.zip"
    for group in ("select", "confirm", "fresh"):
        keys, offsets, _, final, _, folder, plan = group_load(group)
        before = []
        for key in keys:
            with np.load(folder / (key + ".npz"), allow_pickle=False) as row:
                assert row["case_sha256"].item() == plan["case_text_sha256"][key]
                before.append(row["base"])
        name = "scorebank-" + group + ".npz"
        with ZipFile(archive) as reader:
            payload = reader.read(name)
            assert sha256(payload).hexdigest() == json.loads(reader.read("manifest.json"))[name]
        with np.load(BytesIO(payload), allow_pickle=False) as scored:
            np.testing.assert_array_equal(scored["case_ids"], keys)
            np.testing.assert_array_equal(scored["offsets"], offsets)
            np.testing.assert_array_equal(scored["old"], final)
            gold, score = scored["gold"], scored["score"]
        with np.load(CONT / "scale" / (group + "-mask.npz"), allow_pickle=False) as mask:
            np.testing.assert_array_equal(keys, mask["case_ids"])
            np.testing.assert_array_equal(offsets, mask["offsets"])
            novel = mask["novel"]
        save(HERE / ("validation-" + group + ".npz"), case_ids=keys, offsets=offsets, pre=np.concatenate(before),
             reviewed=final, gold=gold, score=score, novel=novel)
    pinned_basis = CONT / "scale/stage-512.zip"
    with ZipFile(pinned_basis) as z:
        basis = sha256(z.read("basis.safetensors")).hexdigest()
        assert basis == json.loads(z.read("SHA256SUMS.json"))["basis.safetensors"]
    write(HERE / "fit-source.json", {"scale_basis_sha256": basis, "reference_scale_stage_sha256": sha(pinned_basis),
                                    "fit_code_sha256": sha(HERE / "fit_anchor.py"), "coupled_prior": "Order-preserving incumbent label alignment across compact features. Selection uses 48 prior validation cases and includes the legacy context reference. Score 72 and 96 only after the model and mixture freeze."})


def pack(blocks):
    fixed_control()
    validation()
    paths = {name: HERE / name for name in ("control.npz", "protocol.json", "t4_review.py", "fit_anchor.py", "fit-source.json",
                                           "validation-select.npz", "validation-confirm.npz", "validation-fresh.npz")}
    paths.update({name: OLD / name for name in ("native.py", "fit.py", "bert-config.json", "review.json", "donor-classifier.npz", "head.safetensors")})
    members = []
    for k in range(blocks):
        group = SHARDS / f"anchor-{k:02d}.npz"
        meta = group.with_suffix(".json")
        assert sha(group) == read(meta)["file_sha256"] and read(meta)["protocol_sha256"] == sha(HERE / "protocol.json")
        paths[group.name], paths[meta.name] = group, meta
        members.append(group.name)
    inventory = {name: sha(source) for name, source in paths.items()}
    release = HERE / f"t4-anchor-{64*blocks}.zip"
    stage = release.with_suffix(".part")
    with ZipFile(stage, "w", ZIP_DEFLATED, compresslevel=3) as output:
        for name, source in paths.items():
            output.write(source, name)
        output.writestr("manifest.json", json.dumps({"files": inventory, "shards": members}, indent=2) + "\n")
    with ZipFile(stage) as output:
        assert output.testzip() is None
    stage.replace(release)
    print(release, release.stat().st_size, sha(release), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("proxy", "pack"))
    ap.add_argument("--blocks", type=int, choices=range(1, 9), default=4)
    args = ap.parse_args()
    if args.action == "proxy":
        proxy()
    else:
        pack(args.blocks)
