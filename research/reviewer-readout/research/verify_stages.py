"""Verify returned anchor ablations on the two unchanged confirmation panels."""

import argparse
from hashlib import file_digest
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from fit_anchor import GROUPS, HERE, hash_file, metrics, rank, write


CONFIRMATIONS = ("confirm", "fresh")


def digest(archive, member):
    with archive.open(member) as handle:
        return file_digest(handle, "sha256").hexdigest()


def npz(archive, member):
    with np.load(BytesIO(archive.read(member)), allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}


def matrix(gold, pred):
    return np.bincount(7 * gold.astype(np.int64) + pred, minlength=49).reshape(7, 7)


def summarize(cm):
    true_positives = np.diagonal(cm, axis1=-2, axis2=-1)
    denominators = np.maximum(cm.sum(-1) + cm.sum(-2), 1)
    return np.stack(((2 * true_positives / denominators).mean(-1),
                     true_positives.sum(-1) / cm.sum(axis=(-2, -1)).clip(min=1),
                     cm[..., 0, 5] + cm[..., 5, 0], cm[..., 0, 5], cm[..., 5, 0]), axis=-1)


def case_matrices(panels, name, no_key):
    counts = []
    for group in CONFIRMATIONS:
        bank = panels[group]
        mask = bank["novel"] if no_key else np.ones(len(bank["gold"]), dtype=bool)
        for a, b in zip(bank["offsets"], bank["offsets"][1:]):
            m = mask[a:b]
            counts.append(matrix(bank["gold"][a:b][m], bank[name][a:b][m]))
    return np.stack(counts)


def comparison(a, b, draws):
    rng = np.random.default_rng(8216309)
    resamples = rng.integers(0, len(a), size=(draws, len(a)))
    current, baseline = summarize(a.sum(0)), summarize(b.sum(0))
    interval = np.quantile(summarize(a[resamples].sum(1)) - summarize(b[resamples].sum(1)), [.025, .975], axis=0)
    fields = ("macro_f1", "accuracy", "pair_errors", "none_to_reasoning", "reasoning_to_none")
    return {key: {"absolute": float(current[j]), "reference": float(baseline[j]),
                  "delta": float(current[j] - baseline[j]), "delta_95ci": interval[:, j].tolist()}
            for j, key in enumerate(fields)}


def audit(packet, returned):
    with ZipFile(packet) as p, ZipFile(returned) as r:
        members = r.namelist()
        assert len(members) == len(set(members))
        expected = json.loads(r.read("SHA256SUMS.json"))
        assert set(expected) == set(members) - {"SHA256SUMS.json"}
        for file, sha in expected.items():
            assert digest(r, file) == sha, file
        for file in ("manifest.json", "protocol.json", "t4_review.py", "fit_anchor.py"):
            assert digest(p, file) == expected[file], file
        inputs = json.loads(p.read("manifest.json"))
        t4 = json.loads(r.read("returned/runtime-receipt.json"))
        assert t4["input_manifest_sha256"] == expected["manifest.json"]
        assert t4["head_sha256"] == inputs["files"]["head.safetensors"]
        assert t4["fixed_control"]["reference_differences"] == 0
        assert t4["fixed_control"]["maximum_absolute_score_error"] <= 2e-5
        assert len(t4["shards"]) == len(inputs["shards"])
        for shard in inputs["shards"]:
            key = shard.replace("anchor-", "reviewed-")
            assert digest(p, shard) == inputs["files"][shard] == t4["shards"][key]["input_sha256"]
        freeze = json.loads(r.read("head-fit/freeze.json"))
        result = json.loads(r.read("head-fit/results.json"))
        latest = f"head-fit/moments-{freeze['cases']:03d}.npz"
        progress = json.loads(r.read("head-fit/progress.json"))
        assert progress["moments_sha256"] == expected[latest]
        assert progress["cases"] == freeze["cases"] == 64 * len(inputs["shards"])
        assert progress["rows"] == freeze["rows"] == sum(items["rows"] for items in t4["shards"].values())
        assert freeze["weights_sha256"] == expected["head-fit/expanded-anchors.safetensors"]
        assert result["freeze_sha256"] == expected["head-fit/freeze.json"]
        assert not freeze["official_test_read"] and freeze["device"] == "CPU"
        assert set(result["groups"]) == set(GROUPS)
        choices = {name: rule["selection"] for name, rule in freeze["ridges"].items()}
        choices["doc-semantic"] = freeze["incumbent_selector"]
        assert rank(choices[freeze["selection"]]) == max(map(rank, choices.values()))
        assert freeze["selection"] == result["selection"]
        panels = {}
        for group in GROUPS:
            bank = npz(r, f"head-fit/{group}-predictions.npz")
            source = npz(p, f"validation-{group}.npz")
            for col in ("case_ids", "offsets", "gold", "novel"):
                np.testing.assert_array_equal(bank[col], source[col])
            np.testing.assert_array_equal(bank["doc-semantic"], source["score"].argmax(1))
            for method in choices:
                assert np.isin(bank[method], np.arange(7)).all()
                rows = result["groups"][group]["scores"][method]
                mask = bank["novel"]
                assert rows["all"] == metrics(bank["gold"], bank[method])
                assert rows["novel"] == metrics(bank["gold"][mask], bank[method][mask])
                if group == "select":
                    assert choices[method] == rows
            panels[group] = bank
        return {
            "cases": freeze["cases"], "rows": freeze["rows"], "selection": freeze["selection"],
            "packet_sha256": hash_file(packet), "returned_sha256": hash_file(returned),
            "t4_device": t4["device"], "parity_control": t4["fixed_control"], "confirmation": {},
            "reviewed_rows": sum(x["reviewed"] for x in t4["shards"].values()),
            "t4_review_seconds": sum(x["seconds"] for x in t4["shards"].values()),
        }, panels


def main(uploads, draws, destination):
    summary, views = {}, {}
    for returned in uploads:
        size = int(returned.stem.split("-")[1])
        stage, panels = audit(HERE / f"t4-anchor-{size}.zip", returned)
        summary[size] = stage
        views[size] = panels
        for mode, mask in (("all", False), ("no_exact_train_key", True)):
            comparisons = {}
            for alternative, reference in (("exact-reviewed", "no-anchor"), ("exact-reviewed", "pre-review"),
                                            ("exact-reviewed", "doc-semantic"), ("no-anchor", "doc-semantic"),
                                            (stage["selection"], "doc-semantic")):
                key = alternative + "_versus_" + reference
                a, b = (case_matrices(panels, method, mask) for method in (alternative, reference))
                comparisons[key] = comparison(a, b, draws)
            stage["confirmation"][mode] = comparisons
        print(size, stage["selection"], stage["t4_device"], flush=True)
    between = {}
    if {256, 512}.issubset(views):
        for group in GROUPS:
            for col in ("case_ids", "offsets", "gold", "novel", "doc-semantic"):
                np.testing.assert_array_equal(views[256][group][col], views[512][group][col])
        for mode, mask in (("all", False), ("no_exact_train_key", True)):
            p = case_matrices(views[512], summary[512]["selection"], mask)
            q = case_matrices(views[256], summary[256]["selection"], mask)
            between[mode] = comparison(p, q, draws)
    write(destination, {"protocol": "Pooled 72- and 96-judgment confirmations. Paired judgment percentile intervals.",
                        "draws": draws, "stages": summary, "512_versus_256": between})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("archives", type=Path, nargs="+")
    cli.add_argument("--draws", type=int, default=5000)
    cli.add_argument("--output", type=Path, default=HERE / "verified-comparison.json")
    args = cli.parse_args()
    main(args.archives, args.draws, args.output)
