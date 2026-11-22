"""Verify serialized random-partition inference and paired validation outcomes."""

import argparse
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from safetensors.numpy import load_file

from forest_runtime import NONE, REASONING, predict
from run_forest import CONT, HERE, SEED, bank, corrections, evaluation, protocol, read, write


COHORTS = ("confirm", "fresh")


def pair_error(y, p):
    return ((y == NONE) & (p == REASONING)) | ((y == REASONING) & (p == NONE))


def paired_total(b, orig, new, scope):
    d = (pair_error(b.gold, new).astype(np.int32) - pair_error(b.gold, orig)) * scope
    per_case = np.add.reduceat(d, b.ends[:-1])
    idx = np.random.default_rng(SEED + 415).integers(len(per_case), size=(5000, len(per_case)))
    return np.quantile(per_case[idx].sum(1), (.025, .975)).tolist()


def independent_replay(archive, spec, freeze):
    data = {}
    normalization = {}
    for name, arm in freeze["arms"].items():
        if "files" not in arm:
            continue
        prototypes = {kind: load_file(HERE / item["file"]) for kind, item in arm["files"].items()}
        chosen = prototypes["edited"]
        for key, values in chosen.items():
            if key != "leaf.weight":
                for variant in ("zero", "shuffled"):
                    np.testing.assert_array_equal(prototypes[variant][key], values)
        assert np.all(prototypes["zero"]["leaf.weight"] == 0)
        norm = {kind: float(np.linalg.norm(p["leaf.weight"].astype(np.float64))) for kind, p in prototypes.items()}
        assert np.isclose(norm["edited"], norm["shuffled"], rtol=2e-7)
        normalization[name] = {"shared_partition_sha256": sha256(b"".join(v.tobytes() for k, v in sorted(chosen.items()) if k != "leaf.weight")).hexdigest(),
                               "shape": list(chosen["leaf.weight"].shape), "weights_written": int(np.count_nonzero(chosen["leaf.weight"])),
                               "norm": norm}
    for group in ("select", *COHORTS):
        b = bank(archive, group, spec)
        source = HERE / (group + "-predictions.npz")
        proof = {"rows": len(b.gold), "cases": len(b.keys), "file_sha256": evaluation.digest(source), "methods": {}}
        with np.load(source, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["zero"], b.base)
            np.testing.assert_array_equal(saved["case_ids"], b.keys)
            np.testing.assert_array_equal(saved["offsets"], b.ends)
            np.testing.assert_array_equal(saved["gold"], b.gold)
            for method, cfg in freeze["arms"].items():
                if "files" not in cfg:
                    continue
                opts = cfg["params"]
                recipe = (freeze["temperature"], opts["gain"], opts["minimum_gap"], opts["pair_only"])
                for variant, item in cfg["files"].items():
                    params = load_file(HERE / item["file"])
                    name = {"edited": method, "shuffled": method + "-shuffled", "zero": "zero"}[variant]
                    new, _ = predict(b.scores, b.experts, b.old, b.ends, params, *recipe)
                    np.testing.assert_array_equal(new, saved[name])
                    proof["methods"][name] = {"predictions_reproduced": len(new), "weight_file_sha256": item["sha256"]}
                    if variant == "edited":
                        for jj in (0, len(b.keys) // 2, len(b.keys) - 1):
                            a, z = b.ends[jj:jj + 2]
                            same, _ = predict(b.scores[a:z], b.experts[a:z], b.old[a:z], np.array([0, z - a]), params, *recipe)
                            np.testing.assert_array_equal(new[a:z], same)
                        proof["methods"][name]["fixed_case_isolation_checks"] = 3
                        proof["methods"][name]["all_pair_errors_delta_95"] = paired_total(b, b.base, new, np.ones(len(new), bool))
                        proof["methods"][name]["no_key_pair_errors_delta_95"] = paired_total(b, b.base, new, b.novel)
        data[group] = proof
    return {"replayed": data, "parameter_checks": normalization, "official_test_read": False}


def pooled():
    src_a, src_b = (HERE / (prefix + "-predictions.npz") for prefix in COHORTS)
    stage = {}
    with np.load(src_a, allow_pickle=False) as a, np.load(src_b, allow_pickle=False) as b:
        def concat(name):
            return np.r_[a[name], b[name]]

        assert len(set(concat("case_ids"))) == len(concat("case_ids")) == 168
        bank = SimpleNamespace(gold=concat("gold"), ends=np.r_[a["offsets"], a["offsets"][-1] + b["offsets"][1:]], novel=concat("novel"))
        old = concat("zero")
        for name in a.files:
            if name in ("case_ids", "offsets", "gold", "novel"):
                continue
            p = concat(name)
            results = evaluation.stats(bank, p)
            for key, mask in (("all", np.ones(len(p), bool)), ("novel", bank.novel)):
                results[key]["switch_outcomes"] = corrections(bank.gold, old, p, mask)
                results[key]["paired_95"] = {**evaluation.interval(bank, old, p, mask),
                                              "pair_errors": paired_total(bank, old, p, mask)}
            stage[name] = results
    return {"cases": len(bank.ends) - 1, "rows": len(bank.gold), "no_key_rows": int(bank.novel.sum()),
            "methods": stage, "pooling": "Descriptive joint summary after separate confirmation results."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=CONT / "legal-context/calibration-banks.zip")
    args = parser.parse_args()
    scope = protocol(args.bank)
    frozen = read(HERE / "freeze.json")
    write(HERE / "runtime-audit.json", independent_replay(args.bank, scope, frozen))
    data = pooled()
    write(HERE / "pooled-confirmation.json", data)
    for name, result in data["methods"].items():
        print(name, result["all"]["macro_f1"], result["novel"]["macro_f1"], result["all"]["pair_errors"], result["novel"]["pair_errors"])
