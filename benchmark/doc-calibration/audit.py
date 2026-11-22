"""Materialize the frozen additive bias in the native decoder and audit the score replay."""

import argparse
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from safetensors.numpy import load, load_file, save_file

import calibrate as c

COHORTS = ("confirm", "fresh")
PATCH = c.HERE / "doc-semantic-bias.safetensors"


def write_patch(lock, archive):
    item = lock["arms"]["bias"]
    member = "doc-semantic.safetensors"
    with ZipFile(c.MAIN_SOURCE) as z:
        primary = z.read(member)
    source = c.read(c.PLAN)
    with ZipFile(c.MAIN_SOURCE) as z:
        frozen = z.read("freeze.json")
        assert sha256(frozen).hexdigest() == source["parent_frozen_sha256"]
        import json
        assert sha256(primary).hexdigest() == json.loads(frozen)["files"][member]
    original = load(primary)
    p = load_file(c.HERE / "bias.safetensors")
    assert p["output.weight"].shape == (0, 7)
    delta = (lock["temperature"] * item["params"][-1]) * p["output.bias"]
    patch = {k: v.copy() for k, v in original.items()}
    patch["coef"][-1] += delta
    save_file(patch, str(PATCH))
    installed = load_file(PATCH)
    actual = installed["coef"][-1] - original["coef"][-1]
    assert np.array_equal(installed["coef"][:-1], original["coef"][:-1])
    changed = {}
    for k in original:
        changed[k] = int(np.count_nonzero(original[k] != installed[k]))
    assert changed["coef"] == 7 and sum(changed.values()) == 7
    rng = np.random.default_rng(761361)
    example = rng.normal(size=(160, len(original["coef"]))).astype(np.float32)
    example[:, -1] = 1.
    left, right = example @ installed["coef"], example @ original["coef"]
    np.testing.assert_allclose(left - right, np.broadcast_to(actual, left.shape), atol=8e-7, rtol=0)
    replay = {}
    for group in ("select", *COHORTS):
        b = c.load_bank(archive, group)
        q = (b.scores + actual).argmax(1)
        path = c.HERE / (group + "-predictions.npz")
        with np.load(path, allow_pickle=False) as f:
            np.testing.assert_array_equal(b.keys, f["case_ids"])
            np.testing.assert_array_equal(q, f["bias"])
        replay[group] = {"rows": len(q), "prediction_sha256": c.digest(path), "disagreements": 0}
    c.write(c.HERE / "native-audit.json", {
        "source_decoder_sha256": sha256(primary).hexdigest(), "patched_decoder_sha256": c.digest(PATCH),
        "changes": changed, "edited_tensor_row": -1, "delta": actual.tolist(),
        "decoder_affine_forward_max_abs_error": float(np.max(np.abs((left - right) - actual))),
        "score_bank_replays": replay,
        "verification_scope": "Native final affine operator checked on deterministic feature rows. Complete case labels checked using exported exact parent scores and the serialized bias difference.",
    })


def pooled():
    arms = c.read(c.HERE / "results.json")["groups"]["confirm"]["arms"]
    a, b = [np.load(c.HERE / (g + "-predictions.npz"), allow_pickle=False) for g in COHORTS]
    try:
        keys = np.r_[a["case_ids"], b["case_ids"]]
        ends = np.r_[a["offsets"], b["offsets"][1:] + a["offsets"][-1]]
        gold = np.r_[a["gold"], b["gold"]]
        novel = np.r_[a["novel"], b["novel"]]
        assert len(set(keys.tolist())) == len(keys) == 168
        bank = c.Bank("combined", keys, ends, gold, None, None, None, novel, None)
        old = np.r_[a["base"], b["base"]]
        outcome = {}
        for name in arms:
            p = np.r_[a[name], b[name]]
            scores = c.stats(bank, p)
            scores.update(
                paired_all=c.interval(bank, old, p, np.ones(len(p), bool)),
                paired_novel=c.interval(bank, old, p, novel),
                fixed=int(np.count_nonzero((p == gold) & (old != gold))),
                broken=int(np.count_nonzero((p != gold) & (old == gold))),
                changed=int(np.count_nonzero(p != old)),
            )
            outcome[name] = scores
        c.write(c.HERE / "pooled-confirmation.json", {
            "judgments": len(keys), "sentences": len(gold), "no_exact_training_key": int(novel.sum()),
            "freeze_sha256": c.digest(c.FROZEN), "cohorts": COHORTS, "scores": outcome,
            "evidence": "Paired judgment bootstrap across both complete validation cohorts.",
        })
        for name in ("base", "bias", "case-affine", "case-experts", "pair-case"):
            r = outcome[name]
            print(name, r["all"]["macro_f1"], r["novel"]["macro_f1"], r["all"]["pair_errors"],
                  r["novel"]["pair_errors"], r["paired_all"]["macro_f1"], flush=True)
    finally:
        a.close()
        b.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=c.LEGAL / "calibration-banks.zip")
    args = parser.parse_args()
    lock = c.read(c.FROZEN)
    write_patch(lock, args.archive)
    pooled()


if __name__ == "__main__":
    main()
