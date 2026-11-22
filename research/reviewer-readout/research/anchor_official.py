"""Lock the expanded anchor model using both selectors, then replay the matched official rows."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from safetensors.numpy import load_file


ROOT = Path("/content/reviewer2")


def assemble_lock(root):
    stages = {}
    for size in (256, 512):
        folder = root / f"anchor-scale-{size}"
        assert (folder / "run.exit").read_text().strip() == "0"
        inventory = folder / "manifest.json"
        fit = folder / "head-fit"
        freeze_path, score_path = fit / "freeze.json", fit / "results.json"
        weights = fit / "expanded-anchors.safetensors"
        t4_path = folder / "returned/runtime-receipt.json"
        t4, freeze, scores = (read(p) for p in (t4_path, freeze_path, score_path))
        arm = freeze["selection"]
        assert t4["input_manifest_sha256"] == hash_file(inventory)
        assert t4["fixed_control"]["reference_differences"] == 0
        assert t4["fixed_control"]["maximum_absolute_score_error"] <= 2e-5
        assert len(t4["device"]["devices"]) == 1 and "T4" in t4["device"]["devices"][0]
        assert freeze["cases"] == size and len(t4["shards"]) * 64 == size
        assert freeze["weights_sha256"] == hash_file(weights)
        assert not freeze["official_test_read"] and scores["freeze_sha256"] == hash_file(freeze_path)
        assert freeze["protocol_sha256"] == hash_file(folder / "protocol.json")
        assert scores["selection"] == arm and scores["groups"]["select"]["judgments"] == 48
        grid = {n: x["selection"] for n, x in freeze["ridges"].items()}
        grid["doc-semantic"] = freeze["incumbent_selector"]
        assert grid == scores["groups"]["select"]["scores"]
        assert rank(grid[arm]) == max(map(rank, grid.values()))
        result = read(fit / "results.json")
        assert result["freeze_sha256"] == hash_file(freeze_path)
        stages[str(size)] = {
            "selected_arm": arm, "quality": list(rank(grid[arm])),
            "freeze_sha256": hash_file(freeze_path), "results_sha256": hash_file(score_path),
            "weight_sha256": hash_file(weights), "basis_sha256": freeze["basis_sha256"],
            "protocol_sha256": freeze["protocol_sha256"], "runtime_receipt_sha256": hash_file(t4_path),
            "device": t4["device"], "selector": scores["groups"]["select"]["scores"][arm],
            "selector_source_sha256": hash_file(folder / "validation-select.npz"),
        }
    assert stages["256"]["protocol_sha256"] == stages["512"]["protocol_sha256"]
    assert stages["256"]["basis_sha256"] == stages["512"]["basis_sha256"]
    assert stages["256"]["selector_source_sha256"] == stages["512"]["selector_source_sha256"]
    winner = max(stages, key=lambda n: stages[n]["quality"])
    return {"chosen_cases": int(winner), "chosen_arm": stages[winner]["selected_arm"], "stages": stages,
            "selection_rule": "Maximize the recorded 48-judgment selector tuple. The shorter prefix wins exact ties.",
            "official_selection": False, "gradient_steps": 0}


def active_design(compact, offsets, anchor, plan, columns):
    from run_scale import design
    from solve import docbasis, probabilities

    basis = docbasis(design(compact, offsets), plan).astype(np.float32)
    if anchor is None:
        np.testing.assert_array_equal(columns, np.arange(basis.shape[1]))
        return basis
    p = .5 * (probabilities(compact[:, :7] * .65) + probabilities(compact[:, 62:69] * .65))
    extra = anchored_features(SimpleNamespace(offsets=offsets), anchor, p)
    shift = extra.shape[1] if columns[-1] >= basis.shape[1] + extra.shape[1] else 0
    np.testing.assert_array_equal(columns, np.r_[np.arange(basis.shape[1]), np.arange(basis.shape[1] + shift, basis.shape[1] + shift + extra.shape[1])])
    return np.column_stack((basis, extra))


def scores(x, off, old, anchor, basis, weights, rule):
    if weights is None:
        return old.argmax(1)
    col, coef = weights
    p = np.empty(len(x), np.int8)
    for group in range(0, len(off) - 1, 16):
        ends = off[group:group + 17]
        a, b = ends[0], ends[-1]
        block = active_design(x[a:b], ends - a, None if anchor is None else anchor[a:b], basis, col)
        mixture = (1. - rule) * old[a:b] + rule * (block @ coef)
        p[a:b] = mixture.argmax(1)
    return p


def case_comparison(y, off, p, ref):
    from verify_stages import matrix, summarize

    def tally(labels):
        return np.stack([matrix(y[a:b], labels[a:b]) for a, b in zip(off, off[1:])])
    x, z = tally(p), tally(ref)
    rng = np.random.default_rng(18409)
    result = []
    for _ in range(40):
        draw = rng.integers(0, len(x), size=(100, len(x)))
        result.extend(summarize(x[draw].sum(1)) - summarize(z[draw].sum(1)))
    fields = ("macro_f1", "accuracy", "pair_errors", "none_to_reasoning", "reasoning_to_none")
    delta = summarize(x.sum(0)) - summarize(z.sum(0))
    interval = np.quantile(result, [.025, .975], axis=0)
    return {k: {"delta": float(delta[i]), "delta_95ci": interval[:, i].tolist()} for i, k in enumerate(fields)}


def replay(root, destination, lock, lock_path, pre_review):
    import run_scale

    size, arm = lock["chosen_cases"], lock["chosen_arm"]
    fit_dir = root / f"anchor-scale-{size}" / "head-fit"
    freeze = read(fit_dir / "freeze.json")
    assert hash_file(fit_dir / "freeze.json") == lock["stages"][str(size)]["freeze_sha256"]
    weight = fit_dir / "expanded-anchors.safetensors"
    scale_basis = root / "scale/basis.safetensors"
    assert hash_file(weight) == freeze["weights_sha256"] and hash_file(scale_basis) == freeze["basis_sha256"]
    plan = load_file(scale_basis)
    if arm == "doc-semantic":
        operands, mixture = None, None
    else:
        matrices = load_file(weight)
        operands = matrices[arm + ".columns"], matrices[arm + ".coef"]
        mixture = freeze["ridges"][arm]["settings"][2]

    file = root / "scale/test-panel.npz"
    if file.is_file():
        compact = checked_npz(file)
        x, off, y, case = (compact[k] for k in ("x", "offsets", "y", "case"))
    else:
        x, off, y, case = run_scale.fixed("test")
    old_scores, original = run_scale.legacy("test")
    for col, match in ((case, original.case_ids), (off, original.offsets), (y, original.gold)):
        np.testing.assert_array_equal(col, match)
    assert len(case) == 712 and len(y) == 149_868
    incumbent, semantic = original.base, old_scores.argmax(1)
    check = metrics(y, incumbent)
    assert (check["none_to_reasoning"], check["reasoning_to_none"]) == (11_323, 14_089)
    assert abs(check["macro_f1"] - .6439163434451753) < 1e-12
    assert abs(metrics(y, semantic)["macro_f1"] - .6722498650307156) < 1e-12
    if arm == "pre-review":
        if pre_review is None:
            raise ValueError("The selected pre-review arm requires the independently verified historical pre-review NPZ")
        pre = checked_npz(pre_review)
        for name, match in (("case_ids", case), ("offsets", off), ("gold", y), ("incumbent", incumbent)):
            np.testing.assert_array_equal(pre[name], match)
        anchor = pre["pre"]
    else:
        anchor = incumbent if arm == "exact-reviewed" else None
    current = scores(x, off, old_scores, anchor, plan, operands, mixture)
    preds = destination / "official-predictions.npz"
    np.savez_compressed(preds, case_ids=case, offsets=off, gold=y, incumbent=incumbent, doc_semantic=semantic, selected=current)
    report = {
        "lock_sha256": hash_file(lock_path), "weights_sha256": hash_file(weight), "basis_sha256": hash_file(scale_basis),
        "chosen_cases": size, "chosen_arm": arm, "rows": len(y), "judgments": len(case), "matched_nonblank": True,
        "historical_official_prior_use": True, "predictions_sha256": hash_file(preds),
        "metrics": {n: metrics(y, p) for n, p in (("incumbent", incumbent), ("doc-semantic", semantic), ("selected", current))},
        "pairs_95": {"doc-semantic": case_comparison(y, off, current, semantic), "incumbent": case_comparison(y, off, current, incumbent)},
    }
    record = destination / "official-results.json"
    write(record, report)
    files = (lock_path, preds, record, fit_dir / "freeze.json", weight, scale_basis, Path(__file__),
             Path(fit_anchor.__file__), Path(verifier.__file__))
    archive = destination.with_name("anchor-official-results.zip")
    staged = archive.with_suffix(".part")
    with ZipFile(staged, "w", ZIP_DEFLATED) as out:
        for f in files:
            out.write(f, f.name)
        out.writestr("SHA256SUMS.json", json.dumps({f.name: hash_file(f) for f in files}))
    staged.replace(archive)
    print(archive, hash_file(archive), json.dumps(report["metrics"]["selected"]), flush=True)


if __name__ == "__main__":
    flags = argparse.ArgumentParser(description=__doc__)
    flags.add_argument("--root", default=ROOT, type=Path)
    flags.add_argument("--pre-review", type=Path)
    flags.add_argument("--freeze-only", action="store_true")
    args = flags.parse_args()
    r = args.root
    for d in (Path(__file__).resolve().parent, r / "scale", r / "legal", r / "context", r / "anchor-scale-512"):
        sys.path.insert(0, str(d))
    import fit_anchor
    import verify_stages as verifier
    from fit_anchor import anchored_features, checked_npz, hash_file, metrics, rank, read, write
    output = r / "anchor-official"
    output.mkdir(exist_ok=True)
    lock_file = output / "official-lock.json"
    locked = assemble_lock(r)
    if lock_file.exists():
        assert read(lock_file) == locked
    else:
        write(lock_file, locked)
    print("Locked on 48 cases", locked["chosen_cases"], locked["chosen_arm"], hash_file(lock_file), flush=True)
    if not args.freeze_only:
        replay(r, output, locked, lock_file, args.pre_review)
