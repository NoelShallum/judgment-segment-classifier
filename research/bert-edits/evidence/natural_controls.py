"""Replay the other already-frozen excerpt weights on the same natural sample."""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
from safetensors.numpy import load_file

ROUND = Path(__file__).resolve().parent
sys.path.insert(0, str(ROUND / "excerpt"))
from edit import HERE, ORIGINAL, ROOT, Group, Tail, paired, patch_forward, read, score, sha, write
from head_followup import new_sample
from audit_sampling import scan


def main():
    files = SimpleNamespace(natural_data=HERE / "natural-input", natural_features=HERE / "natural-captured.zip",
                            data=HERE / "capture-input", initial=HERE / "results")
    group, _, _ = new_sample(files)
    old_path = ROOT / "checkpoints/associative-mlp/model.safetensors"
    assert sha(old_path) == read(old_path.with_name("edit.json"))["target_sha256"]
    old = Tail(old_path)
    base = Tail(ORIGINAL)
    for key in ("wi","bi","wp","bp","gamma","beta"):
        np.testing.assert_array_equal(getattr(base, key), getattr(old, key))
    act, residual, _ = base.baseline(group)
    frozen = read(files.initial / "freeze.json")
    preds = {"older_excerpt_mlp": group.predict(old.class_scores(old.pool(old.residue(group.hidden, act))[0]))}
    for name, receipt in frozen["weights"].items():
        path = files.initial / (name + ".safetensors")
        assert sha(path) == receipt["sha256"]
        logits, _ = patch_forward(base, group.hidden, act, residual, load_file(str(path)))
        preds[name] = group.predict(logits)
    with np.load(HERE / "followup/predictions.npz", allow_pickle=False) as previous:
        assert list(previous["natural/id"]) == [r["id"] for r in group.rows]
        np.testing.assert_array_equal(previous["natural/gold"], group.labels)
        np.testing.assert_array_equal(previous["natural/ffn_only_balanced"], preds["role_pairs"])
        for name in ("untouched_donor", "combined_prevalence_weighted", "classifier_only_prevalence_weighted"):
            preds[name] = previous["natural/" + name]
    stats = {key: score(group, v) for key, v in preds.items()}
    comparisons = {key: paired(group, preds["untouched_donor"], v) for key, v in preds.items()}
    out = ROUND / "natural-controls-predictions.npz"
    np.savez_compressed(out, **preds, gold=group.labels, id=np.array([r["id"] for r in group.rows]))
    _, earlier_rows, earlier_cases = scan(ROOT, {"fit": [], "select": [], "confirm": []})
    mask = np.array([r["case"] not in earlier_cases for r in group.rows])
    rows = [r for r, keep in zip(group.rows, mask) if keep]
    gold = group.labels[mask]
    limited = Group("natural_disjoint_from_older_selector", rows, gold, np.arange(len(rows)), np.ones(len(rows)),
                    np.empty(0), np.empty(0), np.ones(len(rows)))
    sensitivity = {"rows": len(rows), "judgments": len({r["case"] for r in rows}),
                   "original_row_overlap": sum(r["official_row"] in earlier_rows for r in group.rows),
                   "methods": {k: score(limited, pred[mask]) for k, pred in preds.items()}}
    write(ROUND / "natural-controls.json", {"study": "Secondary replay of previously fixed weights on the same 4,900 sentences.",
          "predictions_sha256": sha(out), "weights_freeze_sha256": sha(files.initial / "freeze.json"),
          "excluding_historical_selector_cases": sensitivity,
          "groups": {"natural": {"methods": stats, "paired_vs_donor": comparisons}}})
    for key,v in stats.items():
        print(key, round(100*v["macro_f1"], 3), round(100*v["accuracy"], 3), comparisons[key]["macro_f1_ci95"])


if __name__ == "__main__":
    main()
