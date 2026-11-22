"""Measure which validation errors the previously frozen reviewer can reach."""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
INPUT = HERE.parent / "verified/query-results"
ROLES = ("None", "Facts", "Issue", "Petitioner argument", "Respondent argument", "Reasoning", "Decision")


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def evaluate(gold, pred):
    cm = np.bincount(gold.astype(int)*7 + pred.astype(int), minlength=49).reshape(7, 7)
    f1 = 2*np.diag(cm)/(cm.sum(0)+cm.sum(1)).clip(min=1)
    return {"macro_f1": float(f1.mean()), "accuracy": float((gold == pred).mean()),
            "per_role_f1": dict(zip(ROLES, f1.tolist())),
            "confusion": cm.tolist(), "correct": int((gold == pred).sum())}


def main():
    records = {}
    for name in ("confirm", "fresh"):
        path = INPUT / (name + "-predictions.npz")
        with np.load(path, allow_pickle=False) as data:
            gold, base, review = data["gold"], data["incumbent"], data["review"]
            cases = data["case_ids"]
            assert len(set(cases)) == len(cases)
            mask = np.zeros(len(gold), bool)
            mask[review] = True
            assert int(mask.sum()) == len(review)
            pair = ((gold == 0) & (base == 5)) | ((gold == 5) & (base == 0))
            variants = {"baseline": np.zeros(len(gold), bool),
                        "all_routed_correct": mask,
                        "routed_none_reasoning_correct": mask & pair,
                        "all_none_reasoning_correct": pair}
            scores = {}
            for key, positions in variants.items():
                pred = base.copy()
                pred[positions] = gold[positions]
                scores[key] = evaluate(gold, pred)
            records[name] = {"judgments": len(cases), "sentences": len(gold), "routed": len(review),
                             "none_reasoning_errors": int(pair.sum()),
                             "routed_none_reasoning_errors": int((mask & pair).sum()),
                             "wrong_routed": int((mask & (gold != base)).sum()),
                             "source_sha256": digest(path), "scores": scores}
    official_path = HERE.parents[2]/"results/sentence-wide-bert-benchmark.json"
    old = json.loads(official_path.read_text())["splits"]["test"]["comparison"]["routes"]["confidence"]["labels"]
    matrix = np.asarray(old["confusion"], dtype=np.int64)
    repaired = matrix.copy()
    for actual, wrong in ((0, 5), (5, 0)):
        repaired[actual, actual] += repaired[actual, wrong]
        repaired[actual, wrong] = 0
    def from_matrix(cells):
        f1 = 2*np.diag(cells)/(cells.sum(0)+cells.sum(1)).clip(min=1)
        return {"macro_f1": float(f1.mean()), "accuracy": float(cells.trace()/cells.sum()),
                "correct": int(cells.trace()), "per_role_f1": dict(zip(ROLES, f1.tolist()))}
    official = {"source": "Previously recorded aggregate test confusion matrix. No new test CSV access.",
                "source_sha256": digest(official_path), "baseline": from_matrix(matrix),
                "all_none_reasoning_correct": from_matrix(repaired),
                "none_reasoning_errors": int(matrix[0, 5]+matrix[5, 0])}
    path = HERE / "routing-oracle.json"
    path.write_text(json.dumps({"hypothetical": "Reference labels are used only to calculate ceilings. These are not deployable models.",
                                "other_predictions_held_fixed": True,
                                "other_cohorts": "Validation cases have been inspected in earlier project work.",
                                "groups": records, "official_aggregate": official}, indent=2)+"\n")
    print(path)
    for name, item in records.items():
        print(name, item["routed_none_reasoning_errors"], item["none_reasoning_errors"],
              {k:round(v["macro_f1"]*100,3) for k,v in item["scores"].items()})
    print("official aggregate", official["none_reasoning_errors"], official["all_none_reasoning_correct"])


if __name__ == "__main__":
    main()
