"""Fit the same supervised head with and without Gemini-reviewed authored examples."""
import argparse
import json

import numpy as np

from discriminant_trial import fitted_heads
from encoder import components, features, write_config
from program_trial import get_features, metric, official_sample, paired_document_interval, save_json
from synthetic_review import original_examples
from weight_edit import LABELS, ORIGINAL, ROOT, file_hash, patched_checkpoint, tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encode-only", action="store_true")
    args = parser.parse_args()
    originals = original_examples()
    model, tok, device = components()
    sx = features(model, tok, device, [ex["text"] for ex in originals], "original-synthetic-168-v1")
    if args.encode_only:
        print("Encoded the 168 pre-authored synthetic passages", flush=True)
        return
    reviews = {x["id"]: x for x in (json.loads(s) for s in (ROOT / "synthetic/gemini-reviews.jsonl").read_text().splitlines())}
    if set(reviews) != {ex["id"] for ex in originals}:
        raise ValueError("Complete all planned synthetic-only reviews before fitting")
    keep = [i for i, row in enumerate(originals) if reviews[row["id"]]["agrees_with_original"]]
    accepted_labels = np.array([LABELS.index(originals[i]["intended_role"]) for i in keep])
    gold, _, _ = official_sample("train", 256)
    gx, gy = get_features(model, tok, device, gold, "train-256")
    reference = ROOT / "checkpoints/discriminant-256"
    spec = json.loads((reference / "edit.json").read_text())["selected"]
    combined_x = np.concatenate((gx, sx[keep]))
    combined_y = np.concatenate((gy, accepted_labels))
    shrink, prior, w, b = next(candidate for candidate in fitted_heads(combined_x, combined_y)
                                if candidate[:2] == (spec["shrinkage"], spec["prior_coefficient"]))
    target = ROOT / "checkpoints/augmented-discriminant"
    changes = patched_checkpoint(ORIGINAL, target / "model.safetensors", {"classifier.weight": w, "classifier.bias": b})
    write_config(target)
    info = {"method": "Gold-trained LDA plus original, Gemini-reviewed synthetic examples",
            "fixed_hyperparameters_from_gold_only": spec, "source_sha256": file_hash(ORIGINAL),
            "target_sha256": file_hash(target / "model.safetensors"), "changes": changes,
            "gold_training_rows": len(gy), "synthetic_training_rows": len(keep), "labels": LABELS}
    save_json(target / "edit.json", info)
    gw, gb = (tensor(reference / "model.safetensors", k) for k in ("classifier.weight", "classifier.bias"))
    r = {"protocol": "Both conditions use identical gold examples, encoder, shrinkage and prior coefficient. Accepted original synthetic examples are added once with their author's role label. Gemini reviews are a choice-based consistency screen. Evaluation uses human-labeled LegalSeg sentences.", "checkpoint": info,
         "synthetic_agreement_by_authored_role": {label: {"kept": int((accepted_labels == i).sum()), "reviewed": 24} for i, label in enumerate(LABELS)},
         "official_test_used": False}
    for part in ("select", "confirm"):
        rows, weights, stats = official_sample(part, 160)
        x, y = get_features(model, tok, device, rows, part + "-160")
        base, pred = (x @ gw.T + gb).argmax(1), (x @ w.T + b).argmax(1)
        r[part] = {"stats": stats, "gold_only": metric(y, base, weights), "gold_plus_reviewed": metric(y, pred, weights),
                   "paired_document_95pct_interval_gain": paired_document_interval(rows, y, base, pred, weights)}
    save_json(ROOT / "results/gemini-augmentation-validation.json", r)
    print("Human-gold confirmation macro-F1, gold-only", r["confirm"]["gold_only"]["macro_f1"],
          "gold plus reviewed originals", r["confirm"]["gold_plus_reviewed"]["macro_f1"], "accepted synthetic", len(keep), flush=True)


if __name__ == "__main__":
    main()
