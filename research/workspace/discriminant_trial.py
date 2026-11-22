"""Fit a seven-way closed-form head using labeled training examples and a frozen encoder."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.linalg import solve

from encoder import components, write_config
from program_trial import get_features, metric, official_sample, paired_document_interval, save_json
from weight_edit import LABELS, ORIGINAL, ROOT, TRAIN_COUNTS, file_hash, patched_checkpoint, tensor


SHRINKAGES = (0.03, 0.15, 0.45, 0.80)
PRIOR_COEFFICIENTS = (0.0, 0.25, 0.50, 0.75, 1.0)


def fitted_heads(x, labels):
    x = x.astype(np.float64)
    means = np.stack([x[labels == i].mean(0) for i in range(7)])
    center = means.mean(0)
    means -= center
    residuals = x - center - means[labels]
    covariance = residuals.T @ residuals / (len(x) - len(means))
    scale = covariance.trace() / covariance.shape[0]
    log_prior = np.log(TRAIN_COUNTS / TRAIN_COUNTS.sum())
    for shrinkage in SHRINKAGES:
        regularized = (1 - shrinkage) * covariance
        regularized.flat[::regularized.shape[0]+1] += shrinkage * scale
        w = solve(regularized, means.T, assume_a="pos").T
        intercept = -0.5 * np.sum(means * w, axis=1) - w @ center
        for coefficient in PRIOR_COEFFICIENTS:
            b = intercept + coefficient * log_prior
            yield shrinkage, coefficient, w.astype(np.float32), b.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-training-role", type=int, default=256)
    args = parser.parse_args()
    train, _, train_stats = official_sample("train", args.per_training_role)
    select, select_weights, select_stats = official_sample("select", 160)
    confirm, confirm_weights, confirm_stats = official_sample("confirm", 160)
    if {r[1] for r in train} & {r[1] for r in select + confirm}:
        raise ValueError("Training and validation share document IDs")
    model, tokenizer, device = components()
    x, yt = get_features(model, tokenizer, device, train, f"train-{args.per_training_role}")
    xs, ys = get_features(model, tokenizer, device, select, "select-160")
    candidates, best, winner = [], -np.inf, None
    for shrinkage, prior, w, b in fitted_heads(x, yt):
        report = metric(ys, (xs @ w.T + b).argmax(1), select_weights)
        spec = {"shrinkage": shrinkage, "prior_coefficient": prior}
        candidates.append({**spec, "metrics": report})
        if report["macro_f1"] > best:
            best, winner = report["macro_f1"], (spec, w, b)
    spec, w, b = winner
    target = ROOT / "checkpoints" / f"discriminant-{args.per_training_role}"
    changes = patched_checkpoint(ORIGINAL, target / "model.safetensors",
                                 {"classifier.weight": w, "classifier.bias": b})
    write_config(target)
    info = {"method": "Closed-form shrinkage LDA with labeled encoder activations",
            "selected": spec, "labels": LABELS, "changes": changes,
            "source_sha256": file_hash(ORIGINAL), "target_sha256": file_hash(target / "model.safetensors")}
    save_json(target / "edit.json", info)
    xc, yc = get_features(model, tokenizer, device, confirm, "confirm-160")
    original = (xc @ tensor(ORIGINAL, "classifier.weight").T + tensor(ORIGINAL, "classifier.bias")).argmax(1)
    preds = (xc @ w.T + b).argmax(1)
    results = {"train": train_stats, "selection": select_stats, "confirmation": confirm_stats,
               "candidates": candidates, "selected": info, "selected_development_macro_f1": best,
               "confirmation_original": metric(yc, original, confirm_weights),
               "confirmation_selected": metric(yc, preds, confirm_weights),
               "paired_document_95pct_interval_gain": paired_document_interval(confirm, yc, original, preds, confirm_weights),
               "official_test_used": False}
    save_json(ROOT / "results" / f"discriminant-{args.per_training_role}-validation.json", results)
    print(json.dumps({"validation": best, "confirmation": results["confirmation_selected"]["macro_f1"],
                      "selected": spec, "train_rows": len(train)}), flush=True)


if __name__ == "__main__":
    main()
