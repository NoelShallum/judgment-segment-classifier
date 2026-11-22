"""Train a compact CPU benchmark on official training text and select on validation."""
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time

import fasttext
import numpy as np

from weight_edit import ROOT, LABELS, file_hash
from import_official import FILES


ROLE_NAMES = ("None", "Facts", "Issue", "Arguments of Petitioner",
              "Arguments of Respondent", "Reasoning", "Decision")
TOKEN = re.compile(r"[^\W_]+|[^\w\s]")
HYPERPARAMETERS = dict(dim=48, wordNgrams=2, bucket=250000, minn=0, maxn=0,
                      epoch=5, lr=0.3, minCount=3, loss="softmax", thread=2)
ADJUSTMENTS = (0.0, 0.25, 0.5, 0.75, 1.0)
TARGET = ROOT / "checkpoints" / "lexical"


def normalize(text):
    return " ".join(TOKEN.findall(text.casefold())) or "emptyexcerpt"


def verified_rows(split):
    path = ROOT / "references" / "official" / f"{split}.csv"
    if file_hash(path) != FILES[path.name][1]:
        raise ValueError(f"Unexpected checksum for {split}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def fit():
    TARGET.mkdir(parents=True, exist_ok=True)
    model_path = TARGET / "model.bin"
    if model_path.is_file() and (TARGET / "fit.json").is_file():
        return fasttext.load_model(str(model_path))
    rows, documents = [], set()
    counts = np.zeros(len(ROLE_NAMES), dtype=int)
    for row in verified_rows("train"):
        label = ROLE_NAMES.index(row["Label"])
        rows.append(f"__label__{label} {normalize(row['Text'])}\n")
        counts[label] += 1
        documents.add(row["Index"])
    random.Random(19337).shuffle(rows)
    training_file = ROOT / "cache" / "lexical-train-v1.txt"
    with training_file.open("w", encoding="utf-8") as stream:
        stream.writelines(rows)
    del rows
    start = time.monotonic()
    print(f"Training on {int(counts.sum())} excerpts from {len(documents)} cases", flush=True)
    model = fasttext.train_supervised(input=str(training_file), verbose=0, **HYPERPARAMETERS)
    temporary = TARGET / "model.partial.bin"
    model.save_model(str(temporary))
    os.replace(temporary, model_path)
    info = {"parameters": HYPERPARAMETERS, "shuffle_seed": 19337, "train_rows": int(counts.sum()),
            "train_documents": len(documents), "train_counts": counts.tolist(), "elapsed_s": round(time.monotonic()-start),
            "model_bytes": model_path.stat().st_size, "sha256": file_hash(model_path)}
    (TARGET / "fit.json").write_text(json.dumps(info, indent=2) + "\n")
    print("Trained", info, flush=True)
    return model


def scores(model, text):
    # Use the C++ binding. Its pairs are sorted by probability.
    pairs = model.f.predict(normalize(text) + "\n", 7, 0.0, "strict")
    result = np.full(7, 1e-12)
    for probability, label in pairs:
        result[int(label.removeprefix("__label__"))] = max(probability, 1e-12)
    return np.log(result)


def score_split(model, split):
    scores_by_row, targets, documents = [], [], []
    for row in verified_rows(split):
        scores_by_row.append(scores(model, row["Text"]))
        targets.append(ROLE_NAMES.index(row["Label"]))
        documents.append(row["Index"])
    return np.asarray(scores_by_row), np.asarray(targets), documents


def main():
    model = fit()
    # Import only after training so torch is absent from the training working set.
    from program_trial import fold, metric, save_json
    logits, y, documents = score_split(model, "val")
    info = json.loads((TARGET / "fit.json").read_text())
    prior = np.log(np.array(info["train_counts"]) / info["train_rows"])
    use_for_selection = np.array([fold(doc) != 0 for doc in documents])
    dev = []
    for alpha in ADJUSTMENTS:
        predicted = (logits - alpha * prior).argmax(axis=1)
        dev.append({"adjustment": alpha, "metrics": metric(y[use_for_selection], predicted[use_for_selection])})
    selected = max(dev, key=lambda entry: entry["metrics"]["macro_f1"])
    alpha = selected["adjustment"]
    params = {"adjustment": alpha, "train_log_prior": prior.tolist(), "labels": LABELS, "source_sha256": info["sha256"]}
    save_json(TARGET / "inference.json", params)
    pred = (logits - alpha * prior).argmax(axis=1)
    report = {"fit": info, "selection_rows": int(use_for_selection.sum()), "selection": dev,
              "selected": alpha, "confirmation_rows": int((~use_for_selection).sum()),
              "confirmation_documents": len(set(d for i, d in enumerate(documents) if not use_for_selection[i])),
              "confirmation": metric(y[~use_for_selection], pred[~use_for_selection]),
              "full_validation": metric(y, pred), "official_test_used": False}
    save_json(ROOT / "results" / "lexical-validation.json", report)
    print("Selection", selected["metrics"]["macro_f1"], "confirmation", report["confirmation"]["macro_f1"], flush=True)


if __name__ == "__main__":
    main()
