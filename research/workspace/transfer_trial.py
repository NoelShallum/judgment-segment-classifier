"""Measure the frozen heads on a three-role cross-corpus slice in India and the UK."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from zipfile import ZipFile

import numpy as np

from encoder import components, features
from program_trial import metric, save_json
from weight_edit import LABELS, ORIGINAL, ROOT, tensor


MAPPING = {"FAC": "facts", "Ratio": "reasoning", "RPC": "decision"}


def sample(country):
    archive = ROOT / "references/marro" / f"{country}-test-set.zip"
    groups = defaultdict(list)
    counts = Counter()
    with ZipFile(archive) as data:
        for name in sorted(data.namelist()):
            if not name.endswith(".txt") or name.split("/")[-1].startswith("."):
                continue
            for line_no, line in enumerate(data.read(name).decode("utf-8-sig").splitlines()):
                if not line.strip():
                    continue
                text, local_label = line.rsplit("\t", 1)
                local_label = local_label.strip()
                counts[local_label] += 1
                if local_label in MAPPING:
                    y = LABELS.index(MAPPING[local_label])
                    groups[y].append((name, line_no, text.strip(), y))
    rng = random.Random(45261 if country == "IN" else 45262)
    rows = []
    for group in groups.values():
        rows.extend(rng.sample(group, min(80, len(group))))
    rows.sort()
    num = Counter(r[3] for r in rows)
    weights = np.array([len(groups[r[3]]) / num[r[3]] for r in rows])
    return rows, weights, counts


def evaluate(y, preds, weights):
    score = metric(y, preds, weights)
    roles = tuple(MAPPING.values())
    supported = {LABELS.index(r) for r in roles}
    return {"macro_f1_on_mapped_gold_roles": float(np.mean([score["f1"][r] for r in roles])),
            "accuracy_on_mapped_slice": score["accuracy"],
            "prediction_outside_mapped_roles": float(np.average([p not in supported for p in preds], weights=weights)),
            "per_role_f1": {r: score["f1"][r] for r in roles},
            "per_role_recall": {r: score["recall"][r] for r in roles},
            "confusion_rows_and_columns": LABELS, "weighted_confusion": score["confusion"]}


def main():
    model, tokenizer, device = components()
    direct = ROOT / "checkpoints/selected-direct/model.safetensors"
    heads = {name: (tensor(path, "classifier.weight"), tensor(path, "classifier.bias"))
             for name, path in (("original", ORIGINAL), ("direct_edit", direct))}
    report = {"source": "subinay612/MARRO_Rhetorical-Role-Labeling",
              "revision": "668bc2d187d934f8e3cb71c15b887459ded347f4", "mapping": MAPPING,
              "design": "Up to 80 sentences per mapped gold class per country. Weighted to the actual prevalence of these three source classes. All seven output labels remained available. Other gold roles excluded. No model selection on this corpus.",
              "limitations": "MARRO ratio is a narrower annotation than LegalSeg reasoning. Gold for arguments lacks party identity. This slice cannot estimate seven-role or paragraph accuracy. Identity overlap between Indian corpora has not been verified.", "countries": {}}
    for country in ("IN", "UK"):
        rows, weights, counts = sample(country)
        x = features(model, tokenizer, device, [r[2] for r in rows], "marro-map3-v1-" + country)
        y = np.asarray([r[3] for r in rows])
        report["countries"][country] = {"all_source_counts": dict(counts), "sampled": len(rows),
                                        "sampled_documents": len({r[0] for r in rows}),
                                        "eligible_mapped_rows": sum(counts[k] for k in MAPPING),
                                        "observed_sample_counts": {LABELS[i]: int((y == i).sum()) for i in sorted(set(y))},
                                        "models": {name: evaluate(y, (x @ w.T + b).argmax(1), weights)
                                                   for name, (w, b) in heads.items()}}
        print(country, {n: m["macro_f1_on_mapped_gold_roles"] for n, m in report["countries"][country]["models"].items()}, flush=True)
    save_json(ROOT / "results/marro-transfer.json", report)


if __name__ == "__main__":
    main()
