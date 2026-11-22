"""Audit a pre-frozen checkpoint on a uniformly sampled official-test subset."""
from pathlib import Path
import random

import numpy as np

from encoder import components, features
from program_trial import metric, paired_document_interval, save_json
from score_official_test import DIRECT_HASH, official
from weight_edit import ORIGINAL, ROOT, file_hash, tensor


SAMPLE_SIZE = 1792
SEED = 2026112187


def main():
    direct = ROOT / "checkpoints/selected-direct/model.safetensors"
    assert file_hash(direct) == DIRECT_HASH
    rows, targets, released = official()
    indices = sorted(random.Random(SEED).sample(range(len(rows)), SAMPLE_SIZE))
    sample = [rows[i] for i in indices]
    y, comparison = targets[indices], released[indices]
    model, tokenizer, device = components()
    x = features(model, tokenizer, device, [r[2] for r in sample], f"fixed-test-audit-{SEED}-{SAMPLE_SIZE}")
    def run(path):
        return (x @ tensor(path, "classifier.weight").T + tensor(path, "classifier.bias")).argmax(1)
    base, pred = run(ORIGINAL), run(direct)
    record = {"split": "uniform sample of official test.csv without replacement", "sampling_seed": SEED,
              "rows": len(sample), "judgments": len({r[1] for r in sample}),
              "protocol": "Row IDs were chosen by a fixed pseudorandom draw independent of labels and predictions. The checkpoint was frozen on validation before this interim evaluation. Final test evaluation covers every row separately.",
              "selected_checkpoint_sha256": DIRECT_HASH, "published_release": metric(y, comparison),
              "original_reloaded": metric(y, base), "edited": metric(y, pred),
              "original_replay_matches_published_fraction": float(np.mean(base == comparison)),
              "paired_95pct_gain_vs_replayed": paired_document_interval(sample, y, base, pred, None),
              "paired_95pct_gain_vs_release": paired_document_interval(sample, y, comparison, pred, None)}
    save_json(ROOT / "results/direct-interim-test.json", record)
    print("rows", len(sample), "release", record["published_release"]["macro_f1"], "reloaded",
          record["original_reloaded"]["macro_f1"], "edited", record["edited"]["macro_f1"], flush=True)


if __name__ == "__main__":
    main()
