"""Compute official-test error headroom for each unordered role pair."""
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results/sentence-template-memory-bert.json"
DESTINATION = ROOT / "results/sentence-role-pair-headroom.json"
PUBLISHED_TARGET = .770


def macro_f1(counts):
    return float(np.mean(2 * counts.diagonal() / (counts.sum(0) + counts.sum(1))))


def correct(counts, a, b):
    out = counts.copy()
    out[a, a] += out[a, b]
    out[b, b] += out[b, a]
    out[a, b] = out[b, a] = 0
    assert np.array_equal(out.sum(1), counts.sum(1))
    return out


def main():
    data = SOURCE.read_bytes()
    trial = json.loads(data)["splits"]["test"]
    reference = trial["candidate"]
    counts = np.asarray(reference["confusion"], dtype=np.int64)
    names = list(reference["per_role_f1"])
    base = macro_f1(counts)
    assert counts.shape == (7, 7)
    assert counts.sum() == reference["rows"] and counts.trace() == reference["correct"]
    assert np.isclose(base, reference["macro_f1"], atol=1e-12, rtol=0)
    pairs = []
    for a, b in combinations(range(7), 2):
        to_b, to_a = int(counts[a, b]), int(counts[b, a])
        score = macro_f1(correct(counts, a, b))
        pairs.append((score, a, b, to_b, to_a))
    pairs.sort(reverse=True)
    running = counts
    ranked = []
    for score, a, b, to_b, to_a in pairs:
        running = correct(running, a, b)
        ranked.append({"roles": [names[a], names[b]], "directional_errors": [to_b, to_a],
                       "all_pair_errors": to_b + to_a, "macro_f1_if_pair_fixed": score,
                       "cumulative_macro_f1": macro_f1(running)})
    assert np.array_equal(running, np.diag(counts.sum(1)))
    out = {"source": SOURCE.name, "source_sha256": sha256(data).hexdigest(), "rows": int(counts.sum()),
           "total_errors": int(counts.sum() - counts.trace()), "macro_f1": base,
           "published_target": PUBLISHED_TARGET, "ranked": ranked}
    stage = DESTINATION.with_suffix(".part")
    stage.write_text(json.dumps(out, indent=2) + "\n")
    stage.replace(DESTINATION)


if __name__ == "__main__":
    main()
