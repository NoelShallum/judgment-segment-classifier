"""Measure train-context overlap on every nonblank official validation sentence."""

from collections import Counter, defaultdict

import numpy as np

import exact_memory as m


def main():
    p = m.table()
    recipe = m.read(m.FREEZE)["arms"]["all-seven"]["parameters"]
    by_width = defaultdict(Counter)
    hit = Counter()
    seen = []
    for key, text, gold in m.cases("val"):
        seen.append(key)
        values = m.features(text, p)
        count, _, _, full_duplicate, literal = values
        estimated, used, support, _ = m.propose(np.zeros_like(gold), values, "all-seven", recipe)
        accepted = used > 0
        hit.update({
            "rows": len(gold), "exact_complete_training_judgment_rows": int(full_duplicate.sum()),
            "any_normalized_training_key": int((count.sum((1, 2)) > 0).sum()),
            "accepted_key": int(accepted.sum()), "accepted_correct": int((accepted & (estimated == gold)).sum()),
            "accepted_with_raw_target_sentence_seen": int((accepted & literal).sum()),
            "accepted_key_seen_in_one_distinct_case": int((accepted & (support == 1)).sum()),
            "accepted_key_seen_in_multiple_distinct_cases": int((accepted & (support > 1)).sum()),
        })
        for width in m.WIDTHS:
            part = used == width
            by_width[width].update({"rows": int(part.sum()), "correct": int((part & (estimated == gold)).sum())})
    output = {"judgments": len(seen), **hit, "by_width": by_width,
              "frozen_memory_policy_sha256": m.digest(m.FREEZE),
              "validation_csv_sha256": m.digest(m.CSV / "val.csv"),
              "purpose": "Passage overlap and accepted-key label agreement on all validation judgments. The study uses the existing locked policy and does not select another candidate."}
    m.write(m.HERE / "full-validation-overlap.json", output)
    print(output, flush=True)


if __name__ == "__main__":
    main()
