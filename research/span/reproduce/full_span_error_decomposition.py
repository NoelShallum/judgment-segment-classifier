"""Audit where the two released weights miss exact LegalSeg spans."""
from collections import Counter, defaultdict
from bisect import bisect_left
import numpy as np
from safetensors.numpy import load_file

from discourse_span_edit import text_features
from hier_full import CACHE, gold_cases
from hier_model import CRF, LABELS, path_labels
from hier_trial import OUT
from program_trial import save_json
from seven_paths import ROOT
from span_eval import document
from span_model import merge

CATEGORIES = ("exact", "unreachable_endpoint", "exact_bounds_wrong_role",
              "same_role_one_unit_away", "same_role_other_overlap", "no_same_role_overlap")


def describe(d, prediction):
    offsets = {a: i for i, (a, b) in enumerate(d["units"])}
    offsets[len(d["text"])] = len(d["units"])
    boundaries = {(s["start"], s["end"]): s["label"] for s in prediction}
    predicted_roles = {role: [s for s in prediction if s["label"] == role] for role in LABELS}
    starts = {role: [s["start"] for s in sections] for role, sections in predicted_roles.items()}
    errors = {role: Counter(dict.fromkeys(CATEGORIES, 0)) for role in LABELS}
    exact_wrong = Counter()
    for s in d["gold_spans"]:
        left, right, role = s["start"], s["end"], s["label"]
        actual = boundaries.get((left, right))
        if actual == role:
            key = "exact"
        elif left not in offsets or right not in offsets:
            key = "unreachable_endpoint"
        elif actual is not None:
            key = "exact_bounds_wrong_role"
            exact_wrong[role,actual] += 1
        else:
            ls = starts[role]
            j = max(0, bisect_left(ls, left) - 1)
            nearby = False
            overlap = False
            for pos in range(j, bisect_left(ls, right)):
                p = predicted_roles[role][pos]
                start, end = p["start"], p["end"]
                if end <= left:
                    continue
                overlap = True
                if (abs(offsets[start]-offsets[left]) <= 1 and abs(offsets[end]-offsets[right]) <= 1):
                    nearby = True
                    break
            key = "same_role_one_unit_away" if nearby else "same_role_other_overlap" if overlap else "no_same_role_overlap"
        errors[role][key] += 1
    assert sum(k.total() for k in errors.values()) == len(d["gold_spans"])
    return errors, exact_wrong


def main():
    shares = {k: load_file(str(OUT / f"whole-discourse-{k}.safetensors")) for k in ("exact", "balanced")}
    tally = {k: defaultdict(Counter) for k in shares}
    missed = {k: Counter() for k in shares}
    predicted = {k: Counter() for k in shares}
    for i, (case, rows) in enumerate(gold_cases().items(), 1):
        d = document(case, rows)
        cues, position = text_features(d)
        length = len(d["text"].split())
        group = "short" if length < 2000 else "medium" if length < 8000 else "long"
        with np.load(CACHE / (case+".npz")) as ar:
            raw = ar["raw"]
        for k,p in shares.items():
            z = raw + cues @ p["cue_head"] + position @ p["position_head"]
            decoded = merge(d["units"], path_labels(z, p[CRF]), LABELS)
            result, mistake = describe(d, decoded)
            missed[k].update(mistake)
            predicted[k].update(s["label"] for s in decoded)
            for role, audit in result.items():
                tally[k][role].update(audit)
                tally[k]["length_"+group].update(audit)
        if i % 150 == 0: print("Scored", i, "judgments", flush=True)
    profiles = {}
    for k,v in tally.items():
        roles = {r: dict(v[r], predicted=predicted[k][r]) for r in LABELS}
        whole = Counter()
        for r in LABELS: whole.update(v[r])
        profiles[k] = {"all_spans": dict(whole), "by_role": roles,
                       "by_length": {a.removeprefix("length_"): dict(b) for a,b in v.items() if a.startswith("length_")},
                       "common_fixed_boundary_mistakes": [(*pair, n) for pair,n in missed[k].most_common()[:12]]}
        print(k, dict(whole), flush=True)
    save_json(ROOT / "results/current-discourse-span-error-decomposition.json",
              {"cases": i, "profiles": profiles, "interpretation": "Disjoint gold-span accounting. A near miss has the same role, overlaps the gold span, and both endpoints are within one detected unit of the available gold endpoints. This does not estimate attainable gains after editing."})


if __name__ == "__main__":
    main()
