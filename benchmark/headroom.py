"""Measure how much correcting each kind of mistake could change macro-F1."""
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from audit import NAMES, stats

h = Path(__file__).resolve().parent
data = json.loads((h / "official-audit.json").read_text())["models"]["semantic_neighbors"]
cm = np.array(data["nonblank"]["confusion"])
baseline = stats(cm)

def corrected(matrix, a, b):
    result = matrix.copy()
    for true, other in ((a,b), (b,a)):
        result[true,true] += result[true,other]
        result[true,other] = 0
    return result

pair_only = corrected(cm, 0, 5)
entries = []
for a,b in combinations(range(7), 2):
    fixed = stats(corrected(cm,a,b))
    together = stats(corrected(pair_only,a,b))
    entries.append(dict(roles=[NAMES[a], NAMES[b]], count=int(cm[a,b]+cm[b,a]),
                        oracle_f1=fixed["macro_f1"], gain_points=100*(fixed["macro_f1"]-baseline["macro_f1"]),
                        with_all_none_reasoning_fixed=together["macro_f1"]))
entries.sort(key=lambda row: row["oracle_f1"], reverse=True)
(h / "error-headroom.json").write_text(json.dumps({
    "system": "semantic_neighbors", "rows": baseline["rows"], "measure": "Hypothetical correction on known gold only, retaining every other original prediction.",
    "baseline_f1": baseline["macro_f1"], "pairs": entries
}, indent=2)+"\n")
for e in entries[:9]:
    print(e["roles"],e["count"],round(e["gain_points"],2), round(100*e["with_all_none_reasoning_fixed"],2))
