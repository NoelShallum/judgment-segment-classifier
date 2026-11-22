"""Score the already frozen excerpt heads against the original Gemini answer ledgers."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

ROUND = Path(__file__).resolve().parent
sys.path.insert(0, str(ROUND / "excerpt"))
from edit import (GEMINI, HERE, LABELS, ORIGINAL, ROOT, Tail, eval_ledger, get_group,
                  paired, read, sha, verify_protocol, write)
from head_followup import load_first_edit, read_followup, score_group


def main():
    files = SimpleNamespace(data=HERE / "capture-input", features=HERE / "excerpt-captured.zip",
                            donor=ORIGINAL, initial=HERE / "results", output=HERE / "followup")
    plan, state = verify_protocol(files.data, files.features, HERE / "frozen-capture-protocol.json")
    initial, _ = load_first_edit(files.initial)
    extra = read_followup(files)
    tail = Tail(ORIGINAL)
    earlier = read(files.initial / "report.json")["groups"]
    result, arrays = {}, {}
    for name in plan["archived_panels"]:
        ledger = eval_ledger(ROOT, plan, name)
        group = get_group(files.data, files.features, plan, state, name, ledger)
        stats, preds = score_group(group, tail, initial, extra)
        api = np.array([LABELS.index(ledger[r["id"]]["predictions"][GEMINI]) for r in group.rows])
        for method, old in (("untouched_donor", "untouched_donor"), ("classifier_only_balanced", "classifier_refit"),
                            ("ffn_only_balanced", "role_pairs")):
            assert stats["methods"][method] == earlier[name]["methods"][old]
        assert (group.labels == api).sum() == earlier[name]["methods"][GEMINI]["correct"]
        result[name] = {"rows": len(api), "judgments": stats["judgments"], "methods": stats["methods"],
                        "gemini": earlier[name]["methods"][GEMINI],
                        "vs_gemini": {m: paired(group, api, preds[m]) for m in stats["methods"]}}
        arrays.update({name + "/" + k: v for k, v in (preds | {"gemini": api}).items()})
        print(name, {k: v["correct"] for k, v in stats["methods"].items()}, "Gemini", int((group.labels == api).sum()))
    predictions = ROUND / "followup-panel-predictions.npz"
    np.savez_compressed(predictions, **arrays)
    write(ROUND / "followup-panels.json", {"followup_freeze_sha256": sha(files.output / "freeze.json"),
          "summary": "Frozen models on the same cached Gemini Decisions inputs. No new requests. Secondary analysis.",
          "predictions_sha256": sha(predictions), "panels": result})


if __name__ == "__main__":
    main()
