"""Score paired public predictions and identify which truly align with official rows."""
import csv
import json
from pathlib import Path

import numpy as np

from program_trial import TO_ID, metric, save_json
from weight_edit import ROOT


def inspect(path, official):
    if path.name == "predictions_inlegal.csv":
        with path.open(newline="") as stream:
            widths = [len(r) for r in csv.reader(stream)]
        return {"status": "No paired gold; release contains unlabelled flat predictions",
                "rows_in_file": len(widths), "elements": sum(widths)}
    if path.name == "RhetoricLLaMA_predictions.csv":
        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows = 0
            gold_present = 0
            for row in csv.DictReader(stream):
                rows += 1
                gold_present += bool(row["Label"].strip())
        return {"status": "Free-form generation column. No published decoding and alignment rule applied.",
                "rows_in_file": rows, "rows_with_gold_field": gold_present}
    gold, pred = [], []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if "test_gold" in row:
                g, p = json.loads(row["test_gold"]), json.loads(row["test_pred"])
                if len(g) != len(p):
                    raise ValueError("MTL sequence length mismatch")
                gold.extend(g)
                pred.extend(p)
            else:
                gold.append(int(row["true"]))
                pred.append(int(row["pred"] if "pred" in row else row["predicted"]))
    gold, pred = np.asarray(gold, dtype=int), np.asarray(pred, dtype=int)
    classes = np.unique(np.r_[gold, pred])
    if len(classes) != 7:
        return {"status": "Paired columns contain a different class inventory", "rows": len(gold),
                "ids": classes.tolist()}
    gold, pred = np.searchsorted(classes, gold), np.searchsorted(classes, pred)
    s = metric(gold, pred)
    result = {"rows": len(gold), "macro_f1": s["macro_f1"], "accuracy": s["accuracy"],
              "status": "Scored within-file paired labels", "official_row_alignment_verified": False}
    if len(gold) == len(official):
        cm = np.bincount(7*gold + official, minlength=49).reshape(7, 7)
        mapping = cm.argmax(axis=1)
        if len(np.unique(mapping)) == 7 and np.array_equal(mapping[gold], official):
            result.update(status="All gold labels match official test in row order under a unique label mapping",
                          official_row_alignment_verified=True,
                          released_to_official_ids={str(k): int(v) for k, v in zip(classes, mapping)},
                          per_role_f1=metric(official, mapping[pred])["f1"])
    return result


def main():
    manifest = json.loads((ROOT / "references/all-predictions/verified.json").read_text())
    with (ROOT / "references/official/test.csv").open(newline="", encoding="utf-8-sig") as stream:
        gold = np.array([TO_ID[r["Label"]] for r in csv.DictReader(stream)])
    report = {"metric": "Equal-weight arithmetic mean of F1 for seven roles",
              "known_limitation": "Equal gold sequence verifies label-row order, not independent text identity for prediction files that lack text.",
              "files": []}
    for record in manifest:
        file = ROOT / record["local"]
        result = inspect(file, gold)
        report["files"].append({"file": file.name, "url": record["url"], "sha256": record["sha256"], **result})
        if file.name == "ToInlegalbertpredictions.csv" and "macro_f1" in result:
            report["toin_summary"] = {"coverage": f"{result['rows']:,} public gold/prediction pairs",
                                      "score": f"{result['macro_f1']:.4f}", "method": result["status"]}
        print(file.name, result.get("rows", result.get("rows_in_file")), result.get("macro_f1"), result.get("official_row_alignment_verified"))
    save_json(ROOT / "results/all-published-predictions.json", report)


if __name__ == "__main__":
    main()
