"""Trace the flat hierarchy gold column and identify the split for the role-definition file."""

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import string


ROLES = ("None", "Facts", "Issue", "Arguments of Petitioner", "Arguments of Respondent", "Reasoning", "Decision")
ROLE_ID = {role: i for i, role in enumerate(ROLES)}
PUNCTUATION = str.maketrans(string.punctuation, " " * len(string.punctuation))
DATASET = "https://huggingface.co/datasets/L-NLProc/"
CORPUS = "LegalSeg_CSV/resolve/31217aa845866bbd18a8ad567a785e6f5485435a/"
HIERARCHY = "LegalSeg_Hier_BiLSTM_CRF_Predictions/resolve/929608e0cd47d37151017952d7858ce5bfd648b0/"
CI = "LegalSeg_Role_Aware_Predictions/resolve/352acfeb650da788aaf7acf2c396aeed5b166ee9/"
FILES = {
    "test": ("official/test.csv", CORPUS + "test.csv", "5e1a2f9a03fd7581d4585ef97f55787d058afbed063b5e03b6e3611cbe45f9ee"),
    "val": ("official/val.csv", CORPUS + "val.csv", "8abb2d789325418fc5ab4186c0dc63798e02b8fd82f9e00c324185e6bd86f816"),
    "hierarchy_named": ("baseline_document/predictions.txt", HIERARCHY + "predictions.txt",
                        "9865f9b471f9a407446cfcb7c5bc05ebb2b7ab01e7e5deeb50543c3ad56e3e1a"),
    "hierarchy_flat": ("baseline_document/Hier_BiLSTM_predictions.csv", HIERARCHY + "Hier_BiLSTM_predictions.csv",
                       "f3b6ad51c8eb9b6b525e39e7cd7d6be4dfa876a1893ed735b378fa87cba40440"),
    "role_definitions": ("all-predictions/LegalSeg_Role_Aware_Predictions/role-aware_label_definitions_predictions.csv",
                         CI + "role-aware_label_definitions_predictions.csv",
                         "4e2849caa1a3effe83eda75f2a40d86d29edef494738bdf3b8c621da1f4dcaf9"),
}


def rows(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        yield from csv.DictReader(stream)


def score(gold, prediction):
    matrix = [[0] * 7 for _ in ROLES]
    for truth, output in zip(gold, prediction, strict=True):
        matrix[truth][output] += 1
    tp = [matrix[i][i] for i in range(7)]
    f1 = [2 * tp[i] / (sum(matrix[i]) + sum(r[i] for r in matrix)) for i in range(7)]
    return {"accuracy": sum(tp) / len(gold), "macro_f1": sum(f1) / len(f1)}


def hierarchy(test_path, named_path, flat_path):
    official, keep = bytearray(), {}
    for row in rows(test_path):
        key, text, role = int(row["Index"]), row["Text"], ROLE_ID[row["Label"]]
        official.append(role)
        keep.setdefault(key, bytearray())
        if "\t" not in text and text.lower().translate(PUNCTUATION).split():
            keep[key].append(role)
    named = {}
    for line in named_path.read_text(encoding="utf-8").splitlines():
        filename, sequence = line.split("\t", 1)
        key = int(filename.removeprefix("file_")) - 1
        if key in named:
            raise ValueError(f"Repeated named prediction for case {key}")
        named[key] = bytes(ROLE_ID[x] for x in sequence.rstrip(",").split(","))
    if keep.keys() != named.keys() or any(len(keep[k]) != len(named[k]) for k in keep):
        raise ValueError("Named and filtered official cases have different keys or lengths")
    if list(keep) != sorted(keep):
        raise ValueError("The official test is not in ascending case order")
    flat_true, flat_pred = bytearray(), bytearray()
    for row in rows(flat_path):
        flat_true.append(int(row["True"]))
        flat_pred.append(int(row["Predicted"]))
    filtered = b"".join(keep[k] for k in sorted(keep))
    if flat_true != official[:len(flat_true)] or flat_pred != b"".join(named[k] for k in sorted(named)):
        raise ValueError("The specific prefix or sorted-case prediction identity failed")
    mismatches = [i for i, (a, b) in enumerate(zip(filtered, flat_true, strict=True)) if a != b]
    return {
        "official_cases": len(keep), "official_all_rows": len(official), "rows_in_both_prediction_streams": len(flat_pred),
        "excluded_by_original_text_filter": len(official) - len(filtered),
        "named_case_id_rule": "Subtract one from file_NNNN and sort by numeric official case ID.",
        "named_vectors_match_flat_predictions_exactly": True,
        "canonical_predicted_ids_sha256": hashlib.sha256(flat_pred).hexdigest(),
        "flat_gold_matches_official_prefix_exactly": True, "flat_gold_length": len(flat_true),
        "different_gold_positions_vs_filtered": len(mismatches), "first_different_position_zero_based": mismatches[0],
        "flat_gold_as_distributed": score(flat_true, flat_pred), "case_and_row_filter_corrected_gold": score(filtered, flat_pred),
        "interpretation": "The flat CSV pairs filtered predictions with a prefix of the unfiltered official gold stream. Gold rows removed by the original text filter occur throughout the corpus. The flat and named files contain the same complete prediction stream in ascending official case order.",
    }


def definitions_split(val_path, definitions_path):
    gold, pred = bytearray(), bytearray()
    for released, official in itertools.zip_longest(rows(definitions_path), rows(val_path)):
        if released is None or official is None:
            raise ValueError("Unequal validation and released row counts")
        label = (int(released["true"]) + 1) % 7
        if label != ROLE_ID[official["Label"]]:
            raise ValueError("Released label sequence differs from official validation")
        gold.append(label)
        pred.append((int(released["pred"]) + 1) % 7)
    return {"official_split": "validation", "matched_gold_rows": len(gold), "all_gold_rows_match_in_order": True,
            "gold_sequence_sha256": hashlib.sha256(gold).hexdigest(), "on_validation": score(gold, pred),
            "qualification": "The release supplies numeric gold and predicted labels without sentence text."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, default=Path(__file__).resolve().parents[1] / "references")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths, sources = {}, {}
    for key, (relative, url, digest) in FILES.items():
        path = args.references / relative
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != digest:
            raise ValueError(f"Snapshot checksum mismatch for {relative}")
        paths[key] = path
        sources[key] = {"sha256": actual, "url": DATASET + url}
    result = {"sources": sources, "hierarchy": hierarchy(paths["test"], paths["hierarchy_named"], paths["hierarchy_flat"]),
              "role_aware_label_definitions": definitions_split(paths["val"], paths["role_definitions"])}
    serialized = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
