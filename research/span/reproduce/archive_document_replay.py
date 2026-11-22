"""Replay seven-role LegalSeg projection references using only the Python standard library.

Run with --references pointing at the judgment-role references directory. That directory
contains official/test.csv and the source repositories inside all-predictions/.
"""

import argparse
from collections import Counter
import csv
from dataclasses import dataclass, field
import hashlib
import itertools
import json
from pathlib import Path
from urllib.parse import quote


ROLES = ("None", "Facts", "Issue", "Arguments of Petitioner", "Arguments of Respondent", "Reasoning", "Decision")
ROLE_IDS = {name: index for index, name in enumerate(ROLES)}
FIRST_LINES = {str(i): i for i in range(len(ROLES))}
FIRST_LINES.update({f"{name}-{i}": i for i, name in enumerate(ROLES)})
SOURCES = {
    "official": ("LegalSeg_CSV", "31217aa845866bbd18a8ad567a785e6f5485435a", "test.csv",
                 "5e1a2f9a03fd7581d4585ef97f55787d058afbed063b5e03b6e3611cbe45f9ee"),
    "mtl": ("LegalSeg_MTL_Predictions", "7e17d45d33ce5d447961ff9f4b5ba1afd7d8efba", "predictions.csv",
            "3d31adebfb9ba95ca2c3640fb6b5db7bd31fe2c0ddcaa5a079956afd8f93e5ba"),
    "role_aware_ci": ("LegalSeg_Role_Aware_Predictions", "352acfeb650da788aaf7acf2c396aeed5b166ee9",
                      "predictions_CI.csv", "c0bf506a8d83d113c3ee37d25c8a6d2bdadc4f46db7c28204581cc6d1b893822"),
    "previous_predicted": ("LegalSeg_InLegalBERT_Predictions", "e9233636e75dfa221dd18d14c4ecec66942fda13",
                           "InLegalBert(i-1, label_p, i).csv", "1de4dabd8e5591ec0db2d469de66f7b5af35f06bb1915dfe1f02a7dbfd7fedd1"),
    "previous_gold": ("LegalSeg_InLegalBERT_Predictions", "e9233636e75dfa221dd18d14c4ecec66942fda13",
                      "InLegalBert(i-1, label_t, i)_predictions.csv", "7092d69e650ce0d12814e912b955e19665dbd6769dda957e0e47420893ec4212"),
    "rhetoricllama": ("LegalSeg_RhetoricLLaMA_Predictions", "686d8082bdadc4f08cf21ff068b561ac38a88cc4",
                     "RhetoricLLaMA_predictions.csv", "f4a366643ab5b9edcee38e83549b5c88ff1511fa461b7e018ebaae65711ba1a1"),
}


@dataclass
class Case:
    gold: list = field(default_factory=list)
    token_counts: list = field(default_factory=list)


def checked_sources(root):
    paths, metadata = {}, {}
    for name, (repo, revision, filename, expected) in SOURCES.items():
        path = root / ("official" if name == "official" else f"all-predictions/{repo}") / filename
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"Unexpected file content: {path}")
        paths[name] = path
        metadata[name] = {
            "sha256": expected,
            "url": f"https://huggingface.co/datasets/L-NLProc/{repo}/resolve/{revision}/{quote(filename)}",
        }
    return paths, metadata


def csv_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        yield from csv.DictReader(stream)


def read_cases(path):
    cases = {}
    last = None
    for row in csv_rows(path):
        key = row["Index"]
        if key != last:
            if key in cases:
                raise ValueError(f"Official rows are noncontiguous for {key}")
            cases[key] = Case()
            last = key
        cases[key].gold.append(ROLE_IDS[row["Label"]])
        cases[key].token_counts.append(len(row["Text"].split()))
    return cases


def mtl_cases(path, cases):
    case_by_gold = {}
    for key, case in cases.items():
        fingerprint = tuple(case.gold)
        if fingerprint in case_by_gold:
            raise ValueError("Ambiguous MTL alignment: repeated official document label sequence")
        case_by_gold[fingerprint] = key
    predictions = {}
    for row in csv_rows(path):
        gold = tuple(tag - 3 for tag in json.loads(row["test_gold"]))
        predicted = [tag - 3 for tag in json.loads(row["test_pred"])]
        key = case_by_gold.get(gold)
        if key is None or key in predictions or len(gold) != len(predicted):
            raise ValueError("Unmatched, repeated or unequal-length MTL document")
        if any(p < 0 or p >= len(ROLES) for p in predicted):
            raise ValueError("Unexpected MTL output label")
        predictions[key] = predicted
    if predictions.keys() != cases.keys():
        raise ValueError("MTL does not cover each official document exactly once")
    return predictions


def flat_cases(path, cases, mapping):
    rows = iter(csv_rows(path))
    predictions = {}
    for key, case in cases.items():
        outputs = []
        for gold in case.gold:
            row = next(rows)
            if mapping[int(row["true"])] != gold:
                raise ValueError(f"Released gold labels diverge in document {key}")
            outputs.append(mapping[int(row["pred"])])
        predictions[key] = outputs
    if next(rows, None) is not None:
        raise ValueError("Extra released rows")
    return predictions


def runs(labels):
    start = 0
    for end in range(1, len(labels) + 1):
        if end == len(labels) or labels[end] != labels[start]:
            yield start, end, labels[start]
            start = end


def ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def exact_metrics(cases, prediction):
    exact = predicted_total = gold_total = correct_tokens = total_tokens = 0
    confusion = [[0] * 7 for _ in ROLES]
    for key, case in cases.items():
        gold, predicted = [], []
        for role, output, weight in zip(case.gold, prediction[key], case.token_counts, strict=True):
            confusion[role][output] += 1
            if weight:
                gold.append(role)
                predicted.append(output)
                total_tokens += weight
                correct_tokens += weight * (role == output)
        a, b = set(runs(predicted)), set(runs(gold))
        exact += len(a & b)
        predicted_total += len(a)
        gold_total += len(b)
    by_role = []
    for i in range(len(ROLES)):
        by_role.append(ratio(2 * confusion[i][i], sum(confusion[i]) + sum(row[i] for row in confusion)))
    return {
        "exact_labeled_spans": {"matched": exact, "predicted": predicted_total, "gold": gold_total,
                                "precision": ratio(exact, predicted_total), "recall": ratio(exact, gold_total),
                                "f1": ratio(2 * exact, predicted_total + gold_total)},
        "token_accuracy": ratio(correct_tokens, total_tokens),
        "correct_tokens": correct_tokens, "total_tokens": total_tokens,
        "sentence_accuracy": ratio(sum(confusion[i][i] for i in range(7)), sum(map(sum, confusion))),
        "sentence_macro_f1": sum(by_role) / len(by_role),
    }


def possible_run_counts(outputs):
    # Minimum and maximum run counts conditional on the last assigned role.
    previous = {None: (0, 0)}
    for output in outputs:
        feasible = range(len(ROLES)) if output is None else [output]
        current = {}
        for role in feasible:
            current[role] = (
                min(a + int(last != role) for last, (a, b) in previous.items()),
                max(b + int(last != role) for last, (a, b) in previous.items()),
            )
        previous = current
    return min(a for a, b in previous.values()), max(b for a, b in previous.values())


def score_bounds(gold, outputs):
    certain = possible = total = 0
    for a, b, role in runs(gold):
        edges = ([outputs[a - 1]] if a else []) + ([outputs[b]] if b < len(outputs) else [])
        inside = outputs[a:b]
        total += 1
        if all(value in (None, role) for value in inside) and all(value != role for value in edges):
            possible += 1
            if all(value == role for value in inside) and None not in edges:
                certain += 1
    min_runs, max_runs = possible_run_counts(outputs)
    return certain, possible, min_runs, max_runs, total


def generation_bounds(released, official, cases):
    predictions = {key: [] for key in cases}
    first_line_kinds = Counter()
    for row, original in itertools.zip_longest(csv_rows(released), csv_rows(official)):
        if row is None or original is None:
            raise ValueError("Generation file length mismatch")
        if (row["Index"], row["Text"], row["Label"] or "None") != (original["Index"], original["Text"], original["Label"]):
            raise ValueError("Generation file has a case, sentence text or label mismatch")
        line = (row["llama_p"] or "").lstrip().partition("\n")[0].strip()
        output = FIRST_LINES.get(line)
        first_line_kinds["unresolved" if output is None else "digit" if line.isdigit() else "role_and_number"] += 1
        predictions[row["Index"]].append(output)
    total = Counter()
    for key, case in cases.items():
        labels, outputs = [], []
        for gold, output, tokens in zip(case.gold, predictions[key], case.token_counts, strict=True):
            total["tokens"] += tokens
            total["known_correct_tokens"] += tokens * (gold == output)
            total["unknown_tokens"] += tokens * (output is None)
            if tokens:
                labels.append(gold)
                outputs.append(output)
                total["nonblank_unresolved_rows"] += output is None
        certain, possible, low, high, count = score_bounds(labels, outputs)
        total.update(certain=certain, possible=possible, min_predicted=low, max_predicted=high, gold=count)
    return {
        "alignment": "Every row matches exact case ID, text and role after restoring blank released gold cells to None.",
        "decoding": "Independent sensitivity analysis. Accept the entire first nonblank output line only if it is one digit 0 through 6 or the canonical role name and matching hyphenated digit. Later generated instructions are ignored. Each unresolved row can take any of the seven roles.",
        "intervals": "Deterministic outer bounds conditional on the first-line decoder. The minimum and maximum number of predicted spans are computed by dynamic programming. Bounds on the number of exact matches check each gold span and its immediate neighbors. The displayed bounds need not be jointly attainable.",
        "first_lines": dict(first_line_kinds), "counts": dict(total),
        "exact_span_f1_bounds": [ratio(2 * total["certain"], total["gold"] + total["max_predicted"]),
                                 ratio(2 * total["possible"], total["gold"] + total["min_predicted"])],
        "token_accuracy_bounds": [ratio(total["known_correct_tokens"], total["tokens"]),
                                  ratio(total["known_correct_tokens"] + total["unknown_tokens"], total["tokens"])],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, default=Path(__file__).resolve().parents[1] / "references")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths, sources = checked_sources(args.references)
    cases = read_cases(paths["official"])
    identity = dict(enumerate(range(7)))
    ci_mapping = dict(enumerate([1, 2, 3, 4, 5, 6, 0]))
    result = {
        "sources": sources, "case_count": len(cases), "all_rows": sum(len(c.gold) for c in cases.values()),
        "roles": ROLES,
        "metric": "Published predictions receive official sentence boundaries. Omit blank source rows for spans and whitespace-token accuracy, join trimmed rows with one space, and merge adjacent equal roles. Within one case, boundaries at different nonblank row indices map one-to-one to different character offsets. Compute exact labeled span F1 from pooled counts. The separate sentence scores include every official row.",
        "mtl": {
            "alignment": "Map published labels 3..9 to ordinal roles 0..6. Match each complete ordered gold-label vector to its unique official case. Stop on duplicates on either side, an incomplete match, extra cases or row count discrepancies. The releases supply no case IDs or sentence text for independent text verification.",
            "unique_matching_cases": len(cases), "scores": exact_metrics(cases, mtl_cases(paths["mtl"], cases))
        },
        "role_aware_ci": {
            "alignment": "The released Role-Aware CI code specifies the 0..6 -> 1,2,3,4,5,6,0 mapping. Every relabeled gold row matches in official order. The release omits text and case IDs. This is a file-level result.",
            "scores": exact_metrics(cases, flat_cases(paths["role_aware_ci"], cases, ci_mapping))
        },
        "previous_predicted": {
            "purpose": "Independent replay of the highest numerical archived projection in the existing comparison.",
            "scores": exact_metrics(cases, flat_cases(paths["previous_predicted"], cases, identity))
        },
        "previous_gold_oracle": {
            "purpose": "The source configuration includes the previous true role. This is a diagnostic oracle outside the deployable comparison.",
            "scores": exact_metrics(cases, flat_cases(paths["previous_gold"], cases, identity))
        },
        "rhetoricllama_first_line": generation_bounds(paths["rhetoricllama"], paths["official"], cases),
    }
    serialized = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
