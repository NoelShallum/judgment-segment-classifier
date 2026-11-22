"""Check alignment and replay alternative decoders of the published RhetoricLLaMA text."""

from collections import Counter
import csv
from hashlib import file_digest
from itertools import groupby
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent.parent
ROLE_NAMES = [
    "None",
    "Facts",
    "Issue",
    "Arguments of Petitioner",
    "Arguments of Respondent",
    "Reasoning",
    "Decision",
]
ROLE_ID = {label: i for i, label in enumerate(ROLE_NAMES)}
ABSTAIN = -1
MARKER = re.compile(r"^\s*###\s*(?:response|answer)\s*:\s*", flags=re.IGNORECASE | re.MULTILINE)
ANSWERS = {}
for i, name in enumerate(ROLE_NAMES):
    for text in (str(i), f"{i}.", f"{i})", name, f"{name}-{i}"):
        ANSWERS[text.casefold()] = i


def decode_section(section):
    first = next((line.strip().casefold() for line in section.splitlines() if line.strip()), "")
    return ANSWERS.get(first, ABSTAIN)


def spans(roles):
    """Equal adjacent rows collapse. Row ordinals uniquely index source offsets within the case."""
    result = set()
    start = 0
    for role, group in groupby(roles):
        end = start + sum(1 for _ in group)
        result.add((start, end, role))
        start = end
    return result


class Metrics:
    def __init__(self):
        self.row_pairs = Counter()
        self.token_pairs = Counter()
        self.matches = self.gold_spans = self.predicted_spans = 0

    def case(self, gold, prediction, token_counts):
        seen_gold, seen_pred = [], []
        for a, b, weight in zip(gold, prediction, token_counts, strict=True):
            self.row_pairs[a, b] += 1
            if weight:
                seen_gold.append(a)
                seen_pred.append(b)
                self.token_pairs[a, b] += weight
        actual, predicted = spans(seen_gold), spans(seen_pred)
        self.matches += len(actual & predicted)
        self.gold_spans += len(actual)
        self.predicted_spans += len(predicted)

    def summary(self):
        def accuracy(pairs):
            return sum(v for (a, b), v in pairs.items() if a == b) / pairs.total()

        f1 = []
        for i in range(7):
            denominator = sum(v * ((a == i) + (b == i)) for (a, b), v in self.row_pairs.items())
            f1.append(2 * self.row_pairs[i, i] / denominator if denominator else 0)
        return {
            "rows": self.row_pairs.total(),
            "sentence_accuracy": accuracy(self.row_pairs),
            "sentence_macro_f1": sum(f1) / 7,
            "tokens": self.token_pairs.total(),
            "token_accuracy": accuracy(self.token_pairs),
            "exact_spans": {
                "matched": self.matches,
                "predicted": self.predicted_spans,
                "gold": self.gold_spans,
                "micro_f1": 2 * self.matches / (self.predicted_spans + self.gold_spans),
            },
        }


def main():
    official = ROOT / "references/official/test.csv"
    archive = ROOT / "references/all-predictions/LegalSeg_RhetoricLLaMA_Predictions/RhetoricLLaMA_predictions.csv"
    policies = {name: Metrics() for name in ("leading_abstain", "leading_none", "last_heading_none")}
    parse_counts = Counter()
    cases = set()

    with official.open(encoding="utf-8-sig", newline="") as source, archive.open(encoding="utf-8-sig", newline="") as released:
        pairs = zip(csv.DictReader(source), csv.DictReader(released), strict=True)
        for case_id, rows in groupby(pairs, key=lambda pair: pair[0]["Index"]):
            assert case_id not in cases
            cases.add(case_id)
            truth, words = [], []
            predictions = {name: [] for name in policies}
            for reference, archived in rows:
                gold_label = archived["Label"] or "None"
                assert case_id == archived["Index"] and reference["Text"] == archived["Text"]
                assert reference["Label"] == gold_label
                truth.append(ROLE_ID[gold_label])
                words.append(len(reference["Text"].split()))

                generation = archived["llama_p"]
                first = decode_section(generation)
                sections = MARKER.split(generation)
                last = decode_section(sections[-1]) if len(sections) > 1 else first
                parse_counts["lead_parsed"] += first != ABSTAIN
                parse_counts["rows_with_generated_response_marker"] += len(sections) > 1
                if len(sections) > 1 and first != ABSTAIN and last != ABSTAIN:
                    parse_counts["leading_and_last_both_parsed_with_marker"] += 1
                    parse_counts["leading_and_last_disagree_with_marker"] += first != last

                predictions["leading_abstain"].append(first)
                predictions["leading_none"].append(0 if first == ABSTAIN else first)
                predictions["last_heading_none"].append(0 if last == ABSTAIN else last)
            for name, totals in policies.items():
                totals.case(truth, predictions[name], words)

    hashes = {}
    for path in (official, archive):
        with path.open("rb") as stream:
            hashes[path.name] = file_digest(stream, "sha256").hexdigest()
    result = {
        "sources": {
            "archive": "https://huggingface.co/datasets/L-NLProc/LegalSeg_RhetoricLLaMA_Predictions/blob/686d8082bdadc4f08cf21ff068b561ac38a88cc4/RhetoricLLaMA_predictions.csv",
            "prompt_and_unparsed_export": "https://github.com/ShubhamKumarNigam/LegalSeg/blob/main/code/RhetoricLLaMA/inference.py",
            "paper": "https://aclanthology.org/2025.findings-naacl.63/",
        },
        "hashes": hashes,
        "cases_verified_by_id_text_and_role": len(cases),
        "parse_counts": dict(parse_counts),
        "methods": {
            "decoder": "First nonempty line of the section must be a whole canonical role, role-ID pair, digit 0..6, or digit followed by a period or closing parenthesis. Case insensitive. Uses the archive's original 0..6 role mapping.",
            "leading_abstain": "Use the leading section. Invalid responses count as a wrong row, token, and predicted span with an eighth invalid label. Macro F1 averages the seven gold classes.",
            "leading_none": "Use the leading section. Invalid responses fall back to None.",
            "last_heading_none": "Use the final generated ### Response: or ### Answer: section when it exists, otherwise the leading section. Invalid responses fall back to None.",
            "reference": "All reported spans are reconstructed by supplying the original nonblank row offsets and merging adjacent equal labels. Source case, text, and role equality are asserted on every row. Archive blank Label means the official class None.",
            "attribution": "These decoders are replay assumptions. The cited inference script saves free generations without evaluation decoding. Table 3 separately reports macro F1 0.09 and accuracy 0.39.",
        },
        "scores": {k: v.summary() for k, v in policies.items()},
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
