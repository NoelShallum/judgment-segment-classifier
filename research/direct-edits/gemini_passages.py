"""Compare direct edits and the case model on the twelve archived Gemini passages."""
import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE / "priors"))

from experiment import LABELS, file_hash, write
from predict_document import Predictor
from predict_wide_bert_review import predictor
import sentence_phrase_sequence as sequence
from span_eval import counts, document, summarize


def intervals(candidate, gemini):
    draws = np.random.default_rng(145661).integers(len(candidate), size=(3000, len(candidate)))
    def metrics(data):
        c = np.stack(data)[draws].sum(1)
        cm = c[:, :49].reshape(-1, 7, 7)
        correct = np.diagonal(cm, axis1=1, axis2=2).sum(1)
        return np.stack([2*c[:, 49]/(c[:, 50]+c[:, 51]), correct/cm.sum((1, 2))])
    return np.quantile(metrics(candidate)-metrics(gemini), [.025, .975], axis=1).T.tolist()


def main():
    torch.set_num_threads(1)
    cases_path = ROOT / "seven-spans/cloud.json"
    answers_path = ROOT / "seven-spans/gemini-cloud.json"
    cases = json.loads(cases_path.read_text())
    answers = json.loads(answers_path.read_text())
    cases_by_key = []
    for case in cases:
        doc_id = case["doc"]
        doc = document(doc_id, [(r["row"], r["text"], LABELS.index(r["gold"])) for r in case["rows"]])
        answer = answers[doc_id]
        assert answer["model"] == "gemini-decision-1"
        assert hashlib.sha256(doc["text"].encode()).hexdigest() == answer["raw_sha256"]
        assert [list(x) for x in doc["units"]] == answer["unit_offsets"]
        units = [doc["text"][a:b] for a, b in doc["units"]]
        cases_by_key.append((doc_id, doc, units))
    assert len(cases) == len(cases_by_key) == len(answers)
    predicted = {"Gemini Decisions": {k: answers[k]["unit_labels"] for k, _, _ in cases_by_key}}
    for mode in ("source", "retention", "document-prior", "retention-prior"):
        run = Predictor(HERE / "portable-smoke", mode)
        predicted[mode] = {k: run.predict(units)[0]["labels"] for k, _, units in cases_by_key}
    sequence.phrase.load_predictor = lru_cache(maxsize=1)(sequence.phrase.load_predictor)
    routed = predictor("CPU", .20)
    predicted["current-20-percent"] = {k: routed(units)["labels"] for k, _, units in cases_by_key}
    all_counts = {}
    for method, labels in predicted.items():
        all_counts[method] = []
        for case_id, doc, units in cases_by_key:
            assert len(labels[case_id]) == len(units)
            classes = np.asarray([LABELS.index(x) for x in labels[case_id]])
            all_counts[method].append(counts(doc, doc["units"], classes))
    baseline = all_counts["Gemini Decisions"]
    results = {}
    for name, matrices in all_counts.items():
        results[name] = {**summarize(np.stack(matrices).sum(0)), "span_f1_token_accuracy_delta_ci95": intervals(matrices, baseline)}
    out = HERE / "gemini"
    out.mkdir(exist_ok=True)
    labels_file = out / "passage-predictions.json"
    write(labels_file, predicted)
    write(out / "passage-results.json", {
        "cases": len(cases), "human_rows": sum(len(case["rows"]) for case in cases),
        "automatically_split_units": sum(len(units) for _, _, units in cases_by_key),
        "api_calls_in_this_replay": 0, "input_sha256": file_hash(cases_path), "gemini_answers_sha256": file_hash(answers_path),
        "predictions_sha256": file_hash(labels_file), "scores": results,
        "protocol": "Models received the same automatically generated units of each raw clipped passage. The frozen Gemini record contains a role for each unit. The gold span endpoints come from the original human-labeled rows. Exact labeled spans require both endpoints and the role. Bootstrap resamples twelve passages.",
    })
    print({k: (round(v["exact_labeled_spans"]["f1"], 4), round(v["token_accuracy"], 4)) for k,v in results.items()})


if __name__ == "__main__":
    main()
