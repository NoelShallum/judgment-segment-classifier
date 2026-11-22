"""Verify live supplied-sentence outputs match the independently scored benchmarks."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

import sentence_pairwise_consensus as consensus
import sentence_pairwise_repair as repair
import sentence_phrase_sequence as seq
from hier_model import LABELS
from sentence_document_graph_write import file_hash, save_json

P = seq.phrase


def main():
    direct, weights, direct_hash = consensus.inputs()
    consensus_path = repair.receipt(consensus.TAG)
    plan = json.loads(consensus_path.read_text())
    bank, x, bases = repair.views("confirm", repair.load_recipe())
    base, unary, protected = bases["sequence only"]
    mode, scale, top, threshold, guard = direct["arms"]["sequence only"]["choice"]
    contest = repair.proposals(repair.score_pairs(x, weights, mode), base, unary, scale, top)
    predicted_direct = repair.decide(contest, base, protected, threshold, guard)
    mode, rule, strength, cutoff, guard = plan["arms"]["sequence only"]["choice"]
    alt = consensus.reconcile(repair.score_pairs(x, weights, mode), unary, rule)
    predicted_consensus = consensus.decide(alt, base, unary, protected, strength, cutoff, guard)
    candidates = sorted((int(b-a), str(k), a, b) for k,a,b in zip(bank[0],bank[1],bank[1][1:])
                        if np.any(base[a:b]!=predicted_direct[a:b]) and np.any(base[a:b]!=predicted_consensus[a:b]))
    _, case, a, b = candidates[0]
    docs = {d["id"]: d for d in P.docs("confirm")}
    doc = docs[case]
    text = [doc["text"][s:e].strip() for s, e in doc["reference_units"]]
    record = {"case": case, "sentences": len(text), "methods": {},
              "repair_freeze_sha256": direct_hash, "consensus_freeze_sha256": file_hash(consensus_path)}
    with TemporaryDirectory(dir=P.ROOT / "qa") as dest:
        entry = Path(dest) / "input.json"
        entry.write_text(json.dumps(text))
        executable = str(P.ROOT / "predict_pairwise_judgment.py")
        for method, full in (("direct", predicted_direct), ("consensus", predicted_consensus)):
            pred = full[a:b]
            p = subprocess.check_output([sys.executable, executable, "--method", method, "--input", str(entry)])
            result = json.loads(p)
            assert result["labels"] == [LABELS[k] for k in pred]
            starts, stops, roles = seq.runs(np.array([0, len(pred)]), pred)
            assert result["spans"] == [{"start_sentence": int(s), "end_sentence": int(e), "label": LABELS[k]}
                                       for s, e, k in zip(starts, stops, roles)]
            record["methods"][method] = {"changed_from_sequence": int(np.count_nonzero(pred!=base[a:b])),
                                         "exact_prediction_parity": True}
    save_json(P.RESULT / "sentence-pairwise-cli-qa.json", record)
    print(record, flush=True)


if __name__ == "__main__":
    main()
