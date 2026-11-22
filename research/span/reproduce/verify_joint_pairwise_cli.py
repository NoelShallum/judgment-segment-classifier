"""Check the joint pairwise CPU command against the offline sentence and raw scoring paths."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

import compare_joint_pairwise_raw as raw_benchmark
import sentence_joint_pairwise as joint
import sentence_phrase_sequence as seq
from hier_model import LABELS
from sentence_document_graph_write import save_json
from span_eval import counts, document

P = seq.phrase
TAG = "sentence-joint-pairwise-cli-qa"


def command(path, profile, *extra):
    return json.loads(subprocess.check_output(
        [sys.executable, str(P.ROOT / "sentence_joint_pairwise.py"), "predict",
         "--profile", profile, "--input", str(path), *extra]))


def main():
    _, weights, recipe, frozen, freeze_hash = joint.load()
    bank = P.dense.prepare("confirm")
    lex = P.cached_scores("confirm")
    state = joint.prepare(bank, lex, weights, recipe)
    results = {name: joint.apply(state, recipe, frozen["profiles"][name]["choices"])
               for name in ("roles", "spans")}
    candidates = sorted((int(b-a), str(case), int(a), int(b)) for case,a,b in zip(bank[0], bank[1], bank[1][1:])
                        if all(np.any(p[a:b] != state[4][a:b]) for p in results.values()))
    _, key, a, b = candidates[0]
    refs = {d["id"]: d for d in P.docs("confirm")}
    row = refs[key]
    sentences = [row["text"][s:e].strip() for s,e in row["reference_units"]]
    evidence = {"joint_freeze_sha256": freeze_hash, "sentence_case": key, "sentences": len(sentences), "profiles": {}}
    with TemporaryDirectory(dir=P.ROOT / "qa") as direc:
        case_file = Path(direc) / "sentences.json"
        case_file.write_text(json.dumps(sentences))
        for name, p in results.items():
            expected = p[a:b]
            output = command(case_file, name)
            assert output["labels"] == [LABELS[r] for r in expected]
            s,t,v = seq.runs(np.array([0,len(expected)]), expected)
            assert output["spans"] == [{"start_sentence": int(i), "end_sentence": int(j), "label": LABELS[k]}
                                       for i,j,k in zip(s,t,v)]
            evidence["profiles"][name] = {"revisions": int(np.count_nonzero(expected != state[4][a:b])),
                                           "exact_label_and_span_parity": True}
        passages = json.loads((P.ROOT / "seven-spans" / "cloud.json").read_text())
        block = passages[0]
        doc = document(block["doc"], [(r["row"], r["text"], LABELS.index(r["gold"])) for r in block["rows"]])
        expected = raw_benchmark.block([doc])[0][2]["spans"]
        raw_file = Path(direc) / "judgment.txt"
        raw_file.write_bytes(doc["text"].encode("utf-8"))
        output = command(raw_file, "spans", "--raw")["spans"]
        assert output[0]["start"] == 0 and output[-1]["end"] == len(doc["text"])
        assert all(x["end"] == y["start"] for x,y in zip(output,output[1:]))
        index = np.searchsorted([s["start"] for s in output], [a for a,_ in doc["units"]], side="right")-1
        observed = [LABELS.index(output[i]["label"]) for i in index]
        np.testing.assert_array_equal(counts(doc, doc["units"], observed), expected)
        evidence["raw"] = {"case": doc["id"], "units": len(doc["units"]), "exact_metric_parity": True, "spans": len(output)}
    save_json(P.RESULT / (TAG + ".json"), evidence)
    print(evidence, flush=True)


if __name__ == "__main__":
    main()
