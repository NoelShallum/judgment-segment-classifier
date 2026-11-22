"""Transfer frozen joint pairwise weights to raw judgments and matched Gemini passages."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json

import numpy as np

import sentence_joint_pairwise as joint
import sentence_phrase_sequence as seq
from hier_model import LABELS
from raw_emitter_write import paired
from sentence_document_graph_write import file_hash, save_json, torch
from span_eval import counts, document, summarize

P = seq.phrase
TAG = "sentence-joint-pairwise-raw"


def block(docs):
    torch.set_num_threads(1)
    _, weights, recipe, frozen, _ = joint.load()
    model, cfg, vocab, memory, vectors, lex_weights, *_ = P.load_predictor("target phrases")
    result = []
    for doc in docs:
        text = doc["text"]
        raw = seq.raw_document(text, doc["id"])
        offsets = raw["units"]
        assert offsets == doc["units"]
        chunks = [text[a:b].strip() for a,b in offsets]
        lex = P.sparse(vectors, [P.canonical(t) for t in chunks]) @ lex_weights
        hidden = model.contextualize(model.words(chunks))
        key, dummy, original, anchor, x = P.dense.dense_features(raw, hidden, model, cfg, vocab, memory)
        bank = ([key], np.array([0, len(chunks)]), dummy, original, anchor, x)
        state = joint.prepare(bank, lex, weights, recipe)
        predictions = {"baseline": state[4]}
        for label, item in frozen["profiles"].items():
            predictions[label] = joint.apply(state, recipe, item["choices"])
        result.append((key, len(chunks), {name: counts(doc, offsets, p) for name, p in predictions.items()}))
    return result


def main(split):
    profile_path = P.RESULT / (joint.TAG + "-frozen.json")
    if split == "test":
        docs = P.docs("test")
        assert len(docs) == 712
        source = P.ROOT / "references" / "official" / "test.csv"
    else:
        source = P.ROOT / "seven-spans" / "cloud.json"
        archive = P.ROOT / "seven-spans" / "gemini-cloud.json"
        answers = json.loads(archive.read_text())
        docs = [document(b["doc"], [(r["row"], r["text"], LABELS.index(r["gold"])) for r in b["rows"]])
                for b in json.loads(source.read_text())]
    with ProcessPoolExecutor(4) as pool:
        by_key = {k: (n, counts) for report in pool.map(block, [docs[i::4] for i in range(4)])
                  for k,n,counts in report}
    ordered = [by_key[d["id"]] for d in docs]
    matrices = {name: np.stack([rows[name] for _, rows in ordered]) for name in ordered[0][1]}
    raw_baseline = matrices["baseline"]
    if split == "cloud":
        comparison = []
        for d in docs:
            y = answers[d["id"]]
            assert sha256(d["text"].encode()).hexdigest() == y["raw_sha256"]
            assert [list(pair) for pair in d["units"]] == y["unit_offsets"]
            comparison.append(counts(d, d["units"], [LABELS.index(role) for role in y["unit_labels"]]))
        matrices["Gemini Decisions"] = np.stack(comparison)
    out = {"source_sha256": file_hash(source), "joint_freeze_sha256": file_hash(profile_path),
           "judgments": len(docs), "units": sum(n for n,_ in ordered), "profiles": {}}
    for name, q in matrices.items():
        np.testing.assert_array_equal(q[:, :49].reshape(-1,7,7).sum(2), raw_baseline[:, :49].reshape(-1,7,7).sum(2))
        np.testing.assert_array_equal(q[:, [51,54]], raw_baseline[:, [51,54]])
        score = {"scores": summarize(q.sum(0)), "paired_to_baseline": paired(q, raw_baseline)}
        if split == "cloud":
            score["paired_to_gemini"] = paired(q, matrices["Gemini Decisions"])
        out["profiles"][name] = score
    if split == "test":
        measured = P.dense.read("sentence-phrase-sequence-raw")
        assert measured["predicted_sentence_units"] == out["units"]
        assert measured["profiles"]["spans"]["scores"] == out["profiles"]["baseline"]["scores"]
    else:
        previous = P.dense.read("sentence-pairwise-gemini-replay")["profiles"]
        assert previous["Gemini Decisions"]["scores"] == out["profiles"]["Gemini Decisions"]["scores"]
        assert previous["Phrase sequence"]["scores"] == out["profiles"]["baseline"]["scores"]
        out["gemini_archive_sha256"] = file_hash(archive)
    save_json(P.RESULT / (TAG + "-" + split + ".json"), out)
    print(split, {k:(v["scores"]["exact_labeled_spans"]["f1"], v["scores"]["token_accuracy"])
                  for k,v in out["profiles"].items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("test", "cloud"))
    main(parser.parse_args().split)
