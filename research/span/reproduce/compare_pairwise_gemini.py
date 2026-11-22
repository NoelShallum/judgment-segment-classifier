"""Replay frozen sequence and pairwise weights on Gemini's exact passage units."""
from hashlib import sha256
import json

import numpy as np

import sentence_pairwise_consensus as consensus
import sentence_pairwise_repair as repair
import sentence_phrase_sequence as seq
from sentence_document_graph_write import file_hash, save_json, torch
from span_eval import counts, document, summarize
from raw_emitter_write import paired
from hier_model import LABELS

P = seq.phrase
TAG = "sentence-pairwise-gemini-replay"


def main():
    torch.set_num_threads(2)
    parent, w, repair_hash = consensus.inputs()
    con_path = repair.receipt(consensus.TAG)
    coupled = json.loads(con_path.read_text())
    assert coupled["repair_freeze_sha256"] == repair_hash
    recipe = repair.load_recipe()
    model, cfg, vocab, memory, vectors, lexical_weight, *_ = P.load_predictor("target phrases")
    source = P.ROOT / "seven-spans" / "cloud.json"
    archive = P.ROOT / "seven-spans" / "gemini-cloud.json"
    passages = json.loads(source.read_text())
    answers = json.loads(archive.read_text())
    metrics = {}
    num_units = 0
    for block in passages:
        case = block["doc"]
        truth = document(case, [(row["row"], row["text"], LABELS.index(row["gold"]))
                                for row in block["rows"]])
        text = truth["text"]
        raw = seq.raw_document(text, case)
        offsets = raw["units"]
        reply = answers[case]
        assert sha256(text.encode()).hexdigest() == reply["raw_sha256"]
        assert truth["units"] == offsets
        assert [list(pair) for pair in offsets] == reply["unit_offsets"]
        chunks = [text[a:b].strip() for a, b in offsets]
        lex = P.sparse(vectors, [P.canonical(t) for t in chunks]) @ lexical_weight
        states = model.contextualize(model.words(chunks))
        k, dummy_roles, original, anchor, x = P.dense.dense_features(raw, states, model, cfg, vocab, memory)
        bank = ([k], np.array([0, len(chunks)], dtype=np.int64), dummy_roles, original, anchor, x)
        base, unary, protection = repair.incumbent(bank, lex, recipe)["sequence only"]
        feature = repair.dense_basis(bank, lex)
        option = parent["arms"]["sequence only"]["choice"]
        edit = base
        if option:
            mode, scale, top, threshold, guard = option
            choices = repair.proposals(repair.score_pairs(feature, w, mode), base, unary, scale, top)
            edit = repair.decide(choices, base, protection, threshold, guard)
        option = coupled["arms"]["sequence only"]["choice"]
        combined = base
        if option:
            mode, rule, strength, threshold, guard = option
            alt = consensus.reconcile(repair.score_pairs(feature, w, mode), unary, rule)
            combined = consensus.decide(alt, base, unary, protection, strength, threshold, guard)
        external = np.array([LABELS.index(k) for k in reply["unit_labels"]], dtype=np.int8)
        assert len(external) == len(chunks)
        for name, v in (("Gemini Decisions", external), ("Phrase sequence", base),
                        ("Direct challenger", edit), ("Reconciled pair scores", combined)):
            metrics.setdefault(name, []).append(counts(truth, offsets, v))
        num_units += len(chunks)
    external = np.stack(metrics["Gemini Decisions"])
    out = {"cases": len(passages), "candidate_units": num_units, "repair_freeze_sha256": repair_hash,
           "consensus_freeze_sha256": file_hash(con_path), "passages_sha256": file_hash(source),
           "gemini_archive_sha256": file_hash(archive), "profiles": {}}
    for name, values in metrics.items():
        result = np.stack(values)
        np.testing.assert_array_equal(result[:, :49].reshape(-1, 7, 7).sum(2),
                                      external[:, :49].reshape(-1, 7, 7).sum(2))
        np.testing.assert_array_equal(result[:, [51, 54]], external[:, [51, 54]])
        out["profiles"][name] = {"scores": summarize(result.sum(0)),
                                 "paired_to_gemini": paired(result, external)}
    save_json(P.RESULT / (TAG + ".json"), out)
    print({k: (r["scores"]["exact_labeled_spans"]["f1"], r["scores"]["token_accuracy"])
           for k, r in out["profiles"].items()}, flush=True)


if __name__ == "__main__":
    main()
