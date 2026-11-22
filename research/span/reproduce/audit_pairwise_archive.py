"""Compare the strongest pairwise sentence edit to aligned public author predictions."""
import json

import numpy as np

import predict_pairwise_judgment as runtime
import sentence_pairwise_repair as repair
from sentence_document_graph_write import file_hash, paired, row_summary, save_json

P = repair.P
TAG = "sentence-pairwise-archive-audit"


def main():
    data = P.RESULT / "known-sentence-numerical-archives.npz"
    ref = P.dense.read("known-sentence-numerical-archives")
    assert file_hash(data) == ref["score_file_sha256"]
    pack = repair.receipt(repair.TAG)
    results = P.dense.read(repair.TAG)
    assert results["freeze_sha256"] == file_hash(pack)
    bank = P.dense.prepare("test")
    lex = P.cached_scores("test")
    base, score, _ = repair.incumbent(bank, lex, repair.load_recipe())["sequence only"]
    pred = runtime.predictor("direct")(bank, lex, score, base)
    counts = P.dense.paired_confusion(bank, pred)
    name = ref["best_by_macro_f1"]
    with np.load(data, allow_pickle=False) as archive:
        np.testing.assert_array_equal(bank[0], archive["cases"])
        author = archive[name]
    np.testing.assert_array_equal(counts.sum(2), author.sum(2))
    own = row_summary(counts.sum(0))
    assert own == results["splits"]["test"]["arms"]["sequence only"]["labels"]
    output = {"repair_freeze_sha256": file_hash(pack), "archive_weights_file_sha256": file_hash(data),
              "cases": len(bank[0]), "aligned_sentences": len(bank[2]), "reference": name,
              "local": own, "aligned_author": row_summary(author.sum(0)),
              "paired_delta": paired(counts, author)}
    save_json(P.RESULT / (TAG + ".json"), output)
    print(name, output["aligned_author"]["macro_f1"], own["macro_f1"], output["paired_delta"]["macro_f1"], flush=True)


if __name__ == "__main__":
    main()
