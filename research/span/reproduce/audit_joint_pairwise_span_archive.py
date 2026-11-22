"""Verify the pre-decoding edit against the strongest published exact-span projection."""
import numpy as np

from audit_sentence_phrase_span_archive import archive_counts
import sentence_joint_pairwise as joint
from raw_emitter_write import paired
from sentence_document_graph_write import save_json
from span_eval import summarize

P = joint.P


def main():
    _, weights, recipe, plan, h = joint.load()
    bank = P.dense.prepare("test")
    inputs = joint.prepare(bank, P.cached_scores("test"), weights, recipe)
    pred = joint.apply(inputs, recipe, plan["profiles"]["spans"]["choices"])
    before, after, release, checksum = archive_counts(bank, pred)
    np.testing.assert_array_equal(before[:, :49].reshape(-1,7,7).sum(2), after[:, :49].reshape(-1,7,7).sum(2))
    old, new = (summarize(m.sum(0)) for m in (before, after))
    prior = P.dense.read(joint.TAG)
    assert prior["freeze_sha256"] == h
    assert new["exact_labeled_spans"]["f1"] == prior["splits"]["test"]["profiles"]["spans"]["spans"]["exact_labeled_span_f1"]
    report = {"joint_freeze_sha256": h, "judgments": len(bank[0]), "sentences": len(bank[2]),
              "public_release": release, "public_sha256": checksum, "public_scores": old, "edited_scores": new,
              "paired_change_from_public": paired(after,before), "exact_char_sentence_parity": True}
    save_json(P.RESULT / "sentence-joint-pairwise-span-archive.json", report)
    print(release, old["exact_labeled_spans"]["f1"], new["exact_labeled_spans"]["f1"],
          report["paired_change_from_public"][0], flush=True)


if __name__ == "__main__":
    main()
