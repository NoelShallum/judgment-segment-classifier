"""Trace the released author's fold predictions without opening the official test CSV."""

from collections import defaultdict
import csv
import json
import string

import numpy as np

from lex_counts import HERE, ROOT, dump, file_hash
from embedding_write import confusion, metric


def main():
    parent = ROOT / "references/hier"
    archived = parent / "data_state4.json"
    z = json.loads(archived.read_text())
    tags = json.loads((parent / "tag2idx.json").read_text())
    folds = [bytes(row) for row in z["gold"]]
    assert len(folds) == len(z["idx"]) == len(z["pred"])
    assert all(len(a) == len(b) and len(a) for a,b in zip(folds, z["pred"]))
    role_order = [label for label, idx in sorted(tags.items(), key=lambda record:record[1]) if idx >= 3]
    y = np.concatenate(z["gold"]) - 3
    h = np.concatenate(z["pred"]) - 3
    assert np.min(y) >= 0 and np.max(y) <= 6 and np.min(h) >= 0 and np.max(h) <= 6
    punct = str.maketrans(string.punctuation, " " * len(string.punctuation))
    source_matches = {}
    for name in ("train", "val"):
        path = ROOT / "references/official" / (name + ".csv")
        raw = defaultdict(bytearray)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["Text"].lower().translate(punct).split():
                    raw[row["Index"]].append(tags[row["Label"]])
        by_labels = defaultdict(list)
        for group, role_sequence in raw.items():
            by_labels[bytes(role_sequence)].append(group)
        one = {str(i):by_labels[q][0] for i,q in enumerate(folds) if len(by_labels[q]) == 1}
        source_matches[name] = dict(
            csv_sha256=file_hash(path), eligible_docs=len(raw),
            exact_role_sequence_matches=sum(len(by_labels[q]) > 0 for q in folds),
            uniquely_matching_judgments=len(one), unique_matches=one)
    rank = metric(confusion(y, h))
    release = json.loads((parent / "inventory.json").read_text())
    model = parent / "model_state4.tar"
    record = dict(
        judgements=len(folds), sentences=len(y), labels=role_order, scores=rank,
        source_predictions_sha256=file_hash(archived), source_checkpoint_sha256=file_hash(model),
        public_revision=release["sha"],
        public_model="https://huggingface.co/L-NLProc/LegalSeg_Hier_BiLSTM-CRF",
        repository_training_script="https://github.com/ShubhamKumarNigam/LegalSeg/blob/main/code/Hier_BiLSTM%20CRF/train.py",
        paper_table="https://aclanthology.org/2025.findings-naacl.63.pdf#page=7",
        protocol="Score every archived Fold 4 gold/prediction pair. Index exact sentence-role sequences after the author's ASCII punctuation-and-whitespace filter in train and validation separately. Sequence agreement is weaker than text identity. No test cases were read.",
        source_matches=source_matches)
    dump(HERE / "author-fold4-audit.json", record)
    print(len(folds), len(y), rank["macro_f1"], rank["accuracy"],
          {name:(item["exact_role_sequence_matches"], item["uniquely_matching_judgments"])
           for name,item in source_matches.items()}, flush=True)


if __name__ == "__main__":
    main()
