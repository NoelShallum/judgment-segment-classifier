"""Freeze matched judgment IDs, original sentence numbering, and score plan."""
from hashlib import sha256
import json
from zipfile import ZipFile

from assets import HERE, REPO, REVISION, UPSTREAM, checksum, read, write


ROLE_NAMES = ("None", "Facts", "Issue", "Arguments of Petitioner", "Arguments of Respondent", "Reasoning", "Decision")
ROLE_DESCRIPTIONS = (
    "Other material such as a heading, procedural transition, cited passage, or statutory quotation that does not itself serve another listed role.",
    "Factual background or procedural history relevant to the case.",
    "A question or issue that the court must decide.",
    "Submissions or arguments advanced by the petitioner or appellant.",
    "Submissions or arguments advanced by the respondent.",
    "The present court's own analysis or reasons for resolving the issues.",
    "The present court's final disposition or operative order.",
)
POLICY = "karpo-public-reference-case-blocks-2026-11-22-v1"


def case_rank(key):
    return sha256((POLICY + "|" + key).encode()).digest()


def main():
    upstream = HERE.parent / "decision" / "glide-content-colab.zip"
    with ZipFile(upstream) as archive:
        packed = json.loads(archive.read("packed-full-plan.json"))
        source = archive.read("sampled-texts.json")
    frame = json.loads(source)
    assert frame["source"]["shared_manifest"] == packed["validation_manifest_sha256"]
    fit, val = packed["fixed_fit_ids"], packed["fixed_validation_ids"]
    inputs, labels = {}, {}
    for split, keys, count in (("train", fit, 357), ("val", val, 3558)):
        assert set(keys) == set(frame[split])
        inputs[split] = {k: [t for t, _ in frame[split][k]] for k in keys}
        labels[split] = {k: [ROLE_NAMES.index(r) for _, r in frame[split][k]] for k in keys}
        assert sum(map(len, inputs[split].values())) == count
        assert all(t.strip() and t == t.strip() for c in inputs[split].values() for t in c)
    legacy = HERE.parent / "pooler" / "shared-cases"
    incumbent = {}
    for case in val:
        receipt = read(legacy / (case + ".json"))
        assert receipt["id"] == case and receipt["rows"] == len(inputs["val"][case])
        assert receipt["manifest_sha256"] == frame["source"]["shared_manifest"]
        incumbent[case] = receipt["predictions"]["unchanged"]
    for name, data in (("inputs", inputs), ("truth", labels), ("incumbent", incumbent)):
        write(HERE / (name + ".json"), data)
    by_fold = [sorted(fit, key=case_rank)[i::3] for i in range(3)]
    documents = {k: len(x) for split in inputs.values() for k, x in split.items()}
    plan = {"policy": POLICY, "model": REPO, "revision": REVISION, "source_commit": UPSTREAM,
            "fit_cases": fit, "validation_cases": val, "fallback_validation_first_four_cases": val[:4],
            "training_cross_validation": by_fold,
            "matching": "Use all nonblank sentences of the 12 short training cases and the 16 shared validation judgments. Original rows are contiguous within each supplied state window and targeted once. The gold is stored in a distinct scorer file and has no path into the prompt writer. The shared validation has been used for other research comparisons.",
            "rows": documents, "source_csv_sha256": frame["source"], "glide_sample_bytes_sha256": sha256(source).hexdigest(),
            "outer_archive_sha256": checksum(upstream), "roles": list(ROLE_NAMES), "descriptions": list(ROLE_DESCRIPTIONS),
            "input_sha256": checksum(HERE / "inputs.json"), "truth_sha256": checksum(HERE / "truth.json"),
            "incumbent_sha256": checksum(HERE / "incumbent.json"), "maximum_questions_per_request": 8,
            "normal_token_cap": 1536, "single_question_hard_cap": 2048,
            "readout": "Write exactly the first seven rows of the publisher's weight [255,1024]. Fit a centered class residual ridge in float64 with train-only cross-validation of regularization, gain and class balancing, without gradient updates. Compare unchanged, edited, and equal Frobenius norm shuffled additive weights. Save a full replacement readout tensor and original digests.",
            "controls": "On the first canonical block of each validation case, rotate the seven displayed choices by a nonzero offset, and reverse question order within the identical state separately. Map outputs back by explicit target and role ID. Native-score semantics and readout row semantics are reported separately for the option-rotated readout.",
            "limits": "Retain only the same supplied sentence cuts. An overlength singleton is clipped from the middle under the hard cap while retaining both ends. Record its original tokens and clipping on every such row. Full validation inputs come before controls. A first-four case report is an honest lower-compute fallback fixed before inference.",
            "ridge": [.02, .12, .5], "gains": [.4, .7, 1.], "balance": [0., .35, .7], "target_margin": 4.,
            "shuffle_seed": 693001, "bootstrap_seed": 77162, "bootstrap_draws": 3000,
            "package_sha256_by_original_file": {n: checksum(HERE / n) for n in ("assets.py", "prepare.py")}}
    write(HERE / "protocol.json", plan)
    print("Frozen inputs", len(fit), len(val), sum(documents[k] for k in fit), sum(documents[k] for k in val), checksum(HERE / "protocol.json"))


if __name__ == "__main__":
    main()
