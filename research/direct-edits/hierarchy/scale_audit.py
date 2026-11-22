"""Audit the frozen scale write against native inference and exact sentence-index spans."""

import csv
import json
import sys

import numpy as np
from safetensors.numpy import load_file

from native_memory import BASE, HERE, ROOT, TO_ID, confusion_docs, decode, digest, dump, file_hash, metric
from retention_replay import load_arm
from scale_program import FOLDER, save_np

sys.path.insert(0, str(HERE.parent))
from span_comparison import interval, measures, tally


def collect_rows(source, cases):
    rows = {key: ([], []) for key in cases}
    with source.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            key = row["Index"]
            text = row["Text"].strip()
            if key in rows and text:
                rows[key][0].append(text)
                rows[key][1].append(TO_ID[row["Label"]])
    return rows


def shared_comparison(source, candidate, patch_sha):
    """Posthoc comparison on the already-scored common validation judgments."""
    manifest = HERE.parent / "shared-confirmation.json"
    reference = json.loads(manifest.read_text())
    assert reference["official_csv_sha256"] == file_hash(source)
    ids = reference["extended_validation_judgment_ids"]
    rows = collect_rows(source, ids)
    ends = [0]
    gold = []
    manifest_sha = file_hash(manifest)
    author = load_arm(None, None)
    models = {"native_donor":author, "frozen_4984":candidate}
    patch_ids = {"frozen_4984":patch_sha}
    chain = HERE / "chain-program/reserved96.json"
    if chain.exists():
        r = json.loads(chain.read_text())
        origin = HERE / "chain-program/freeze.json"
        assert r["freeze_sha256"] == file_hash(origin)
        frozen = json.loads(origin.read_text())
        selected = frozen["checkpoints"][r["chosen"]]
        models["crf_posterior"] = load_arm(f"chain-program/{selected['file']}", selected["sha256"])
        patch_ids["crf_posterior"] = selected["sha256"]
    score = {key:[] for key in (*models, "current_review_20pct")}
    for key in ids:
        texts, labels = rows[key]
        words = author.words(texts)
        gold.extend(labels)
        ends.append(len(gold))
        for name, model in models.items():
            z = model.logits(model.contextualize(words))
            score[name].extend(decode(z, [0, len(z)], model.transitions))
        path = HERE.parent / "pooler" / "shared-cases" / f"{key}.json"
        receipt = json.loads(path.read_text())
        assert receipt["id"] == key and receipt["rows"] == len(labels) and receipt["manifest_sha256"] == manifest_sha
        strong = receipt["predictions"]["unchanged"]
        assert len(strong) == len(labels)
        score["current_review_20pct"].extend(strong)
    gold = np.asarray(gold)
    conf = {}
    spans = {}
    for key, values in score.items():
        pred = np.asarray(values)
        conf[key] = confusion_docs(gold, pred, ends)
        spans[key] = np.stack([tally(gold[a:b], pred[a:b]) for a,b in zip(ends, ends[1:])])
    rank = {key: dict(**metric(value.sum(0)), spans=measures(spans[key])) for key,value in conf.items()}
    pred_archive = FOLDER / "shared48-audit-predictions.npz"
    save_np(pred_archive, keys=ids, ends=ends, gold=gold, **score)
    draw = np.random.default_rng(522170).integers(len(ids), size=(2500, len(ids)))
    paired = {}
    for cand in patch_ids:
        paired[cand] = {}
        for name in ("native_donor", "current_review_20pct"):
            diff = []
            for sample in draw:
                tgt, other = (metric(conf[k][sample].sum(0)) for k in (cand, name))
                diff.append([tgt["macro_f1"] - other["macro_f1"], tgt["accuracy"] - other["accuracy"]])
            paired[cand][name] = dict(macro_f1_accuracy_ci95=np.quantile(diff, [.025, .975], axis=0).T.tolist(),
                                      labeled_span_f1_ci95=interval(spans[cand], spans[name]))
    dump(FOLDER / "shared48-audit.json", dict(
        judgments=len(ids), rows=len(gold), keys=ids, manifest_sha256=manifest_sha,
        prediction_sha256=file_hash(pred_archive), physical_native_patch_sha256=patch_sha,
        native_patch_hashes=patch_ids,
        status="Descriptive comparison using the previously fixed shared cases and already-frozen predictors. Not used to choose a patch.",
        scores=rank, paired_candidate_minus_control=paired["frozen_4984"],
        paired_posterior_minus_control=paired.get("crf_posterior")))
    print("Shared 48", len(gold), {k:round(v["macro_f1"], 5) for k,v in rank.items()}, flush=True)


def main():
    chosen_result = json.loads((FOLDER / "reserved72.json").read_text())
    plan = json.loads((FOLDER / "plan.json").read_text())
    archive = FOLDER / "reserved72-predictions.npz"
    file = FOLDER / "frozen-native-patch.safetensors"
    assert chosen_result["frozen_model"] == file_hash(file)
    source = ROOT / "references/official/val.csv"
    assert file_hash(source) == plan["val_csv_sha256"]
    before = load_file(str(BASE))
    after = load_file(str(file))
    tensors = {}
    for key, edited in after.items():
        original = before[key]
        changed = edited != original
        where = np.flatnonzero(changed.any(1) if edited.ndim == 2 else changed)
        expected = np.arange(100, 200) if ".lstm." in key else np.arange(3, 10)
        np.testing.assert_array_equal(where, expected)
        tensors[key] = dict(shape=list(edited.shape), changed_values=int(changed.sum()),
                            changed_rows_or_indices_inclusive=[int(where[0]), int(where[-1])],
                            before_sha256=digest(original), after_sha256=digest(edited),
                            delta_l2=float(np.linalg.norm(edited.astype(float) - original)))

    with np.load(archive, allow_pickle=False) as bank:
        ids = bank["keys"].tolist()
        y = bank["gold"]
        ends = bank["ends"]
        predictions = {key: bank[key] for key in chosen_result["models"]}
    assert ids == chosen_result["cases"] == plan["validation"] and len(ids) == 72
    assert len(y) == chosen_result["rows"] and ends[-1] == len(y)
    spans = {}
    for key, pred in predictions.items():
        assert metric(confusion_docs(y, pred, ends).sum(0)) == chosen_result["models"][key]
        spans[key] = np.stack([tally(y[a:b], pred[a:b]) for a,b in zip(ends, ends[1:])])
    sample = ids[:4]
    rows = collect_rows(source, sample)
    native = load_arm("scale/frozen-native-patch.safetensors", chosen_result["frozen_model"])
    primary = chosen_result["chosen"]
    for key, a, b in zip(sample, ends, ends[1:]):
        texts, gold = rows[key]
        np.testing.assert_array_equal(y[a:b], gold)
        live = native.logits(native.contextualize(native.words(texts)))
        pred = decode(live, [0, len(live)], native.transitions)
        np.testing.assert_array_equal(pred, predictions[primary][a:b])

    profiles = {key: measures(value) for key,value in spans.items()}
    comparisons = {key: interval(spans[primary], spans[key])
                   for key in ("retention-legacy_240", "retention-unedited_readout", "donor-unedited_readout")}
    dump(FOLDER / "reserved72-audit.json", dict(
        rows=len(y), judgments=len(ids), validation_csv_sha256=file_hash(source),
        source_predictions_sha256=file_hash(archive), native_patch_sha256=file_hash(file),
        original_checkpoint_sha256=file_hash(BASE), tensors=tensors,
        native_parity={"judgments":sample, "rows":int(ends[4]), "label_disagreements":0},
        labels="None, Facts, Issue, Arguments of Petitioner, Arguments of Respondent, Reasoning, Decision",
        metric="Exact labeled span requires role and both endpoints on supplied sentence boundaries.",
        span_metrics=profiles, primary_paired_span_f1_ci95=comparisons))
    print("Frozen scale native parity", len(sample), int(ends[4]), "zero label differences",
          "span F1", profiles[primary]["exact_labeled_span_f1"], flush=True)
    shared_comparison(source, native, chosen_result["frozen_model"])


if __name__ == "__main__":
    main()
