"""Score frozen models on all official LegalSeg test rows and pair results by case."""
import argparse
from collections import Counter
import csv
import json
import math

import numpy as np

from cache_full import BASE_HASH, BLOCK, block_dir, verify_block
from encoder import fingerprint
from import_official import FILES
from program_trial import TO_ID, metric, paired_document_interval, save_json
from weight_edit import ORIGINAL, ROOT, file_hash, tensor


DIRECT_HASH = "0a32fe939d96384c3cf3d879e5cfbce76c67172a8f3920961a99df3ba65b7edf"


def official():
    source = ROOT / "references/official/test.csv"
    if file_hash(source) != FILES["test.csv"][1]:
        raise ValueError("Wrong test data revision")
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = [(i, row["Index"], row["Text"] or "", TO_ID[row["Label"]])
                for i, row in enumerate(csv.DictReader(stream))]
    truth = np.array([r[3] for r in rows])
    assert len(rows) == 149881 and len({row[1] for row in rows}) == 712
    released = ROOT / "references/baseline_sentence/InLegalBert(i)_predictions.csv"
    with released.open(encoding="utf-8-sig", newline="") as stream:
        output = [(int(row["true"]), int(row["pred"])) for row in csv.DictReader(stream)]
    author_gold, author_predictions = np.asarray(output).T
    if not np.array_equal(truth, author_gold):
        raise ValueError("Author prediction file does not align to the official gold")
    return rows, truth, author_predictions


def compare(rows, truth, prediction, comparison):
    return {"macro_f1_gain": metric(truth, prediction)["macro_f1"] - metric(truth, comparison)["macro_f1"],
            "paired_case_bootstrap_95pct_gain": paired_document_interval(rows, truth, comparison, prediction, None),
            "changed_rows": int(np.count_nonzero(prediction != comparison))}


def score_bert(rows, truth, release):
    directory = block_dir("test")
    expected = math.ceil(len(rows) / BLOCK)
    paths = {}
    for i in range(expected):
        for variant in (directory, directory/"openvino-gpu-fp32", directory/"openvino-cpu-fp32"):
            if (variant / f"{i:04d}.json").is_file():
                paths[i] = variant
                break
    missing = [i for i in range(expected) if i not in paths]
    if missing:
        raise RuntimeError(f"Awaiting {len(missing)} of {expected} test blocks")
    edited_file = ROOT / "checkpoints/selected-direct/model.safetensors"
    if file_hash(edited_file) != DIRECT_HASH or file_hash(ORIGINAL) != BASE_HASH:
        raise ValueError("Frozen weights do not match")
    head_files = {"original": ORIGINAL, "direct": edited_file}
    for name, folder, record_file in (("gold_discriminant","discriminant-256","discriminant-256-validation.json"),
                                     ("synthetic_augmented","augmented-discriminant","gemini-augmentation-validation.json")):
        record_path = ROOT/"results"/record_file
        other = ROOT/"checkpoints"/folder/"model.safetensors"
        if record_path.is_file() and other.is_file():
            record = json.loads(record_path.read_text())
            spec = record.get("selected") or record["checkpoint"]
            if file_hash(other) != spec["target_sha256"]:
                raise ValueError(f"Frozen validation checkpoint mismatch for {name}")
            head_files[name] = other
    associative=ROOT/"checkpoints/associative-direct/model.safetensors"
    if associative.exists():
        head_files["associative_direct"]=associative
    heads = {name: (tensor(p,"classifier.weight"), tensor(p,"classifier.bias")) for name,p in head_files.items()}
    predictions = {name: [] for name in heads}
    backend_rows = Counter()
    uncertain = set()
    # Use two already-replayed blocks to estimate a conservative margin band for
    # each linear head. Re-run potential ties through PyTorch before scoring.
    thresholds = {name: 0.001 for name in heads}
    different_engine = [v for v in set(paths.values()) if v != directory]
    if different_engine:
        for variant in different_engine:
            parity = json.loads((ROOT/"results"/(variant.name+"-parity.json")).read_text())
            if not parity["comparable"] or parity["source"] != BASE_HASH:
                raise ValueError("Unverified FP32 inference backend")
            max_difference = max(probe["max_absolute_feature_difference"] for probe in parity["blocks"])
            for name,(w,_) in heads.items():
                thresholds[name] = max(thresholds[name],4*max_difference*float(np.linalg.norm(w,ord=1,axis=1).max()))
    for i in range(expected):
        chunk = rows[i*BLOCK:(i+1)*BLOCK]
        directory_for_block = paths[i]
        if not verify_block(directory_for_block, i, fingerprint([r[2] for r in chunk]), len(chunk)):
            raise ValueError(f"Missing block {i}")
        metadata = json.loads((directory_for_block / f"{i:04d}.json").read_text())
        backend_rows[metadata["device"]] += len(chunk)
        x = np.load(directory_for_block / f"{i:04d}.npy", allow_pickle=False)
        for name, (w, b) in heads.items():
            out = x @ w.T + b
            predictions[name].extend(out.argmax(1))
            if directory_for_block != directory:
                top_values = np.partition(out, -2, axis=1)[:, -2:]
                uncertain.update(i*BLOCK+int(j) for j in np.flatnonzero(top_values[:,1]-top_values[:,0] <= thresholds[name]))
    predictions = {name: np.asarray(values) for name, values in predictions.items()}
    if uncertain:
        from encoder import components, features
        ids = sorted(uncertain)
        print(f"Replaying {len(ids)} near-tie rows through the original PyTorch encoder",flush=True)
        model, tok, device = components()
        check = features(model,tok,device,[rows[i][2] for i in ids],"official-fp32-near-ties")
        for name, (w,b) in heads.items():
            predictions[name][ids] = (check @ w.T + b).argmax(1)
    base, edited = predictions["original"], predictions["direct"]
    assert len(edited) == len(truth)
    additional = {name: {"sha256":file_hash(head_files[name]),"metrics":metric(truth,pred),
                         "vs_author_released":compare(rows,truth,pred,release)}
                  for name,pred in predictions.items() if name not in ("original","direct")}
    whole={}
    coverage=ROOT/"results/window-coverage-test.json"
    if coverage.is_file():
        spec=json.loads(coverage.read_text())
        if spec["whole_population_fingerprint"]!=fingerprint([r[2] for r in rows]):
            raise ValueError("Full-window population changed")
        saved=np.load(ROOT/"cache"/spec["feature_file"],allow_pickle=False)
        idx,x=saved["indices"],saved["values"]
        for name,(w,b) in heads.items():
            if name not in ("original","direct","associative_direct"):continue
            pred=predictions[name].copy()
            pred[idx]=(x@w.T+b).argmax(1)
            whole[name]={"metrics":metric(truth,pred),"changed_vs_tail":int(np.count_nonzero(pred!=predictions[name]))}
    return {"base_checkpoint_sha256": BASE_HASH, "edited_checkpoint_sha256": DIRECT_HASH, "backend_rows":dict(backend_rows),"product_full_excerpt":whole,
            "pytorch_near_tie_replays":len(uncertain),"margin_thresholds":thresholds, "other_frozen_heads":additional,
            "official_encoder_replay": metric(truth, base), "edited": metric(truth, edited),
            "edited_vs_replayed": compare(rows, truth, edited, base),
            "edited_vs_author_released": compare(rows, truth, edited, release),
            "replay_vs_author_released": compare(rows, truth, base, release)}


def score_lexical(rows, truth, release):
    import fasttext
    from lexical_trial import TARGET, scores
    fit = json.loads((TARGET / "fit.json").read_text())
    frozen = json.loads((TARGET / "inference.json").read_text())
    assert file_hash(TARGET / "model.bin") == fit["sha256"] == frozen["source_sha256"]
    model = fasttext.load_model(str(TARGET / "model.bin"))
    prior_adjustment = frozen["adjustment"] * np.asarray(frozen["train_log_prior"])
    pred = np.array([np.argmax(scores(model, row[2]) - prior_adjustment) for row in rows])
    return {"fit_sha256": fit["sha256"], "selected_adjustment": frozen["adjustment"],
            "model": metric(truth, pred), "vs_author_released": compare(rows, truth, pred, release)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["direct", "lexical"])
    args = parser.parse_args()
    rows, truth, release = official()
    result = (score_bert if args.model == "direct" else score_lexical)(rows, truth, release)
    report = {"split": "official test.csv", "rows": len(rows), "judgments": len({r[1] for r in rows}),
              "official_sha256": FILES["test.csv"][1], "metric": "unweighted macro-averaged F1 over all seven roles",
              "published_sentence_only": metric(truth, release), "frozen_model": result}
    path = ROOT / "results" / f"{args.model}-official-test.json"
    save_json(path, report)
    selected = result["edited"] if args.model == "direct" else result["model"]
    print(json.dumps({"model": args.model, "macro_f1": selected["macro_f1"], "accuracy": selected["accuracy"],
                      "rows": report["rows"]}))


if __name__ == "__main__":
    main()
