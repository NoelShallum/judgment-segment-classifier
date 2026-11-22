"""Check real LegalSeg judgments through the extracted CPU inference entrypoint."""

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

from context_assets import HERE, MODEL_DIR, sha, verify_release


def smoke(csv_path, donor_dir):
    verify_release()
    plan = json.loads((HERE / "anchor-evidence/qa-expected.json").read_text(encoding="utf-8"))
    if sha(csv_path) != plan["validation_sha256"]:
        raise ValueError("Supply the pinned public LegalSeg validation CSV")
    text = {case: [] for case in plan["cases"]}
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["Index"] in text and row["Text"].strip():
                text[row["Index"]].append(row["Text"])
    env = dict(os.environ, PYTHONPATH="", CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2")
    scores = {}
    with TemporaryDirectory() as work:
        tmp = Path(work)
        for case, expect in plan["cases"].items():
            assert len(text[case]) == len(expect["labels"])
            payload = tmp / f"{case}.json"
            payload.write_text(json.dumps(text[case], ensure_ascii=False), encoding="utf-8")
            start = perf_counter()
            result = subprocess.run([sys.executable, str(HERE / "classify_anchor.py"), "--input", str(payload),
                                     "--donor-dir", str(donor_dir.resolve())], env=env, cwd=tmp, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(f"CPU inference for judgment {case} failed\n{result.stderr}")
            current = json.loads(result.stdout)
            assert current["head_sha256"] == plan["model_sha256"]
            assert current["anchor_arm"] == plan["arm"] and current["fit_judgments"] == plan["fit_judgments"]
            joined = []
            for piece in current["spans"]:
                assert piece["start_sentence"] == len(joined)
                joined.extend([piece["label"]] * (piece["end_sentence"] - piece["start_sentence"]))
            assert joined == current["labels"]
            changes = {}
            for key in ("labels", "context_labels", "incumbent_labels", "pre_review_labels"):
                assert len(current[key]) == len(expect[key])
                changes[key] = [{"position": i, "cpu": p, "archived": q} for i, (p, q) in enumerate(zip(current[key], expect[key])) if p != q]
            elapsed = round(perf_counter() - start, 2)
            scores[case] = {"sentences": len(joined), "predictions_sha256": expect["predictions_sha256"],
                            "backbone_windows": current["backbone_windows"], "mismatches": changes, "seconds": elapsed}
            print(f"{case}: {len(joined) - len(changes['labels'])}/{len(joined)} final labels, {elapsed}s", file=sys.stderr, flush=True)
    return {"tests": scores, "release_manifest_sha256": sha(HERE / "SHA256SUMS.json"), "model_sha256": plan["model_sha256"],
            "validation_sha256": plan["validation_sha256"], "executed": "Fresh extracted CLI on CPU in FP32 with FP16-rounded donor weights and pool outputs",
            "platform": platform.platform(), "python": platform.python_version()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--donor-dir", default=MODEL_DIR, type=Path)
    args = parser.parse_args()
    proof = smoke(args.validation, args.donor_dir)
    print(json.dumps(proof, indent=2, ensure_ascii=False))
    if any(m for each in proof["tests"].values() for m in each["mismatches"].values()):
        raise SystemExit(1)
