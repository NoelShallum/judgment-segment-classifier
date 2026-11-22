"""Check the released CPU classifier on two complete public validation judgments."""

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

import numpy as np

from context_assets import HERE, MODEL_DIR, donors_at, sha, verify_release
from classify import LABELS


def verify(source, donors, backend):
    expectation = json.loads((HERE / "evidence/qa-plan.json").read_text(encoding="utf-8"))
    if sha(source) != expectation["validation_sha256"]:
        raise ValueError("Expected the pinned LegalSeg validation CSV")
    verify_release()
    donors_at(donors)
    texts = {case: [] for case in expectation["cases"]}
    with source.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["Index"] in texts and row["Text"].strip():
                texts[row["Index"]].append(row["Text"])
    env = {**os.environ, "PYTHONPATH": "", "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "TOKENIZERS_PARALLELISM": "false", "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2"}
    results = {}
    with TemporaryDirectory() as td:
        temp = Path(td)
        for case, cohort in expectation["cases"].items():
            casefile = temp / (case + ".json")
            casefile.write_text(json.dumps(texts[case], ensure_ascii=False), encoding="utf-8")
            path = HERE / "evidence" / (cohort + "-predictions.npz")
            with np.load(path, allow_pickle=False) as expected:
                num = list(expected["case_ids"]).index(case)
                a, b = map(int, expected["offsets"][num:num + 2])
                target = [LABELS[i] for i in expected["doc-semantic"][a:b]]
                base = [LABELS[i] for i in expected["incumbent"][a:b]]
            assert len(texts[case]) == len(target)
            cli = "classify.py" if backend == "fp16" else "classify_fast.py"
            cmd = [sys.executable, str(HERE / cli), "--input", str(casefile), "--donor-dir", str(donors.resolve())]
            start = perf_counter()
            output = subprocess.run(cmd, cwd=temp, env=env, capture_output=True, text=True)
            if output.returncode:
                raise RuntimeError(f"CLI failed on judgment {case}: {output.stderr}")
            actual = json.loads(output.stdout)
            final = actual["labels"]
            assert len(final) == len(actual["incumbent_labels"]) == len(target)
            spans = []
            for block in actual["spans"]:
                assert block["start_sentence"] == len(spans) and block["end_sentence"] > len(spans)
                spans.extend([block["label"]] * (block["end_sentence"] - block["start_sentence"]))
            assert spans == final

            def diff(a, b):
                return [{"sentence": j, "cpu": y, "t4": z} for j, (y, z) in enumerate(zip(a, b)) if y != z]

            swapped = diff(final, target)
            original_swaps = diff(actual["incumbent_labels"], base)
            result = {"cohort": cohort, "sentences": len(target), "final_mismatches": swapped,
                      "incumbent_mismatches": original_swaps, "final_match": len(target) - len(swapped),
                      "incumbent_match": len(base) - len(original_swaps), "cpu_changes_from_incumbent": len(diff(final, actual["incumbent_labels"])),
                      "archived_changes_from_incumbent": len(diff(target, base)), "backbone_windows": actual["backbone_windows"],
                      "output_role_count": len(set(final)), "elapsed_seconds": round(perf_counter()-start, 2),
                      "expected_predictions_sha256": sha(path)}
            results[case] = result
            print(f"Judgment {case} · {result['final_match']}/{len(target)} labels matched · {result['elapsed_seconds']} s", file=sys.stderr, flush=True)
    return {"platform": platform.platform(), "python": platform.python_version(), "tested_on": f"CPU {backend.upper()}, ordered public validation sentences through {cli}",
            "validation_csv_sha256": sha(source), "package_manifest_sha256": sha(HERE / "SHA256SUMS.json"), "cases": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--donor-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--backend", choices=("fp16", "fp32"), default="fp16")
    args = parser.parse_args()
    print(json.dumps(verify(args.validation, args.donor_dir, args.backend), indent=2))
