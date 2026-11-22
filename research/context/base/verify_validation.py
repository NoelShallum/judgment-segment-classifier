"""Compare fresh CPU CLI results with the frozen T4 predictions for two judgments."""

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from tempfile import TemporaryDirectory

from assets import HERE, sha, verify_donor, verify_release


def verify(validation, donor):
    spec = json.loads((HERE / "evidence/validation-fixtures.json").read_text(encoding="utf-8"))
    if sha(validation) != spec["csv_sha256"]:
        raise ValueError("Expected the pinned LegalSeg validation CSV")
    data = {case: [] for case in spec["judgments"]}
    with validation.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            text = row["Text"]
            if row["Index"] in data and text.strip():
                data[row["Index"]].append(text)
    verify_release()
    donor = verify_donor(donor)
    runs = {}
    env = {**os.environ, "PYTHONPATH": "", "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "TOKENIZERS_PARALLELISM": "false", "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2"}
    with TemporaryDirectory() as location:
        temp = Path(location)
        for case, fixture in spec["judgments"].items():
            path = temp / (case + ".json")
            assert len(data[case]) == len(fixture["edited"])
            path.write_text(json.dumps(data[case], ensure_ascii=False), encoding="utf-8")
            span_checks = {}
            for variant in ("incumbent", "edited"):
                cmd = [sys.executable, str(HERE / "classify.py"), "--input", str(path), "--donor", str(donor), "--head", variant]
                completed = subprocess.run(cmd, cwd=temp, env=env, capture_output=True, text=True)
                if completed.returncode:
                    raise RuntimeError(f"CLI error on {case}/{variant}: {completed.stderr}")
                result = json.loads(completed.stdout)
                expected = fixture[variant]
                if result["labels"] != expected or result["referrals"] != fixture["referrals"]:
                    where = [i for i, (a, b) in enumerate(zip(result["labels"], expected)) if a != b]
                    raise ValueError(f"T4 replay disagreement for {case}/{variant}: labels at {where}, "
                                     f"same route = {result['referrals'] == fixture['referrals']}")
                recovered = []
                for span in result["spans"]:
                    assert span["start_sentence"] == len(recovered)
                    stop = span["end_sentence"]
                    assert stop > len(recovered)
                    recovered.extend([span["label"]] * (stop - len(recovered)))
                assert recovered == expected
                span_checks[variant] = len(result["spans"])
            runs[case] = {"source": fixture["cohort"], "sentences": len(data[case]),
                          "same_route": True, "prediction_differences": 0,
                          "changed_by_patch": sum(x != y for x, y in zip(fixture["incumbent"], fixture["edited"])),
                          "spans": span_checks}
            print(f"Matched {case} ({len(data[case])} sentences) for both heads", file=sys.stderr, flush=True)
    return {"verified": runs, "platform": platform.platform(), "python": platform.python_version(),
            "validation_csv_sha256": spec["csv_sha256"], "manifest_sha256": sha(HERE / "SHA256SUMS.json"),
            "engine": "CPU PyTorch with native FP32 InLegalBERT, run through the public CLI"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--donor", type=Path, required=True)
    options = parser.parse_args()
    print(json.dumps(verify(options.validation, options.donor), indent=2))
