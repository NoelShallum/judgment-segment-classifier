"""Evaluate the two train-scaled families after their selection receipts exist."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
stages = (8, 24, 72)

def wait_success(filename):
    sentinel = root / filename
    while not sentinel.exists() or not sentinel.read_text().strip():
        time.sleep(5)
    if sentinel.read_text().strip() != "0":
        raise RuntimeError(f"{filename} reports {sentinel.read_text().strip()}")

def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

def run(*args):
    subprocess.run([sys.executable, "-u", *args], cwd=root, check=True)

gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
if gpu not in {"T4", "Tesla T4", "NVIDIA T4"}:
    raise RuntimeError(f"Expected one T4, got {gpu!r}")

wait_success("run.exit")
wait_success("latent.exit")
scale_choice = json.loads((root / "selection.json").read_text())
receipts = {n: root / f"latent-{n}-freeze.json" for n in stages}
quality = {n: json.loads(p.read_text())["quality"] for n, p in receipts.items()}
chosen = max(stages, key=quality.__getitem__)
record = {
    "semantic_stage": chosen,
    "semantic_stage_selector_scores": quality,
    "stage_receipt_sha256": {n: sha(p) for n, p in receipts.items()},
    "context_stage": scale_choice["chosen"],
    "context_selection_sha256": sha(root / "selection.json"),
    "policy": "Highest predeclared selection quality in each family. Earlier stage wins an exact tie. Scores use only the 48 selector judgments.",
    "gpu": gpu,
}
dest = root / "paired-family-freeze.json"
encoded = json.dumps(record, indent=2) + "\n"
if dest.exists() and dest.read_text() != encoded:
    raise RuntimeError("Existing selection receipt differs")
dest.write_text(encoded)
print("Official families locked", record["context_stage"], chosen, sha(dest), flush=True)
run("run_scale.py", "official")
run("latent.py", str(chosen), "--official")
print("Finished both official evaluations", flush=True)
