"""Run the frozen full-test direct edit in an authenticated ephemeral Colab runtime."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def invoke(*args):
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main():
    os.environ.update(OPENBLAS_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device attached to this Colab session")
    print("GPU", torch.cuda.get_device_name(0), flush=True)
    invoke("-m", "pip", "install", "--quiet", "transformers==4.53.3", "tokenizers==0.21.4", "safetensors", "scipy")
    manifest = json.loads((ROOT / "public-manifest.json").read_text())
    for item in manifest:
        if item["owner"] not in {"checkpoint", "tokenizer", "baseline_sentence"}:
            continue
        if item["owner"] == "baseline_sentence" and "(i)_" not in item["file"]:
            continue
        target = ROOT / item["path"]
        if target.is_file() and digest(target) == item["sha256"]:
            continue
        target.parent.mkdir(exist_ok=True, parents=True)
        temp = target.with_name(target.name + ".part")
        import shutil
        with urllib.request.urlopen(item["url"], timeout=120) as response, temp.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024*1024)
        if digest(temp) != item["sha256"]:
            raise ValueError(f"Upstream checksum mismatch for {item['file']}")
        temp.replace(target)
    original = ROOT / "references/checkpoint/InLegalBERT(i)model.safetensors"
    invoke("portable-edit/rebuild.py", "--source", str(original), "--output-dir", str(ROOT / "checkpoints/selected-direct"))
    invoke("-u", "cache_full.py", "--split", "test", "--device", "cuda")
    invoke("score_official_test.py", "direct")
    print("Full test complete", flush=True)


if __name__ == "__main__":
    main()
