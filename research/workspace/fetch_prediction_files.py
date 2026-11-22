"""Verify every public LegalSeg prediction archive used for the expanded benchmark audit."""
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

import requests

from weight_edit import ROOT


META = ROOT / "references/papers/additional-predictions-meta.json"


def valid(data, spec):
    if len(data) != spec["size"]:
        return False
    if spec.get("lfs"):
        return hashlib.sha256(data).hexdigest() == spec["lfs"]["sha256"]
    blob = f"blob {len(data)}\0".encode() + data
    return hashlib.sha1(blob).hexdigest() == spec["blobId"]


def acquire(job):
    dataset, spec = job
    repo = dataset["repo"]
    filename = spec["rfilename"]
    url = f"https://huggingface.co/datasets/{repo}/resolve/{dataset['sha']}/{quote(filename)}"
    dest = ROOT / "references/all-predictions" / repo.split("/")[-1] / filename
    dest.parent.mkdir(exist_ok=True, parents=True)
    data = dest.read_bytes() if dest.exists() else b""
    if data.startswith(b"\x1f\x8b") and not valid(data, spec):
        data = gzip.decompress(data)
    if not valid(data, spec):
        response = requests.get(url, timeout=(15, 90))
        response.raise_for_status()
        data = response.content
    if not valid(data, spec):
        raise ValueError(f"Cannot verify publisher object {filename}")
    if not dest.is_file() or dest.stat().st_size != len(data):
        stage = dest.with_suffix(".part")
        stage.write_bytes(data)
        stage.replace(dest)
    header = data[:512].split(b"\n")[0].decode("utf-8-sig", errors="replace").strip()
    return {"repo": repo, "revision": dataset["sha"], "url": url, "local": str(dest.relative_to(ROOT)),
            "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "header": header}


def main():
    datasets = json.loads(META.read_text())
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(acquire, ((d, s) for d in datasets for s in d["files"])))
    (ROOT / "references/all-predictions/verified.json").write_text(json.dumps(results, indent=2) + "\n")
    for row in results:
        print(Path(row["local"]).name, row["header"])


if __name__ == "__main__":
    main()
