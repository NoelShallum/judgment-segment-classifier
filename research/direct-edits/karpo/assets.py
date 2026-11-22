"""Fetch and verify pinned public Karpo artifacts without credentials."""
import argparse
from hashlib import file_digest, sha1
import json
from pathlib import Path
import subprocess
from urllib.parse import quote
from urllib.request import urlopen


HERE = Path(__file__).resolve().parent
MODEL = HERE / "model"
REPO = "MachinePulseAI/karpo-decision-0.8B"
REVISION = "fae75d7bf93fc7e5c37b1ff631b64e9474e4d797"
UPSTREAM = "6600011eb5cf46232d056ec0717dc726146d29de"
SHARD = "model-00001-of-00001.safetensors"
FILES = ("LICENSE", "NOTICE", "licenses/Qwen3.5-LICENSE", "chat_template.jinja", "config.json", "decision_config.json",
         "model.safetensors.index.json", "processor_config.json", "readout.safetensors", "tokenizer.json", "tokenizer_config.json", SHARD)
WEIGHT_SHA = "656bc586f18e9c26df29033ebd50036b33004de82b2c61e4a670124e88d3143f"
READOUT_SHA = "e290da9edde55094aff5aa26c7b6d0f5a65b08eac590570af35a2c6fb0d6983e"


def checksum(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def write(path, value):
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text())


def public_manifest():
    out = HERE / "source.json"
    if out.exists():
        return read(out)
    with urlopen(f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true", timeout=45) as stream:
        doc = json.load(stream)
    assert doc["sha"] == REVISION and doc["safetensors"]["parameters"] == {"BF16": 852985920}
    siblings = {f["rfilename"]: f for f in doc["siblings"]}
    assert siblings[SHARD]["lfs"]["sha256"] == WEIGHT_SHA
    assert siblings["readout.safetensors"]["lfs"]["sha256"] == READOUT_SHA
    record = {"repo": REPO, "revision": REVISION, "source_commit": UPSTREAM,
              "license": doc["cardData"]["license"], "backbone_parameters": doc["safetensors"]["total"],
              "files": {f: {"size": siblings[f]["size"], "git_blob": siblings[f]["blobId"],
                              "sha256": siblings[f].get("lfs", {}).get("sha256")} for f in FILES}}
    write(out, record)
    return record


def check(path, record):
    if not path.is_file() or path.stat().st_size != record["size"]:
        return False
    if record["sha256"] is not None:
        return checksum(path) == record["sha256"]
    with path.open("rb") as stream:
        h = sha1(f'blob {record["size"]}\0'.encode())
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest() == record["git_blob"]


def assets(include_backbone):
    manifest = public_manifest()
    assert manifest["repo"] == REPO and manifest["revision"] == REVISION
    for name in FILES:
        if name == SHARD and not include_backbone:
            continue
        path = MODEL / name
        record = manifest["files"][name]
        if check(path, record):
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".download")
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{quote(name)}?download=true"
        subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "3",
                        "--retry-delay", "2", "--connect-timeout", "30", "--output", str(temp), url], check=True)
        if not check(temp, record):
            raise ValueError(f"Pinned publisher artifact checksum failed for {name}")
        temp.replace(path)
        print("Verified", name, path.stat().st_size, checksum(path), flush=True)
    return MODEL


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backbone", action="store_true")
    assets(cli.parse_args().backbone)


if __name__ == "__main__":
    main()
