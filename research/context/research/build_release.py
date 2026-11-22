"""Assemble a portable CPU release from the frozen context experiment and base ZIP."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED, ZipFile

from safetensors.numpy import load_file

from context_assets import FROZEN_SHA256, HERE, WEIGHTS, sha


RUN = HERE.parent
ROOT = HERE.parents[2]
OUTPUT = ROOT.parents[1] / "outputs/judgment-role"
BASE_ZIP = OUTPUT / "AttoFlow-LegalSeg-reviewer-64-59.zip"
BASE_SHA256 = "91789b881e06d06de36b7e308c264df78f04269182c498c011a8eb4cdae240e4"
CONTEXT_ZIP = RUN / "legal-results.zip"
STAGING = HERE / "portable"


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def extract(archive, name, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(name) as source, dest.open("wb") as output:
        shutil.copyfileobj(source, output)


def assemble():
    STAGING.mkdir(exist_ok=True)
    assert sha(BASE_ZIP) == BASE_SHA256
    with ZipFile(BASE_ZIP) as base:
        assert base.testzip() is None
        for name in base.namelist():
            assert name and not Path(name).is_absolute() and ".." not in Path(name).parts
            extract(base, name, STAGING / "base" / name)
    for name in ("README.md", "classify.py", "context_assets.py", "verify_validation.py"):
        shutil.copy2(HERE / name, STAGING / name)
    for name in ("requirements.txt", "example.json"):
        shutil.copy2(STAGING / "base" / name, STAGING / name)
    initial_lock = json.loads((RUN / "official-lock.json").read_text(encoding="utf-8"))
    assert sha(CONTEXT_ZIP) == initial_lock["development_archive_sha256"]
    with ZipFile(CONTEXT_ZIP) as context:
        assert context.testzip() is None
        contents = json.loads(context.read("SHA256SUMS.json"))
        for filename in (*WEIGHTS, "freeze.json", "evaluation.json", "verification.json", "confirm-predictions.npz", "fresh-predictions.npz",
                         "select-predictions.npz", "legal_encode.py", "legal_solve.py", "legal_verify.py"):
            dest = "weights" if filename in WEIGHTS or filename == "freeze.json" else "research" if filename.endswith(".py") else "evidence"
            path = STAGING / dest / filename
            extract(context, filename, path)
            assert sha(path) == contents[filename]
    for name in ("past-fit.json", "both-fit.json", "past-select.json", "both-select.json", "past-confirm.json", "both-confirm.json",
                 "past-fresh.json", "both-fresh.json"):
        with ZipFile(CONTEXT_ZIP) as context:
            extract(context, name, STAGING / "evidence" / name)
            assert sha(STAGING / "evidence" / name) == contents[name]
    shutil.copy2(RUN.parent / "context-memory/solve.py", STAGING / "research/solve.py")
    shutil.copy2(HERE / "build_release.py", STAGING / "research/build_release.py")
    shutil.copy2(RUN / "official-lock.json", STAGING / "evidence/official-lock.json")
    frozen_path = STAGING / "weights/freeze.json"
    assert sha(frozen_path) == FROZEN_SHA256
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    assert frozen["selected"] == "doc-semantic"
    tensor_meta = {}
    for name in WEIGHTS:
        path = STAGING / "weights" / name
        assert sha(path) == frozen["files"][name]
        tensor_meta[name] = {key: {"shape": list(v.shape), "dtype": v.dtype.str, "sha256": hashlib.sha256(v.tobytes(order="C")).hexdigest()}
                             for key, v in sorted(load_file(path).items())}
    write(STAGING / "weights/tensors.json", tensor_meta)
    write(STAGING / "evidence/qa-plan.json", {"case_selection": "6045 includes every predicted role. 5922 exercises long-window pooling. Selection precedes CPU evaluation.",
                                             "cases": {"6045": "confirm", "5922": "fresh"},
                                             "validation_sha256": sha(ROOT / "references/official/val.csv"),
                                             "base_archive_sha256": BASE_SHA256, "evaluation_archive_sha256": sha(CONTEXT_ZIP)})


def archive(target):
    target.parent.mkdir(parents=True, exist_ok=True)
    items = {p.relative_to(STAGING).as_posix(): p for p in STAGING.rglob("*")
             if p.is_file() and p != STAGING / "SHA256SUMS.json" and "__pycache__" not in p.parts}
    sums = {name: sha(path) for name, path in sorted(items.items())}
    write(STAGING / "SHA256SUMS.json", sums)
    stage = target.with_suffix(".part")
    with ZipFile(stage, "w", ZIP_DEFLATED, compresslevel=4) as archive:
        for name, path in sorted(items.items()):
            archive.write(path, name)
        archive.write(STAGING / "SHA256SUMS.json", "SHA256SUMS.json")
    with ZipFile(stage) as archive:
        assert archive.testzip() is None and set(archive.namelist()) == set(items) | {"SHA256SUMS.json"}
        for name, digest in sums.items():
            with archive.open(name) as handle:
                assert hashlib.file_digest(handle, "sha256").hexdigest() == digest
    stage.replace(target)
    print(target, target.stat().st_size, sha(target), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("command", choices=("assemble", "archive"))
    cli.add_argument("--destination", default=OUTPUT / "AttoFlow-LegalSeg-context-model.zip", type=Path)
    args = cli.parse_args()
    if args.command == "assemble":
        assemble()
    else:
        archive(args.destination)
