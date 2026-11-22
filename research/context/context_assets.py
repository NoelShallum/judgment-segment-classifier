"""Install and verify the three pinned public LegalSeg InLegalBERT donors."""

import argparse
import hashlib
import json
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile


HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models"
REPO = "L-NLProc/LegalSeg_InLegalBERT"
REVISION = "44d7925e5ab422ecf04acd5dcbb0b62a6f8d8f53"
DONORS = {
    "incumbent": ("InLegalBERT(i)model.safetensors", "bbe8af20d1524c7535b1cbe68c6d4c3c844c6cc9fccef46c03c9ff56e7c18cea"),
    "past": ("InLegalBERT(i-2, i-1, i)model.safetensors", "a81f509d250f98b44872e89e10186d5dcc17d37905b69aa1d9a077fb841a62ac"),
    "both": ("InLegalBERT(i-1, i, i+1)model.safetensors", "19cd593e269b1a57ffa500014d99b2fd9a76d2f76cf315d239a8fda205fc122f"),
}
FROZEN_SHA256 = "296fdffeb275516a2bde015edb4820984e468b9b3144904b18a8969a29c8a4da"
WEIGHTS = ("past-edited.safetensors", "past-pca.safetensors", "both-edited.safetensors", "both-pca.safetensors", "doc-semantic.safetensors")


def sha(path):
    with Path(path).open("rb") as src:
        return hashlib.file_digest(src, "sha256").hexdigest()


def verify_release():
    manifest = json.loads((HERE / "SHA256SUMS.json").read_text(encoding="utf-8"))
    for name, digest in manifest.items():
        path = (HERE / name).resolve()
        if not path.is_relative_to(HERE) or not path.is_file() or sha(path) != digest:
            raise ValueError(f"Release SHA-256 mismatch or missing file: {name}")
    frozen_path = HERE / "weights/freeze.json"
    if sha(frozen_path) != FROZEN_SHA256:
        raise ValueError("Unexpected frozen context model selection")
    freeze = json.loads(frozen_path.read_text(encoding="utf-8"))
    for f in WEIGHTS:
        if manifest["weights/" + f] != freeze["files"][f]:
            raise ValueError(f"Unexpected frozen context tensor file: {f}")
    return manifest


def verify_donor(kind, path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {kind} donor at {path}. Run python context_assets.py install.")
    if sha(path) != DONORS[kind][1]:
        raise ValueError(f"{kind} donor SHA-256 mismatch at {path}")
    return path


def donors_at(directory):
    return {name: verify_donor(name, directory / (name + ".safetensors")) for name in DONORS}


def install(name, dest, existing):
    if dest.exists():
        return verify_donor(name, dest)
    if existing is None:
        from huggingface_hub import hf_hub_download

        model_name, _ = DONORS[name]
        existing = hf_hub_download(REPO, filename="InLegalBERT/" + model_name, revision=REVISION, token=False)
    source = verify_donor(name, existing)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with NamedTemporaryFile(dir=dest.parent, prefix=name + "-", suffix=".part", delete=False) as output:
            temp = Path(output.name)
            with source.open("rb") as incoming:
                copyfileobj(incoming, output)
        verify_donor(name, temp)
        if dest.exists():
            return verify_donor(name, dest)
        temp.replace(dest)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("verify", help="Check the packaged files and optionally a donor directory")
    check.add_argument("--donor-dir", type=Path)
    fetch = commands.add_parser("install", help="Fetch or copy the pinned public donors")
    fetch.add_argument("--donor-dir", default=MODEL_DIR, type=Path)
    for key in DONORS:
        fetch.add_argument("--" + key + "-file", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        print(f"Verified {len(verify_release())} release files")
        if args.donor_dir is not None:
            for name, path in donors_at(args.donor_dir).items():
                print(name, path)
    else:
        for name in DONORS:
            print(name, install(name, args.donor_dir / (name + ".safetensors"), getattr(args, name + "_file")), flush=True)
