"""Verify release files and install the pinned public InLegalBERT donor."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from shutil import copyfileobj


HERE = Path(__file__).resolve().parent
REPO = "L-NLProc/LegalSeg_InLegalBERT"
REVISION = "44d7925e5ab422ecf04acd5dcbb0b62a6f8d8f53"
DONOR_NAME = "InLegalBERT/InLegalBERT(i)model.safetensors"
DONOR_SHA256 = "bbe8af20d1524c7535b1cbe68c6d4c3c844c6cc9fccef46c03c9ff56e7c18cea"
EDITED_SHA256 = "db701dc770fe3aa5cf26d3e83d89ff4e819df913be8c496e71bd949dc4dffd9b"
INCUMBENT_SHA256 = "e70462f5a9bd376efdab223ebee4ffc855877ea7bd159a394bb240e379a703c3"
DEFAULT_DONOR = HERE / "models/InLegalBERT.safetensors"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_donor(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing InLegalBERT donor at {path}. Run python assets.py install.")
    if sha(path) != DONOR_SHA256:
        raise ValueError(f"InLegalBERT donor SHA-256 mismatch for {path}. Expected {DONOR_SHA256}.")
    return path


def verify_release():
    manifest = json.loads((HERE / "SHA256SUMS.json").read_text(encoding="utf-8"))
    for name, expected in manifest.items():
        path = (HERE / name).resolve()
        if not path.is_relative_to(HERE) or not path.is_file() or sha(path) != expected:
            raise ValueError(f"SHA-256 mismatch or missing release file: {name}")
    if manifest["runtime/review/edited.safetensors"] != EDITED_SHA256:
        raise ValueError("The manifest does not identify the released edited readout")
    if manifest["runtime/review/incumbent.safetensors"] != INCUMBENT_SHA256:
        raise ValueError("The manifest does not identify the archived incumbent readout")
    return manifest


def install(output, existing):
    output = output.expanduser().resolve()
    if output.exists():
        return verify_donor(output)
    if existing is None:
        from huggingface_hub import hf_hub_download

        existing = Path(hf_hub_download(repo_id=REPO, revision=REVISION, filename=DONOR_NAME, token=False))
    source = verify_donor(existing)
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="wb", dir=output.parent, prefix="donor-", suffix=".part", delete=False) as stream:
        temp = Path(stream.name)
        try:
            with source.open("rb") as donor:
                copyfileobj(donor, stream)
            stream.flush()
            verify_donor(temp)
            if output.exists():
                return verify_donor(output)
            temp.replace(output)
        finally:
            temp.unlink(missing_ok=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("verify", help="Check all files listed in the release manifest")
    check.add_argument("--donor", type=Path, help="Also verify this local donor file")
    fetch = commands.add_parser("install", help="Install the pinned donor")
    fetch.add_argument("--from-file", type=Path, help="Use an already downloaded official donor")
    fetch.add_argument("--output", default=DEFAULT_DONOR, type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        print(f"Verified {len(verify_release())} release files")
        if args.donor is not None:
            print(f"Verified donor {verify_donor(args.donor)}")
    else:
        print(install(args.output, args.from_file))


if __name__ == "__main__":
    main()
