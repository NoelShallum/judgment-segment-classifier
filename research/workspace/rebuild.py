"""Recreate the edited model from its public predecessor and the bundled replacement head."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request

import numpy as np

from weight_edit import patched_checkpoint, file_hash

SOURCE_URL = ("https://huggingface.co/L-NLProc/LegalSeg_InLegalBERT/resolve/"
              "44d7925e5ab422ecf04acd5dcbb0b62a6f8d8f53/InLegalBERT/InLegalBERT%28i%29model.safetensors")
SOURCE_HASH = "bbe8af20d1524c7535b1cbe68c6d4c3c844c6cc9fccef46c03c9ff56e7c18cea"
TARGET_HASH = "0a32fe939d96384c3cf3d879e5cfbce76c67172a8f3920961a99df3ba65b7edf"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Existing original safetensors file. Omit to download it.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    kit = Path(__file__).resolve().parent
    dest = args.output_dir.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    result = dest / "model.safetensors"
    already_built = result.is_file() and file_hash(result) == TARGET_HASH
    if result.exists() and not already_built:
        raise SystemExit("The output directory already contains a different model.safetensors.")
    if not already_built:
        with tempfile.TemporaryDirectory(dir=dest) as scratch:
            source = args.source
            if source is None:
                source = Path(scratch) / "published.safetensors"
                with urllib.request.urlopen(SOURCE_URL, timeout=120) as response, source.open("wb") as stream:
                    shutil.copyfileobj(response, stream, length=1024*1024)
            if file_hash(source) != SOURCE_HASH:
                raise SystemExit("Original checkpoint checksum mismatch.")
            with np.load(kit / "head.npz", allow_pickle=False) as head:
                amended = Path(scratch) / "edited.safetensors"
                patched_checkpoint(source, amended, {k: head[k] for k in ("classifier.weight", "classifier.bias")})
            if file_hash(amended) != TARGET_HASH:
                raise SystemExit("Rebuilt checkpoint checksum mismatch.")
            os.replace(amended, result)
    if dest != kit:
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                     "vocab.txt", "predict.py", "windowing.py", "edit.json", "README.md", "NOTICE", "LICENSE",
                     "TOKENIZER-LICENSE.txt", "requirements.txt"):
            shutil.copyfile(kit / name, dest / name)
    print(f"Verified {result} (SHA-256 {TARGET_HASH})")


if __name__ == "__main__":
    main()
