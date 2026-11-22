"""Pack local train/validation windows. Only the Colab coordinator uses --launch."""

import argparse
import json
from pathlib import Path
import subprocess
from zipfile import ZipFile, ZIP_STORED

from protocol import digest

HERE = Path(__file__).resolve().parent
SESSION = "attoflow-role-context-t4"
SOURCES = ("donor.json", "plan.json", "preparation.json", "protocol.py", "remote_capture.py", "windows.jsonl.gz")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--launch", action="store_true", help="Coordinator only. Existing single-T4 instance.")
    args = p.parse_args()
    package = HERE / "entailment-input.zip"
    stage = package.with_suffix(".zip.part")
    manifest = {name: digest(HERE / name) for name in SOURCES}
    with ZipFile(stage, "w", compression=ZIP_STORED) as z:
        for name in SOURCES:
            z.write(HERE / name, name)
        z.writestr("input-manifest.json", json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    stage.replace(package)
    fingerprint = digest(package)
    print(package, package.stat().st_size, fingerprint, flush=True)
    if args.launch:
        subprocess.run(["colab", "upload", "-s", SESSION, str(package), "/content/reviewer2/entailment-input.zip"], check=True)
        subprocess.run(["colab", "exec", "-s", SESSION, "--env", f"ATTO_ENTAIL_SHA256={fingerprint}",
                        "--file", str(HERE / "remote_launch.py")], check=True)


if __name__ == "__main__":
    main()
