"""Package the locked score, physical edits, and matched boundary diagnostics."""
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


HERE = Path(__file__).resolve().parent
SOURCE = (
    "RESULTS.md", "EXPERIMENT.md", "T4.md", "protocol.json", "analytic-specification.json",
    "classification-sample.json", "span-repair-protocol.json", "collect.py", "joint_solver.py",
    "score.py", "span_headroom.py", "verify_edits.py", "classifier64.py", "complementarity.py",
    "preregister.py", "t4_worker.py", "make_summary.py", "deps/gliner2-wheel.json", "model/source.json", "fetch.log",
)
REQUIRED = (
    "freeze.json", "freeze-sha256.json", "check.json", "four.json", "sixteen.json",
    "sixteen-predictions.json", "sixteen-complementarity.json", "sixteen-span-headroom.json",
    "sixteen-span-repair-predictions.json", "forward-parity.json", "edit-delta.safetensors",
    "edit-replacement.safetensors", "shuffled-delta.safetensors", "shuffled-replacement.safetensors",
)
OPTIONAL = (
    "publisher-classify-64.json", "four-complementarity.json", "four-span-headroom.json",
    "early-training-only.json", "early-joint.json",
)


def checksum(data):
    return hashlib.sha256(data).hexdigest()


def main():
    results = HERE / "results"
    paths = [HERE / name for name in SOURCE] + [results / name for name in REQUIRED]
    paths += [results / name for name in OPTIONAL if (results / name).exists()]
    manifest = {path.relative_to(HERE).as_posix(): checksum(path.read_bytes()) for path in paths}
    target = HERE / "gliner2-boundary-evidence.zip"
    partial = target.with_suffix(".tmp")
    with ZipFile(partial, "w", compression=ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(HERE))
        archive.writestr("manifest.sha256.json", json.dumps(manifest, indent=2))
    with ZipFile(partial) as archive:
        assert archive.testzip() is None
        assert all(checksum(archive.read(name)) == digest for name, digest in manifest.items())
    partial.replace(target)
    print(target.name, len(paths), target.stat().st_size, checksum(target.read_bytes()))


if __name__ == "__main__":
    main()
