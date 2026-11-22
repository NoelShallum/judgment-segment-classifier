"""Package the audited source snapshot, pinned metadata, and benchmark inputs."""
import argparse
from hashlib import sha256
from zipfile import ZIP_DEFLATED, ZipFile

from assets import HERE, MODEL, SHARD, UPSTREAM, checksum, public_manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=("smoke", "full"))
    phase = ap.parse_args().phase
    source = HERE / ("karpo-decision-" + UPSTREAM)
    files = {name: HERE / name for name in ("assets.py", "runner.py", "source.json", "inputs.json", "protocol.json",
                                            "requirements-t4.txt", "t4.sh", "REMOTE.md")}
    files["require_t4.py"] = HERE.parent / "decision" / "require_t4.py"
    files["publisher/LICENSE"] = source / "LICENSE"
    for path in sorted((source / "src").rglob("*.py")):
        files["publisher/" + path.relative_to(source / "src").as_posix()] = path
    for name in public_manifest()["files"]:
        if name != SHARD:
            files["model/" + name] = MODEL / name
    if phase == "full":
        for name in ("program.py", "truth.json", "incumbent.json"):
            files[name] = HERE / name
        if (HERE / "run" / "token-plan.json").is_file() and (HERE / "run" / "token-layout").is_dir():
            files["token-plan.json"] = HERE / "run" / "token-plan.json"
            for file in sorted((HERE / "run" / "token-layout").rglob("*.json")):
                files[file.relative_to(HERE / "run").as_posix()] = file
    stage = HERE / f"karpo-t4-{phase}-worker.zip"
    tmp = stage.with_suffix(".part")
    digests = {name: checksum(file) for name, file in files.items()}
    with ZipFile(tmp, "w", compression=ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, name)
        archive.writestr("SHA256SUMS", "\n".join(f"{digest}  {name}" for name, digest in sorted(digests.items())) + "\n")
    with ZipFile(tmp) as archive:
        assert archive.testzip() is None
        assert all(sha256(archive.read(name)).hexdigest() == value for name, value in digests.items())
    tmp.replace(stage)
    print(stage.name, len(files), stage.stat().st_size, checksum(stage), flush=True)


if __name__ == "__main__":
    main()
