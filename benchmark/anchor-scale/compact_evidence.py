"""Copy independently scorable records while preserving the source moment checksums."""

import argparse
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from fit_anchor import HERE, hash_file
from verify_stages import audit, digest


def compact(source, destination):
    size = int(source.stem.split("-")[1])
    audit(HERE / f"t4-anchor-{size}.zip", source)
    stage = destination.with_suffix(".part")
    with ZipFile(source) as original, ZipFile(stage, "w", ZIP_DEFLATED) as lean:
        intact = json.loads(original.read("SHA256SUMS.json"))
        keep, omit = {}, {}
        for member, sha in intact.items():
            if Path(member).name.startswith("moments-"):
                omit[member] = sha
            else:
                lean.writestr(member, original.read(member))
                keep[member] = sha
        metadata = json.dumps({"source_archive_sha256": hash_file(source), "omitted_fit_moments": omit}, indent=2) + "\n"
        lean.writestr("archived-fit-moments.json", metadata)
    with ZipFile(stage) as lean:
        for member, sha in keep.items():
            assert digest(lean, member) == sha
        keep["archived-fit-moments.json"] = digest(lean, "archived-fit-moments.json")
    with ZipFile(stage, "a", ZIP_DEFLATED) as lean:
        lean.writestr("SHA256SUMS.json", json.dumps(keep, indent=2) + "\n")
    stage.replace(destination)
    audit(HERE / f"t4-anchor-{size}.zip", destination)
    print(destination, destination.stat().st_size, hash_file(destination), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("files", type=Path, nargs="+")
    args = cli.parse_args()
    for source in args.files:
        destination = source.with_name(source.stem.replace("-dev", "-evidence") + ".zip")
        assert destination != source
        compact(source, destination)
