"""Make a checked transfer package from the public corpus and frozen route records."""
import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from native import read, sha

HERE = Path(__file__).resolve().parent


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("priority", "full"))
    args = cli.parse_args()
    cfg = read(HERE/"protocol.json")
    g = cfg["groups"]
    members = [f"cases/{key}.npz" for key in (
        g["fit"][:96] + g["select"] + g["shared"][:16] if args.stage == "priority"
        else g["fit"] + g["select"] + g["confirm"] + g["shared"])]
    base = ["review.json","native.py","fit.py","run_t4.py","bert-config.json","protocol.json","donor-classifier.npz","head.safetensors"]
    output = HERE / (args.stage + "-worker.zip")
    temporary = output.with_suffix(".part")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=2) as archive:
        for name in base + members:
            archive.write(HERE/name, name)
    temporary.replace(output)
    with ZipFile(output) as archive:
        assert archive.testzip() is None
    print(output.name, output.stat().st_size, len(members), sha(output), flush=True)


if __name__ == "__main__":
    main()
