"""Build the transfer artifact from the exact frozen case rows and pinned runtime."""

import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import json

HERE = Path(__file__).resolve().parent


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def main():
    lock = json.loads((HERE/"protocol.json").read_text())
    members = {name:HERE/name for name in ("protocol.json","review.json","bert-config.json","encode.py","fit.py","run_t4.py","viterbi.cpp")}
    members["native.py"] = HERE.parent/"bert-t4/native.py"
    for keys in lock["groups"].values():
        for key in keys:
            name = f"cases/{key}.npz"
            members[name] = HERE/name
    out = HERE/"packed-worker-v2.zip"
    stage = out.with_suffix(".part")
    checks = {name:sha(path) for name,path in members.items()}
    with ZipFile(stage,"w",compression=ZIP_DEFLATED,compresslevel=3) as archive:
        for name,path in members.items():
            archive.write(path,name)
        archive.writestr("contents-sha256.json",json.dumps(checks,indent=2))
    stage.replace(out)
    with ZipFile(out) as archive:
        assert archive.testzip() is None
    print(out,out.stat().st_size,sha(out),flush=True)


if __name__ == "__main__":
    main()
