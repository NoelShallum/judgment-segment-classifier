"""Materialize either documented FP32 InLegalBERT pooler patch on the pinned public donor."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct

import numpy as np
from safetensors.numpy import load_file


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def apply(donor,patch,receipt,output):
    if sha(donor)!=receipt["donor_sha256"] or sha(patch)!=receipt["patch_sha256"]:
        raise ValueError("Source or patch checksum mismatch")
    if output.resolve()==donor.resolve() or output.exists():
        raise FileExistsError("Choose a new output checkpoint path")
    with donor.open("rb") as f:
        n=struct.unpack("<Q",f.read(8))[0]
        header=json.loads(f.read(n))
    tensors=load_file(str(patch))
    if set(tensors)!={"bert.pooler.dense.weight","bert.pooler.dense.bias"}:
        raise ValueError("Expected both physical pooler tensors")
    output.parent.mkdir(parents=True,exist_ok=True)
    import tempfile
    with tempfile.NamedTemporaryFile(dir=output.parent,prefix=output.name+".",delete=False) as temp:
        partial=Path(temp.name)
    try:
        shutil.copyfile(donor,partial)
        with partial.open("r+b") as f:
            for key,value in tensors.items():
                spec=header[key]
                if spec["dtype"]!="F32" or tuple(spec["shape"])!=tuple(value.shape) or not np.isfinite(value).all():
                    raise ValueError(f"Invalid tensor {key}")
                f.seek(8+n+spec["data_offsets"][0])
                f.write(np.ascontiguousarray(value,dtype="<f4").tobytes())
        if sha(partial)!=receipt["result_sha256"]:
            raise ValueError("Patched checkpoint checksum mismatch")
        # Exclusive publication avoids replacing a concurrent user's file.
        output.hardlink_to(partial)
    finally:
        partial.unlink(missing_ok=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("donor",type=Path)
    p.add_argument("patch",type=Path)
    p.add_argument("receipt",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    apply(args.donor,args.patch,json.loads(args.receipt.read_text()),args.output)
    print(sha(args.output))
