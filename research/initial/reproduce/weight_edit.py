"""Direct, gradient-free edits to the published seven-output LegalSeg tensor file."""
import hashlib
import json
import os
import shutil
import struct
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
ORIGINAL=ROOT/"references/checkpoint/InLegalBERT(i)model.safetensors"
LABELS=("none","facts","issue","petitioner_argument","respondent_argument","reasoning","decision")
# Published TRAIN column, LegalSeg Table 2. No dev or test frequencies.
TRAIN_COUNTS=np.array([603059,169653,12791,64987,50097,202346,19574])

def file_hash(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def offsets(path):
    with Path(path).open("rb") as f:
        length=struct.unpack("<Q",f.read(8))[0]
        header=json.loads(f.read(length))
    return 8+length,header

def tensor(path,key):
    start,header=offsets(path)
    spec=header[key]
    if spec["dtype"]!="F32":raise ValueError("Expected FP32 "+key)
    return np.memmap(path,mode="r",dtype="<f4",offset=start+spec["data_offsets"][0],shape=tuple(spec["shape"])).copy()

def prior_bias(strength=0.35):
    log_counts=np.log(TRAIN_COUNTS.astype(float))
    return np.asarray(-strength*(log_counts-log_counts.mean()),dtype=np.float32)

def patched_checkpoint(source,target,changes):
    """Copy identical bytes, then replace only named, same-shape FP32 tensors."""
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if Path(source).resolve()==target.resolve(): raise ValueError("Target must be a copy.")
    temp=target.with_name(target.name+".partial")
    start,header=offsets(source)
    checked={}
    for k,a in changes.items():
        a=np.ascontiguousarray(a,dtype="<f4")
        if header[k]["dtype"]!="F32" or tuple(header[k]["shape"])!=a.shape or not np.isfinite(a).all():raise ValueError(k)
        checked[k]=a
    shutil.copyfile(source,temp)
    with temp.open("r+b") as f:
        for k,a in checked.items():
            f.seek(start+header[k]["data_offsets"][0]);f.write(a.tobytes())
        f.flush();os.fsync(f.fileno())
    for k,a in checked.items():
        if not np.array_equal(tensor(temp,k),a):raise ValueError("Write verification failed: "+k)
    os.replace(temp,target)
    return {k:{"shape":list(a.shape),"changed":int(np.count_nonzero(a!=tensor(source,k)))} for k,a in checked.items()}

def main():
    directory=ROOT/"checkpoints/direct-prior"
    old=tensor(ORIGINAL,"classifier.bias")
    delta=prior_bias()
    target=directory/"model.safetensors"
    changes=patched_checkpoint(ORIGINAL,target,{"classifier.bias":old+delta})
    card={"method":"direct published-training-prior edit","coefficient":0.35,"labels":LABELS,"train_counts":TRAIN_COUNTS.tolist(),
          "recipe":"delta[c] = -0.35 * (log(train_count[c]) - mean(log(train_counts)))",
          "training":"No optimizer, gradients, benchmark test predictions or test text used to derive this edit.",
          "changes":changes,"delta_bias":delta.tolist(),"before_sha256":file_hash(ORIGINAL),"after_sha256":file_hash(target)}
    (directory/"edit.json").write_text(json.dumps(card,indent=2)+"\n")
    print(json.dumps(card,indent=2))
if __name__=="__main__":main()
