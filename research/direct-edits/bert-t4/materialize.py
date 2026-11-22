"""Write a replayable author-format checkpoint from the T4 study's checked patch."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
from safetensors.numpy import load

from weight_edit import file_hash, offsets, patched_checkpoint, tensor

VALUE = "bert.encoder.layer.11.attention.self.value"


def apply(donor, archive, arm, output):
    with ZipFile(archive) as bundle:
        patch = bundle.read(arm+".safetensors")
        frozen = json.loads(bundle.read("freeze.json"))
        assert file_hash(donor)==frozen["source_sha256"]
        assert hashlib.sha256(patch).hexdigest()==frozen["sha256"][arm+".safetensors"]
        edited = load(patch)
    keys = (VALUE+".weight",VALUE+".bias")
    assert set(edited)==set(keys)
    assert edited[keys[0]].shape==(768,768) and edited[keys[1]].shape==(768,)
    for name,rec in zip(keys,frozen["edit_tensor_stats"][arm],strict=True):
        assert hashlib.sha256(tensor(donor,name).tobytes()).hexdigest()==rec["source_tensor_sha256"]
        assert hashlib.sha256(edited[name].tobytes()).hexdigest()==rec["edited_tensor_sha256"]
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with TemporaryDirectory(dir=output.parent) as scratch:
        stage=Path(scratch)/"verified-checkpoint.safetensors"
        changed=patched_checkpoint(donor,stage,edited)
        donor_start,original=offsets(donor)
        patched_start,after=offsets(stage)
        assert donor_start==patched_start and original==after
        unaffected=hashlib.sha256()
        with donor.open("rb") as baseline,stage.open("rb") as result:
            for name,spec in original.items():
                if name=="__metadata__":
                    continue
                a,b=spec["data_offsets"]
                if name in edited:
                    np.testing.assert_array_equal(tensor(stage,name),edited[name])
                else:
                    baseline.seek(donor_start+a)
                    result.seek(patched_start+a)
                    checked=baseline.read(b-a)
                    assert result.read(b-a)==checked,name
                    unaffected.update(checked)
        output.hardlink_to(stage)
    receipt={"donor_sha256":file_hash(donor),"study_archive_sha256":file_hash(archive),
              "edited_checkpoint_sha256":file_hash(output),"unedited_tensor_bytes_sha256":unaffected.hexdigest(),
              "changed":changed,"arm":arm}
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("donor",type=Path)
    p.add_argument("archive",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--arm",choices=("targeted","shuffled"),default="targeted")
    a=p.parse_args()
    apply(a.donor,a.archive,a.arm,a.output)
