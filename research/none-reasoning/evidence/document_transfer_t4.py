"""Replay fixed standalone patches through the real whole-judgment reviewer."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

root=Path("/content/reviewer2")
package=root/"none-reasoning-input.zip"
with package.open("rb") as source:
    assert hashlib.file_digest(source,"sha256").hexdigest()==os.environ["ATTOFLOW_PAIR_PACKAGE"]
data=root/"pair-document-data"
data.mkdir(exist_ok=True)
with ZipFile(package) as z:
    assert z.testzip() is None
    assert all(not PurePosixPath(n).is_absolute() and ".." not in PurePosixPath(n).parts for n in z.namelist())
    manifest=json.loads(z.read("MANIFEST.json"))
    assert set(z.namelist())==set(manifest)|{"MANIFEST.json"}
    for name,sha in manifest.items():
        assert hashlib.sha256(z.read(name)).hexdigest()==sha
    z.extractall(data)
sys.path.insert(0,str(data))
sys.path.insert(1,str(root/"deps"))

import numpy as np
from safetensors.torch import load_file
import torch
from fit import Readout, prob, summarize
from native import batches, donor_at, move, read, require_t4, sha


@torch.inference_mode()
def main():
    device=require_t4()
    torch.set_num_threads(1)
    assert (root/"bootstrap.ok").is_file()
    plan=read(data/"round2-plan.json")
    cases=plan["groups"]["fresh"]
    source=[]
    for case in cases:
        with np.load(data/"cases"/(case+".npz"),allow_pickle=False) as z:
            assert z["case_sha256"].item()==plan["case_text_sha256"][case]
            source.append({k:z[k] for k in ("ids","mask","positions","base","original","aux","prior","gold")})
    sizes=[len(r["gold"]) for r in source]
    starts=np.r_[0,np.cumsum(sizes)]
    cat=lambda k:np.concatenate([r[k] for r in source])
    bank=SimpleNamespace(cases=cases,starts=starts,gold=cat("gold"),parent=cat("base"),
        review=np.concatenate([s+r["positions"] for s,r in zip(starts,source)]),
        original=cat("original"),prior=cat("prior"),aux=move(cat("aux"),"cuda"))
    ids, mask=cat("ids"),cat("mask")
    assert len(ids)==len(bank.review)==len(bank.original)==len(bank.prior)
    with np.load(data/"prior-predictions.npz",allow_pickle=False) as z:
        np.testing.assert_array_equal(z["case_ids"],cases)
        np.testing.assert_array_equal(z["gold"],bank.gold)
        np.testing.assert_array_equal(z["review"],bank.review)
        old=z["incumbent"]
    model=donor_at(data,root/"donor.safetensors","cuda")
    head=Readout(data,"cuda")
    original_weight,original_bias=head.weight.clone(),head.bias.clone()
    output={}
    for backbone in ("donor","neurons28"):
        if backbone=="neurons28":
            weights=load_file(str(data/"neurons28.safetensors"),device="cuda")
            layer=model.encoder.layer[-1]
            positions=weights["slots"].long()
            layer.intermediate.dense.weight[positions]=weights["intermediate_weight"]
            layer.intermediate.dense.bias[positions]=weights["intermediate_bias"]
            layer.output.dense.weight[:,positions]=weights["output_weight"]
        pooled=[]
        for count,sl in enumerate(batches(len(ids),96),1):
            got=model(move(ids[sl],"cuda",torch.long),attention_mask=move(mask[sl],"cuda",torch.bool)).pooler_output
            pooled.append(got)
            if count%12==0: print(backbone,"batches",count,flush=True)
        pool=torch.cat(pooled)
        for arm in ("baseline","targeted","shuffled"):
            if arm=="baseline":
                head.weight,head.bias=original_weight,original_bias
            else:
                patch=load_file(str(data/(backbone+"-"+arm+".safetensors")),device="cuda")
                head.weight,head.bias=patch["classifier.weight"],patch["classifier.bias"]
            probabilities=prob(head.raw(bank.aux,pool))
            output[backbone+"-"+arm]=head.review(bank,probabilities)
        print(backbone,{k:round(summarize(bank,v)["macro_f1"]*100,4) for k,v in output.items()},flush=True)
    baseline=output["donor-baseline"]
    drift=int(np.count_nonzero(baseline!=old))
    if drift:
        raise RuntimeError(f"Native baseline differs from archived predictions on {drift} rows.")
    result={"device":device,"case_count":len(cases),"sentences":len(bank.gold),"reviewed":len(bank.review),
            "cached_baseline_prediction_differences":drift,
            "freeze_sha256":sha(data/"pair-freeze.json"),
            "scores":{k:summarize(bank,v) for k,v in output.items()}}
    destination=root/"none-reasoning-document-returned.zip"
    temp=root/"none-reasoning-document-returned.part"
    import io
    buffer=io.BytesIO()
    np.savez_compressed(buffer,case_ids=np.asarray(cases),offsets=starts,review=bank.review,gold=bank.gold,**output)
    contents={"scores.json":json.dumps(result,indent=2,allow_nan=False).encode(),
              "predictions.npz":buffer.getvalue()}
    with ZipFile(temp,"w",ZIP_DEFLATED) as z:
        for k,v in contents.items():z.writestr(k,v)
        z.writestr("SHA256SUMS.json",json.dumps({k:hashlib.sha256(v).hexdigest() for k,v in contents.items()},indent=2))
    temp.replace(destination)
    print("DOCUMENT_DONE",destination,sha(destination),flush=True)


main()
