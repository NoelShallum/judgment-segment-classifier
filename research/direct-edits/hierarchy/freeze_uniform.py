"""Serialize the validated 200-float physical edit and fix validation IDs before scoring."""
import csv
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file,save_file
from lex_counts import HERE,ROOT,file_hash,dump

def freeze():
    src=ROOT/"references/hier/hier.safetensors";f=load_file(str(src))
    p=np.load(HERE/"forget-plan.npz");gain=.8;deltas=(p["uniform"][:,0]*gain).astype(np.float32)
    gate=["emitter.lstm.bias_ih_l0","emitter.lstm.bias_ih_l0_reverse"];out={}
    for k,d in zip(gate,deltas):
        value=f[k].copy();value[100:200]+=d;out[k]=value
    tmp=HERE/"uniform-retention.part";save_file(out,str(tmp));path=HERE/"uniform-retention.safetensors";tmp.replace(path)
    val=ROOT/"references/official/val.csv"
    with val.open(newline="",encoding="utf-8-sig") as fp:ids={r["Index"] for r in csv.DictReader(fp)}
    # Exclude any quoted numerical judgment ID found anywhere in the earlier receipts; false positives only reduce eligibility.
    broad=set()
    for directory in (ROOT/"results",ROOT/"astra-role-edge"):
        if not directory.exists():continue
        for receipt in directory.glob("*.json"):
            for token in re.findall(rb'"(\d{4})"',receipt.read_bytes()):
                key=token.decode()
                if key in ids:broad.add(key)
    for dirname in ("known-sentence-pair-edit","hier-emitter-edits-v1","whole-joint-states"):
        broad.update(p.stem.rsplit("-",1)[-1] for p in (ROOT/"cache"/dirname).glob("*.npy"))
    shared=json.loads((HERE.parent/"shared-confirmation.json").read_text())["extended_validation_judgment_ids"]
    all_selected=set(shared)
    for split in ("select","confirm","holdout"):
        with np.load(ROOT/"cache/sentence-recurrent-inverse"/(split+".npz")) as f_:all_selected.update(f_["keys"])
    untouched=ids-broad-all_selected
    family_only=ids-all_selected
    population=untouched if len(untouched)>=32 else family_only
    order=sorted(population,key=lambda k:hashlib.sha256(("attoflow-physical-forget-v1|"+k).encode()).digest())
    info=dict(train_plan_sha256=file_hash(HERE/"forget-plan.npz"),frozen_sha256=file_hash(path),source_sha256=file_hash(src),slices={k:[100,200] for k in gate},
        deltas=deltas.tolist(),gradient_steps=0,total_native_weights_changed=200,selection_ref=file_hash(HERE/"forget-frozen.json"),validation_source_sha256=file_hash(val),
        documented_historical_nonexposure_eligible=len(untouched),family_disjoint_eligible=len(family_only),ids=order[:48],
        selection_basis="reserved by hash, no human outcomes or text used",historical_exposure="No numeric ID matched any of previous receipts or primary state caches" if population is untouched else "Historically exposed IDs possible; disjoint from family selection, prior shared 48 and major confirm/holdout cohorts")
    dump(HERE/"uniform-retention-prospective.json",info)
    print(deltas,info["documented_historical_nonexposure_eligible"],info["family_disjoint_eligible"],len(info["ids"]),info["historical_exposure"],flush=True)
if __name__=="__main__":freeze()
