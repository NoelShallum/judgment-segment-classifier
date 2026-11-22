import json
from pathlib import Path
import hashlib
import re
import numpy as np
from safetensors.numpy import load_file,save_file
from lex_counts import HERE,ROOT,dump,file_hash
def main():
    primary=HERE/"uniform-retention-prospective.json";plan=json.loads(primary.read_text())
    source=ROOT/"references/hier/hier.safetensors";assert file_hash(source)==plan["source_sha256"]
    original=load_file(str(source));target=load_file(str(HERE/"uniform-retention.safetensors"));ks=["emitter.lstm.bias_ih_l0","emitter.lstm.bias_ih_l0_reverse"]
    offset=np.concatenate([target[k][100:200]-original[k][100:200] for k in ks])
    rng=np.random.default_rng(82849216);positions=rng.choice(800,len(offset),replace=False)
    raw=np.concatenate([original[k] for k in ks]);shuf=raw.copy();shuf[positions]+=offset[rng.permutation(len(offset))]
    directions=raw.copy();ix=np.r_[np.arange(100,200),np.arange(500,600)];directions[ix]+=offset[rng.permutation(len(offset))]
    paths={}
    for name,w in (("gate-shuffled",shuf),("direction-shuffled",directions)):
        out={k:w[400*i:400*(i+1)].copy() for i,k in enumerate(ks)};p=HERE/f"uniform-{name}.safetensors";tmp=p.with_suffix(".part");save_file(out,str(tmp));tmp.replace(p);paths[name]=dict(file=p.name,sha256=file_hash(p),changed_entries=int(np.count_nonzero(w!=raw)),delta_L2=float(np.linalg.norm((w-raw).astype(float))))
    head=HERE/"native-base-240-71.safetensors"
    combine={**load_file(str(head)),**target}; dest=HERE/"uniform-retention-plus-head.safetensors";tmp=dest.with_suffix(".part");save_file(combine,str(tmp));tmp.replace(dest)
    freeze=dict(primary_cohort_file=primary.name,primary_cohort_sha256=file_hash(primary),primary_count=48,
          primary_gate="Before opening any of these 48 role outcomes, advance only if native targeted macro-F1 exceeds frozen head-only with a positive lower bound of a 95% judgment-cluster bootstrap, point macro-F1 exceeds the matched shuffled-gate control, and point accuracy is at least frozen head-only. The alternative direction-shuffle is a secondary specificity test; constants are positive in both directions, so that contrast is expected to be small.",
          assessment="The newly reserved 48 exclude every run in this experiment, major previous selection/confirmation/384-case holdout and the shared team 48. Historical exposure to unrelated programs is possible. Only six official-val IDs did not match any numeric strings in the historical receipt scan, so the cohort does not certify a pristine external benchmark.",
          bootstraps=4000,bootstrap_seed=220782,donor=file_hash(source),head_file=head.name,head_sha256=file_hash(head),target_file="uniform-retention.safetensors",target_sha256=file_hash(HERE/"uniform-retention.safetensors"),combined_file=dest.name,combined_sha256=file_hash(dest),controls=paths,baseline_tokens="exactly the nonblank source CSV sentences in order, independently decoded by judgment, unrestricted document context",cases=plan["ids"],gradient_steps=0)
    dump(HERE/"uniform-retention-gate.json",freeze)
    print(file_hash(HERE/"uniform-retention-gate.json"),freeze["combined_sha256"],paths,flush=True)
if __name__=="__main__":main()
