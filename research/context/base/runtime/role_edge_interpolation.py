"""Select physically interpolated whole-judgment decoder weights on validation."""
import json
import sys
from itertools import product

import numpy as np
from safetensors.numpy import save_file

from seven_paths import ROOT

EDGE = ROOT / "astra-role-edge"
sys.path.insert(0, str(EDGE))
from run_role_edge import (CRF, DONOR, HierModel, bootstrap, decode, document,
                           file_hash, full_score, gold_cases, independent_validation,
                           load_file, metric, prepare, sample_whole, save_json, torch)

FIELDS = (CRF, "cue_head", "position_head")
GRID = tuple(product((0., .25, .5, .75, 1.), (.1, .15, .2, .25, .3, .35)))
INCUMBENT = {"exact": (1., .2), "balanced": (0., .2)}


def materialize(parents, share, gain, memory=None):
    tensors = {k: parents["balanced"][k] + share * (parents["exact"][k] - parents["balanced"][k])
               for k in FIELDS}
    if memory is not None:
        tensors.update({"edge.log_ratio":memory["edge.log_ratio"],
                        "edge.gain":np.asarray(gain,np.float32),
                        "edge.mode":np.asarray(0,np.int16)})
    return tensors


def score(bank, parents, share, gain, section=None):
    tr = section[CRF] if section else materialize(parents,share,gain)[CRF]
    return np.stack([item.score(decode(item.logits["balanced"] + share *
                                     (item.logits["exact"] - item.logits["balanced"]),
                                     tr,phi,gain)) for item,phi in bank])


def main():
    torch.set_num_threads(2)
    parent_meta=json.loads((EDGE/"deployment.json").read_text())
    parents={}
    for k,v in parent_meta["profiles"].items():
        path=EDGE/v["file"]
        assert file_hash(path)==v["sha256"]
        parents[k]=load_file(str(path))
    vocab=json.loads((EDGE/"vocab.json").read_text())
    current=parents["exact"]
    assert np.array_equal(current["edge.log_ratio"],parents["balanced"]["edge.log_ratio"])
    model=HierModel(DONOR)
    docs=sample_whole()
    bank=prepare(docs["select"],"select",model,parents,vocab,current)
    values={spec:metric(score(bank,parents,*spec).sum(0)) for spec in GRID}
    selected={}
    for k,spec in INCUMBENT.items():
        control=values[spec]
        eligible=[s for s in GRID if k=="exact" or values[s][1]>=control[1]]
        selected[k]=max(eligible,key=lambda s:(values[s],s==spec))
    print("Selected",[(k,v,values[v],values[INCUMBENT[k]]) for k,v in selected.items()],flush=True)
    hashes={}
    for k,s in selected.items():
        path=ROOT/"checkpoints/hierarchical"/f"whole-role-edge-interpolated-{k}.safetensors"
        save_file(materialize(parents,*s,current),str(path))
        hashes[k]=file_hash(path)
    decision=ROOT/"results/role-edge-interpolation-frozen.json"
    save_json(decision,{"selection_case_ids":[x["id"] for x in docs["select"]],
                        "parents":parent_meta["profiles"],"selected":selected,
                        "control":INCUMBENT,"weight_hashes":hashes,
                        "grid":[[a,b,*s] for (a,b),s in values.items()]})
    used={d["id"] for block in docs.values() for d in block}
    excluded=set(json.loads((EDGE/"overlap.json").read_text())["independent_exclusions_for_text_disjoint_check"])
    docs["independent"]=[d for d in independent_validation(used) if d["id"] not in excluded]
    out={"method":"Bilinear selection over convex mixtures of exact and balanced emission and core transition weights, and a shared scale for the existing lexical edge tensor. Written as fixed weights. Balanced requires matching or exceeding the current balanced token accuracy on validation.","freeze_sha256":file_hash(decision),"splits":{}}
    for split in ("select","confirm","independent","test"):
        if split=="test":
            docs[split]=[document(k,r) for k,r in gold_cases().items()]
        if split!="select":
            bank=prepare(docs[split],split,model,parents,vocab,current)
        out["splits"][split]={}
        for k,s in selected.items():
            candidate=score(bank,parents,*s)
            base=score(bank,parents,*INCUMBENT[k])
            out["splits"][split][k]={"cases":len(bank),"selected":s,"candidate":full_score(candidate),
                                     "parent":full_score(base),"gap_ci95":bootstrap(candidate,base)}
            print(split,k,metric(candidate.sum(0)),metric(base.sum(0)),flush=True)
        save_json(ROOT/"results/role-edge-interpolation.json",out)


if __name__=="__main__":
    main()
