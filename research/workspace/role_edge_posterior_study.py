"""Evaluate an exact-span posterior decoder after frozen whole-case weight writes."""
import json
import sys
from itertools import product

import numpy as np
from safetensors.numpy import save_file

from role_edge_interpolation import (EDGE, INCUMBENT, CRF, DONOR, HierModel, ROOT,
    bootstrap, decode, document, file_hash, full_score, gold_cases, independent_validation,
    load_file, materialize, metric, prepare, sample_whole, save_json, score, torch)
from span_posterior_decode import posterior, spans, evaluate_spans, best_path

# Each proposal tiles the case and uses one of the validated decoder families.
PROPOSALS=((0.,.1),(0.,.2),(0.,.35),(.5,.15),(.5,.25),(1.,.1),(1.,.2),(1.,.35))
SETTINGS=tuple(product((.6,1.,1.6),(.05,.15,.25,.35),(0.,.025,.08)))


def compile_bank(bank,parents):
    compiled=[]
    for item,phi in bank:
        proposals=[]
        for share,gain in PROPOSALS:
            tr=materialize(parents,share,gain)[CRF]
            logit=item.logits["balanced"]+share*(item.logits["exact"]-item.logits["balanced"])
            proposals.append(spans(decode(logit,tr,phi,gain)))
        candidates=np.unique(np.concatenate(proposals),axis=0)
        candidates=candidates[np.argsort(candidates[:,1],kind="stable")]
        expected={}
        for name in INCUMBENT:
            expected[name]={}
            for temperature in (.6,1.,1.6):
                probs=posterior(item.logits[name],parents[name][CRF],.2*phi,temperature)
                expected[name][temperature]=evaluate_spans(probs,candidates,item.token_role_counts.sum(1))
        compiled.append((item,candidates,expected))
    return compiled


def measured(bank,name,t,cost,weight):
    result=[]
    for item,candidates,expected in bank:
        marginal,token=expected[name][t]
        path=best_path(candidates,marginal-cost+weight*token,len(item.logits[name]))
        result.append(item.score(path))
    return np.stack(result)


def main():
    torch.set_num_threads(1)
    manifest=json.loads((EDGE/"deployment.json").read_text())
    parents={k:load_file(str(EDGE/v["file"])) for k,v in manifest["profiles"].items()}
    vocab=json.loads((EDGE/"vocab.json").read_text())
    model=HierModel(DONOR)
    groups=sample_whole()
    selection=prepare(groups["select"],"select",model,parents,vocab,parents["exact"])
    candidates=compile_bank(selection,parents)
    grid={};selected={}
    for k,s in INCUMBENT.items():
        control=metric(score(selection,parents,*s).sum(0))
        best=control
        grid[k]=[]
        selected[k]=None
        for spec in SETTINGS:
            q=metric(measured(candidates,k,*spec).sum(0))
            grid[k].append([*spec,*q])
            if (k=="exact" or q[1]>=control[1]) and q>best:
                selected[k]=spec
                best=q
        print(k,"selected",selected[k],best,"parent",control,flush=True)
    freeze=ROOT/"results/role-edge-posterior-frozen.json"
    save_json(freeze,{"selected":selected,"grid":grid,"proposals":PROPOSALS,
                       "selection_ids":[d["id"] for d in groups["select"]]})
    saved={}
    for k,s in selected.items():
        if s is None:continue
        f=ROOT/"checkpoints/hierarchical"/f"whole-role-edge-posterior-{k}.safetensors"
        save_file({"decoder.risk":np.asarray(s,dtype=np.float64),
                   "decoder.proposals":np.asarray(PROPOSALS,dtype=np.float64)},str(f))
        saved[k]=file_hash(f)
    report={"method":"Maximal same-role interval posterior probabilities from a fixed role-edge CRF. Decode over the union of eight fixed-weight proposal paths using posterior expected hits minus a span count cost and optional token reward. This is a proxy for expected corpus F1. Temperature, cost and reward are frozen on development. No gradient steps.",
            "freeze_sha256":file_hash(freeze),"weights":saved,"selected":selected,"splits":{}}
    used={d["id"] for v in groups.values() for d in v}
    omitted=set(json.loads((EDGE/"overlap.json").read_text())["independent_exclusions_for_text_disjoint_check"])
    groups["independent"]=[d for d in independent_validation(used) if d["id"] not in omitted]
    for split in ("select","confirm","independent","test"):
        if split=="test":
            groups[split]=[document(k,v) for k,v in gold_cases().items()]
        raw=selection if split=="select" else prepare(groups[split],split,model,parents,vocab,parents["exact"])
        bank=candidates if split=="select" else compile_bank(raw,parents) if any(selected.values()) else None
        result={}
        for name,inc in INCUMBENT.items():
            control=score(raw,parents,*inc)
            patch=measured(bank,name,*selected[name]) if selected[name] else control
            result[name]={"cases":len(raw),"candidate":full_score(patch),"parent":full_score(control),
                          "gap_ci95":bootstrap(patch,control)}
            print(split,name,metric(patch.sum(0)),metric(control.sum(0)),flush=True)
        report["splits"][split]=result
        save_json(ROOT/"results/role-edge-posterior.json",report)


if __name__=="__main__":
    main()
