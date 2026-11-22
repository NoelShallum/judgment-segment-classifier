"""Apply a written emission calibration selectively to uncertain sentence decisions."""
import json
from itertools import product

import numpy as np
from safetensors.numpy import load_file,save_file

from sentence_emission_calibration import (
    DONOR,EDGE,ONE,OUT,ROOT,ZERO,HierModel,KnownSentences,I,Z,aligned_embeddings,
    document,file_hash,gold_cases,independent_validation,load_decoder,
    prepare_score,row_summary,sample_whole,save_json,torch,transition_scores
)
from sentence_role_retention import measure,paired
from hier_full import CACHE,cm

LIMITS=(.35,.75,1.25,2.,4.)


def calibrate(raw,weight,bias,mode,limit):
    if mode=="identity":
        return raw
    edit=raw@weight+bias-raw
    if mode=="full":
        return raw+edit
    if mode=="uncertain":
        highest=np.partition(raw,-2,axis=1)[:,-2:]
        gap=highest[:,1]-highest[:,0]
    else:
        gap=raw.max(1)-raw[:,5]
    focused=np.clip(1.-gap/limit,0.,1.)
    if mode=="reasoning-only":
        edit[:,[0,1,2,3,4,6]]=0
        edit[:,5]=np.maximum(edit[:,5],0.)
    return raw+focused[:,None]*edit


def score(bank,weights,bias,definition,per_case=False):
    x=bank.logits
    try:
        bank.logits=np.ascontiguousarray(calibrate(x,weights,bias,*definition),dtype=np.float64)
        y=bank.decode(ZERO,ONE)
    finally:
        bank.logits=x
    if per_case:
        return np.stack([cm(bank.gold[a:z],y[a:z]) for a,z in zip(bank.sizes,bank.sizes[1:])])
    return measure(cm(bank.gold,y))


def main():
    torch.set_num_threads(2)
    p=load_file(str(OUT/"known-sentence-role-pair.safetensors"))
    edited=load_file(str(OUT/"known-sentence-emission-calibration-retention.safetensors"))
    w,b=edited["emission.weight"],edited["emission.bias"]
    model=HierModel(DONOR)
    vocab=json.loads((EDGE/"vocab.json").read_text())
    engine=load_decoder()
    cohorts=sample_whole()

    def prepare(split):
        if split=="test":
            docs=[]; raw=[]
            for k,rr in gold_cases().items():
                docs.append(document(k,rr))
                all_lines=[line for _,line,_ in rr if line.strip()]
                if len(all_lines)!=len(rr):
                    raw.append(model.logits(model.contextualize(model.words(all_lines))))
                else:
                    with np.load(CACHE/(k+".npz")) as a:raw.append(a["context"])
        else:
            docs=cohorts[split]
            raw=[model.logits(x) for x in aligned_embeddings(model,docs,split)]
        ds=[{**d,"units":d["reference_units"]} for d in docs]
        cells=[(prepare_score(d,v,{"exact":p}),transition_scores(d["text"],d["units"],vocab,p)) for d,v in zip(ds,raw)]
        return KnownSentences(cells,p,engine).bind(ds)

    choices=[("identity",0.),("full",0.),*product(("uncertain","reasoning-nearby","reasoning-only"),LIMITS)]
    val=prepare("select")
    values={k:score(val,w,b,k) for k in choices}
    parent=values[("identity",0.)]
    pool=[k for k in choices if values[k][1]>=parent[1]-.005 and values[k][2]>=parent[2]-.005]
    winner=max(pool,key=lambda k:(values[k][0],values[k][1],k==("identity",0.)))
    file=ROOT/"results/sentence-calibration-gate-frozen.json"
    save_json(file,{"choice":winner,"values":[[*k,*v.tolist()] for k,v in values.items()],
                    "selection_ids":[d["id"] for d in cohorts["select"]]})
    print("Selected gate",winner,values[winner].tolist(),"preceding",parent.tolist(),flush=True)
    no_patch=winner==("identity",0.)
    if not no_patch:
        save_file({**edited,"emission.gate_code":np.asarray(("identity","full","uncertain","reasoning-nearby","reasoning-only").index(winner[0]),np.int16),
                   "emission.gate_limit":np.asarray(winner[1],np.float32)},str(OUT/"known-sentence-gated-calibration.safetensors"))
    visited={d["id"] for x in cohorts.values() for d in x}
    excluded=set(json.loads((EDGE/"overlap.json").read_text())["independent_exclusions_for_text_disjoint_check"])
    if not no_patch:
        cohorts["independent"]=[d for d in independent_validation(visited) if d["id"] not in excluded]
    report={"choice":winner,"freeze_sha256":file_hash(file),"splits":{},
             "protocol":"Fixed calibration can be gated by the top-two score margin, the reasoning score gap, or limited to its positive reasoning contribution. A finite selection grid requires sentence accuracy and reasoning F1 within 0.5 points of the preceding profile on validation. No gradients."}
    for k in ("select","confirm","independent","test") if not no_patch else ("select",):
        x=val if k=="select" else prepare(k)
        c=score(x,w,b,winner,True)
        base=score(x,w,b,("identity",0.),True)
        report["splits"][k]={"cases":len(c),"candidate":row_summary(c.sum(0)),"preceding":row_summary(base.sum(0)),
                              "paired_to_preceding":paired(c,base)}
        save_json(ROOT/"results/sentence-calibration-gate.json",report)
        print(k,measure(c.sum(0)).tolist(),measure(base.sum(0)).tolist(),flush=True)


if __name__=="__main__":
    main()
