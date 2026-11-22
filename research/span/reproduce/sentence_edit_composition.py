"""Compose independently written transition and emission edits before held-out replay."""
from itertools import product
import json

import numpy as np
from safetensors.numpy import load_file,save_file

from sentence_emission_calibration import (
    CRF,DONOR,EDGE,ONE,OUT,ROOT,ZERO,HierModel,KnownSentences,I,Z,aligned_embeddings,
    document,file_hash,gold_cases,independent_validation,load_decoder,
    prepare_score,row_summary,sample_whole,save_json,torch,transition_scores,transform
)
from sentence_role_retention import measure,paired
from hier_full import CACHE

WEIGHTS=(CRF,"edge.log_ratio","position_head")
MIX=list(product((0.,.5,1.),(0.,.25,.5,1.)))


def edit(base,structural,calibrator,gain,amount):
    result={**base}
    for k in WEIGHTS:
        result[k]=(base[k]+gain*(structural[k]-base[k])).astype(np.float32)
    result["emission.weight"]=(I+amount*(calibrator["emission.weight"]-I)).astype(np.float32)
    result["emission.bias"]=(amount*calibrator["emission.bias"]).astype(np.float32)
    return result


def main():
    torch.set_num_threads(2)
    current=load_file(str(OUT/"known-sentence-role-pair.safetensors"))
    prior=load_file(str(EDGE/"role-edge-balanced.safetensors"))
    structure=load_file(str(OUT/"known-sentence-role-retention-retention.safetensors"))
    calibration=load_file(str(OUT/"known-sentence-emission-calibration-retention.safetensors"))
    vocabulary=json.loads((EDGE/"vocab.json").read_text())
    model=HierModel(DONOR)
    engine=load_decoder()
    cohorts=sample_whole()

    def docs_with_raw(split):
        if split=="test":
            docs=[]; raws=[]
            for k,rr in gold_cases().items():
                docs.append(document(k,rr))
                text=[line for _,line,_ in rr if line.strip()]
                if len(text)!=len(rr):
                    raws.append(model.logits(model.contextualize(model.words(text))))
                else:
                    with np.load(CACHE/(k+".npz")) as f:raws.append(f["context"])
        else:
            docs=cohorts[split]
            raws=[model.logits(x) for x in aligned_embeddings(model,docs,split)]
        return [{**d,"units":d["reference_units"]} for d in docs],raws

    def bank(docs,raw,program):
        entries=[]
        for d,q in zip(docs,raw):
            values=prepare_score(d,q,{"exact":program})
            edge=transition_scores(d["text"],d["units"],vocabulary,program)
            entries.append((values,edge))
        return KnownSentences(entries,program,engine).bind(docs)

    def score(bank_,profile,per_case=False):
        return transform(bank_,profile.get("emission.weight",I),profile.get("emission.bias",Z),per_case)

    docs,raw=docs_with_raw("select")
    result={}
    for g in {x for x,_ in MIX}:
        t=edit(current,structure,calibration,g,0.)
        b=bank(docs,raw,t)
        for u in [a for x,a in MIX if x==g]:
            result[g,u]=score(b,edit(current,structure,calibration,g,u)).tolist()
    before=score(bank(docs,raw,prior),prior)
    allow=[k for k,v in result.items() if np.all(np.asarray(v)[2:]>=before[2:]-.005)]
    choice=max(allow or result.keys(),key=lambda e:tuple(result[e]))
    model_file=OUT/"known-sentence-composed.safetensors"
    save_file(edit(current,structure,calibration,*choice),str(model_file))
    freeze=ROOT/"results/sentence-edit-composition-frozen.json"
    save_json(freeze,{"choice":choice,"grid":[[*k,*v] for k,v in sorted(result.items())],
                      "selection_case_ids":[d["id"] for d in docs],"patch_sha256":file_hash(model_file),
                      "composite_role_floors":(before[2:]-.005).tolist(),
                      "used_role_floors":bool(allow)})
    print("Composite validation choice",choice,result[choice],"preceding",result[(0.,0.)],flush=True)
    picked=load_file(str(model_file))
    used={d["id"] for v in cohorts.values() for d in v}
    excluded=set(json.loads((EDGE/"overlap.json").read_text())["independent_exclusions_for_text_disjoint_check"])
    cohorts["independent"]=[d for d in independent_validation(used) if d["id"] not in excluded]
    summary={"choice":choice,"freeze_sha256":file_hash(freeze),"splits":{},
             "protocol":"Compose independently validation-frozen decoder and emission edits. Validation chooses mixture by macro-F1 subject to reasoning and decision F1 floors within 0.5 points of the original sentence-context parent. If no mixture meets the floors, choose the highest macro-F1 and record the constraint failure. No gradients."}
    for s in ("select","confirm","independent","test"):
        if s!="select":docs,raw=docs_with_raw(s)
        x=score(bank(docs,raw,picked),picked,True)
        before=score(bank(docs,raw,current),current,True)
        legacy=score(bank(docs,raw,prior),prior,True)
        summary["splits"][s]={"cases":len(x),"candidate":row_summary(x.sum(0)),
                              "preceding":row_summary(before.sum(0)),"original":row_summary(legacy.sum(0)),
                              "paired_to_preceding":paired(x,before),"paired_to_original":paired(x,legacy)}
        save_json(ROOT/"results/sentence-edit-composition.json",summary)
        print(s,measure(x.sum(0)).tolist(),measure(before.sum(0)).tolist(),flush=True)


if __name__=="__main__":
    main()
