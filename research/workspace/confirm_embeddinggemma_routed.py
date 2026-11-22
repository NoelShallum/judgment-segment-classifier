"""Confirm frozen EmbeddingGemma repair weights on text-disjoint validation judgments."""
import json
import numpy as np

from embeddinggemma_routed_write import (
    EDGE,FREEZE,DONOR,OUT,PARENT,ROOT,HierModel,aligned_embeddings,case_inputs,design,embeddings_for,
    file_hash,independent_validation,load_file,predict,rows_for_evaluation,sample_whole,save_json,
    summarize,torch,
)
from sentence_uncertainty_runtime import probability
from verify_embeddinggemma_routed import replay


def main():
    torch.set_num_threads(1)
    frozen=json.loads(FREEZE.read_text())
    assert frozen["parent_sha256"]==file_hash(PARENT)
    tried=sample_whole()
    used={d["id"] for g in tried.values() for d in g}
    excluded=set(json.loads((EDGE/"overlap.json").read_text())["independent_exclusions_for_text_disjoint_check"])
    docs=[d for d in independent_validation(used) if d["id"] not in excluded]
    assert {d["id"] for d in docs}.isdisjoint(set(frozen["train_cases"])|set(frozen["selection_cases"]))
    model=HierModel(DONOR)
    parent=load_file(str(PARENT))
    vocab=json.loads((EDGE/"vocab.json").read_text())
    cases=[case_inputs(d,h,model,parent,vocab) for d,h in zip(docs,aligned_embeddings(model,docs,"independent"))]
    routes=rows_for_evaluation(cases)
    print("Independent routes",len(cases),len(routes),flush=True)
    encoded=embeddings_for(cases,routes,"independent")
    old=json.loads((ROOT/"results/sentence-joint-edit.json").read_text())["splits"]["independent"]["candidate"]
    scores,paired=replay(cases,routes,encoded,frozen)
    variants={}
    for arm in frozen["choices"]:
        path=OUT/f"embeddinggemma-routed-{arm}.safetensors"
        w=load_file(str(path))
        dim,_,mix,margin=w["settings"]
        read=probability(design(cases,routes,encoded,arm,int(dim)),w)
        variants[arm]=value=summarize(cases,routes,predict(cases,routes,read,mix,margin))
        assert value["full_parent"]==old
        assert all(value[k]==scores[arm][k] for k in ("full","routed_scores","changed"))
        print(arm,value["routed_scores"]["macro_f1"],value["full"]["macro_f1"],value["routed_ci"]["macro_f1"],value["changed"],flush=True)
    save_json(ROOT/"results/embeddinggemma-routed-independent.json",{
        "freeze_sha256":file_hash(FREEZE),"case_ids":[d["id"] for d in docs],"results":variants,
        "paired_to_hierarchy_routed":paired})


if __name__=="__main__":
    main()
