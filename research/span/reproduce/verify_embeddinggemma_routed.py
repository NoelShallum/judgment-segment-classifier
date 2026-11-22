"""Recompute validation predictions from the frozen EmbeddingGemma repair tensors."""
import json
import numpy as np

from embeddinggemma_routed_write import (
    ARTIFACT,EDGE,FREEZE,DONOR,OUT,PARENT,RESULT,ROOT,HierModel,aligned_embeddings,case_inputs,cm,
    design,embeddings_for,file_hash,load_file,paired,row_summary,rows_for_evaluation,sample_whole,save_json,torch,
)


def replay(cases,routed,vec,frozen):
    reference=[c.current for c in cases]
    j,i=np.asarray(routed).T
    labels={};scores={};counts={}
    for arm,setting in frozen["choices"].items():
        path=OUT/f"embeddinggemma-routed-{arm}.safetensors"
        assert file_hash(path)==frozen["weights_sha256"][arm]
        w=load_file(str(path))
        np.testing.assert_array_equal(w["settings"],setting[1:])
        dim,_,mix,margin=w["settings"]
        x=design(cases,routed,vec,arm,int(dim))
        raw=np.clip(((x-w["center"])*w["scale"])@w["weight"]+w["bias"],0,None)
        p=raw/np.maximum(raw.sum(1,keepdims=True),1.e-9)
        value=(1-mix)*np.stack([cases[a].prior[b] for a,b in routed])+mix*p
        parent=np.fromiter((reference[a][b] for a,b in routed),dtype=np.int64)
        pick=value.argmax(1)
        k=np.arange(len(routed))
        choose=np.where(value[k,pick]-value[k,parent]>=margin,pick,parent)
        full=[y.copy() for y in reference]
        for a,b,y in zip(j,i,choose):
            full[a][b]=y
        labels[arm]=np.stack([cm(c.gold,y) for c,y in zip(cases,full)])
        gold=np.asarray([cases[a].gold[b] for a,b in routed])
        counts[arm]=np.stack([cm(gold[j==a],choose[j==a]) for a in range(len(cases))])
        scores[arm]={"full":row_summary(labels[arm].sum(0)),"routed_scores":row_summary(counts[arm].sum(0)),
                     "changed":int((choose!=parent).sum())}
    return scores, {arm:paired(counts[arm],counts["hierarchy"]) for arm in ("gemma","combined")}


def main():
    torch.set_num_threads(1)
    frozen=json.loads(FREEZE.read_text())
    observed=json.loads(RESULT.read_text())
    assert observed["freeze_sha256"]==file_hash(FREEZE)
    assert frozen["parent_sha256"]==file_hash(PARENT)
    assert frozen["encoder_provenance"]==json.loads((ARTIFACT/"provenance.json").read_text())
    for item in frozen["encoder_provenance"]["files"]:
        assert file_hash(ARTIFACT/item["file"])==item["sha256"]
    p=load_file(str(PARENT))
    vocab=json.loads((EDGE/"vocab.json").read_text())
    model=HierModel(DONOR)
    groups=sample_whole()
    archived=json.loads((ROOT/"results/sentence-joint-edit.json").read_text())["splits"]
    proof={"freeze_sha256":file_hash(FREEZE),"splits":{}}
    for key in ("select","confirm"):
        docs=groups[key]
        d=[case_inputs(q,h,model,p,vocab) for q,h in zip(docs,aligned_embeddings(model,docs,key))]
        rows=rows_for_evaluation(d)
        v=embeddings_for(d,rows,key)
        a,relative=replay(d,rows,v,frozen)
        for arm,q in a.items():
            old=observed["splits"][key][arm]
            assert len(rows)==old["routed"] and old["full_parent"]==archived[key]["candidate"]
            assert all(q[k]==old[k] for k in ("full","routed_scores","changed")),(key,arm)
        proof["splits"][key]={"cases":len(d),"routed":len(rows),"replay":a,"paired_to_hierarchy_routed":relative}
        print("Verified",key,len(d),len(rows),{k:r["macro_f1"] for k,r in relative.items()},flush=True)
    save_json(ROOT/"results/embeddinggemma-routed-qa.json",proof)


if __name__=="__main__":
    main()
