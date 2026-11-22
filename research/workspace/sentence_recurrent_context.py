"""Test the frozen recurrent weight write under different input context budgets."""
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json

import numpy as np

import sentence_recurrent_inverse as study
from sentence_document_graph_write import DONOR,EDGE,OUT,HierModel,document,file_hash,load_file,paired,row_summary,save_json,torch

TAG="sentence-recurrent-context"
WIDTHS=(1,4,16,64,None)
SAMPLE=24


def run(task):
    docs,words,patch=task
    torch.set_num_threads(1)
    model=HierModel(DONOR)
    cfg=load_file(str(study.dense.PARENT))
    vocab=json.loads((EDGE/"vocab.json").read_text())
    original=study.raw_parameters(model)
    result={}
    for width in WIDTHS:
        records=[]
        for d,x in zip(docs,words):
            local=width or len(x)
            pieces=[d["text"][a:b].strip() for a,b in d["reference_units"]]
            gold=d["reference_roles"]
            parts=[document(d["id"],[(j,pieces[j],gold[j]) for j in range(i,min(i+local,len(x)))])
                   for i in range(0,len(x),local)]
            slices=[x[i:i+local] for i in range(0,len(x),local)]
            out=[]
            for variant in (original,patch):
                study.install(model,variant)
                preds=study.predict_cases(model,parts,slices,cfg,vocab)
                out.append(study.statistics(parts,preds).sum(0))
            records.append(out)
        result[str(width)]=np.asarray(records)
    return result


def main():
    f=study.RESULT/(study.TAG+"-frozen.json")
    plan=study.read(study.TAG+"-frozen")
    gate=max(study.ARMS[1:],key=lambda a:(plan["arms"][a]["within_accuracy_floor"],plan["arms"][a]["select_macro_f1"]))
    target=OUT/(study.TAG+"-"+str(study.ARMS.index(gate))+".safetensors")
    assert file_hash(target)==plan["arms"][gate]["sha256"]
    weights=load_file(str(target))
    docs,words=study.prepare("holdout")
    n=min(SAMPLE,len(docs))
    chosen=sorted(range(len(docs)),key=lambda i:sha256((TAG+"-v1|"+docs[i]["id"]).encode()).digest())[:n]
    docs=[docs[i] for i in chosen]
    words=[words[i] for i in chosen]
    with ProcessPoolExecutor(3) as pool:
        parts=list(pool.map(run,[(docs[i::3],words[i::3],weights) for i in range(3)]))
    result={"weight_freeze_sha256":file_hash(f),"weight_sha256":file_hash(target),"arm":gate,
            "cases":[d["id"] for d in docs],"rows":sum(len(x) for x in words),"windows":{}}
    for name in map(str,WIDTHS):
        by_case=np.concatenate([q[name] for q in parts])
        parent,edited=by_case[:,0],by_case[:,1]
        result["windows"][name]={"parent":row_summary(parent.sum(0)),"edited":row_summary(edited.sum(0)),
                                 "paired_to_parent":paired(edited,parent)}
        p=result["windows"][name]
        assert p["parent"]["rows"]==p["edited"]["rows"]==result["rows"]
        print(name,p["parent"]["macro_f1"],p["edited"]["macro_f1"],flush=True)
    save_json(study.RESULT/(TAG+".json"),result)


if __name__=="__main__":
    main()
