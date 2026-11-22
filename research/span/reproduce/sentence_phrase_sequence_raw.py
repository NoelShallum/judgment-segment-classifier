"""Transfer frozen phrase sequence weights to raw judgments and score character spans."""
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np
from safetensors.numpy import load_file

import sentence_phrase_sequence as seq
from sentence_document_graph_write import OUT,file_hash,save_json,torch
from span_eval import counts,summarize
from raw_emitter_write import paired

P=seq.phrase
TAG="sentence-phrase-sequence-raw"


def work(docs):
    torch.set_num_threads(1)
    model,cfg,vocab,memory,vectors,weights,*_=P.load_predictor("target phrases")
    engine=seq.load_decoder()
    freeze=P.RESULT/(seq.TAG+"-frozen.json")
    plan=json.loads(freeze.read_text())
    path=OUT/(seq.TAG+".safetensors")
    assert file_hash(path)==plan["checkpoint_sha256"]
    w=load_file(str(path))
    shapes=[]
    for doc in docs:
        offsets=doc["units"]
        text=[doc["text"][a:b].strip() for a,b in offsets]
        item=seq.raw_document(doc["text"],doc["id"])
        assert item["units"]==offsets
        state=model.contextualize(model.words(text))
        k,y,original,anchor,x=P.dense.dense_features(item,state,model,cfg,vocab,memory)
        index=np.array([0,len(text)],dtype=np.int64)
        z=P.sparse(vectors,[P.canonical(t) for t in text])@weights
        bank=([k],index,y,original,anchor,x)
        base,u=seq.source_unary(bank,z)
        predictions={"released decoder":original,"phrase baseline":base}
        for profile,(mode,strength,stay,guard) in plan["rules"].items():
            if mode=="none":
                predictions[profile]=base
            else:
                edge=seq.edge_arrays(bank,z,mode,w)
                predictions[profile]=seq.decode(engine,index,u,base,edge,strength,stay,guard)
        shapes.append((doc["id"],len(offsets),{k:counts(doc,offsets,v) for k,v in predictions.items()}))
    return shapes


def main():
    docs=P.docs("test")
    assert len(docs)==712
    s=P.ROOT/"references/official/test.csv"
    freeze=P.RESULT/(seq.TAG+"-frozen.json")
    with ProcessPoolExecutor(4) as pool:
        packs=list(pool.map(work,[docs[i::4] for i in range(4)]))
    rows={k:(n,v) for each in packs for k,n,v in each}
    order=[rows[d["id"]] for d in docs]
    matrices={name:np.stack([v[name] for _,v in order]) for name in order[0][1]}
    result={"source_sha256":file_hash(s),"sequence_freeze_sha256":file_hash(freeze),
           "cases":len(docs),"predicted_sentence_units":sum(n for n,_ in order),"profiles":{}}
    baseline=matrices["phrase baseline"]
    for name,values in matrices.items():
        result["profiles"][name]={"scores":summarize(values.sum(0)),
                                  "paired_to_phrase_baseline":paired(values,baseline)}
    fused={k:r["scores"]["exact_labeled_spans"]["f1"] for k,r in result["profiles"].items()}
    gold={r["scores"]["exact_labeled_spans"]["gold"] for r in result["profiles"].values()}
    assert len(gold)==1 and all(v.sum()>0 for v in matrices.values())
    save_json(P.RESULT/(TAG+".json"),result)
    print(len(docs),result["predicted_sentence_units"],fused,flush=True)


if __name__=="__main__":
    main()
