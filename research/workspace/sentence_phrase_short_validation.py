"""Verify four-sentence transfer against the matched dense readout on a new cohort."""
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json

import numpy as np

import sentence_phrase_readout as p
from sentence_document_graph_write import OUT,cm,document,file_hash,load_file,paired,row_summary,save_json,torch

TAG="sentence-phrase-short-validation"
WIDTH=4
PER_QUARTILE=12


def shard(docs):
    torch.set_num_threads(1)
    context=p.load_predictor("contextual phrases")
    model,cfg,vocab,lex,vectors,weight,head,opt=context
    plan=json.loads((p.RESULT/(p.TAG+"-frozen.json")).read_text())
    base_spec=plan["arms"]["dense control"]
    base_path=OUT/(p.TAG+"-0.safetensors")
    assert file_hash(base_path)==base_spec["weights_sha256"]
    baseline=load_file(str(base_path))
    plain=base_spec["choice"]
    result=[]
    for src in docs:
        ref=src["reference_units"]
        text=[src["text"][a:b].strip() for a,b in ref]
        roles=src["reference_roles"]
        w=model.words(text)
        z=p.sparse(vectors,[p.canonical(t) for t in text])@weight
        counters=np.zeros((3,7,7),np.int64)
        for a in range(0,len(ref),WIDTH):
            b=min(len(ref),a+WIDTH)
            d=document(src["id"],[(i,text[i],roles[i]) for i in range(a,b)])
            hidden=model.contextualize(w[a:b])
            k,y,original,anch,x=p.dense.dense_features(d,hidden,model,cfg,vocab,lex)
            bank=([k],[0,len(y)],y,original,anch,x)
            edits=[original]
            for name,state,args in (("dense control",baseline,plain),("contextual phrases",head,opt)):
                estimate=p.dense.output(p.design(bank,z[a:b],name,args[0]),state)
                edits.append(p.predict_rows(bank,estimate,args) if args[4] else original)
            for j,guess in enumerate(edits):
                counters[j]+=cm(y,guess)
        result.append(counters)
    return np.stack(result)


def main():
    docs=p.docs("phrase-prospect")
    groups=np.array_split(np.asarray(sorted(range(len(docs)),key=lambda j:len(docs[j]["reference_units"]))),4)
    selected=[docs[int(j)] for q in groups for j in sorted(q,key=lambda i:
                         sha256((TAG+"|"+docs[i]["id"]).encode()).digest())[:PER_QUARTILE]]
    freeze=p.RESULT/(p.TAG+"-frozen.json")
    cohort=p.RESULT/"sentence-dense-readout-phrase-prospect.json"
    path=p.RESULT/(TAG+"-frozen.json")
    receipt={"model_freeze_sha256":file_hash(freeze),"parent_cohort_sha256":file_hash(cohort),
             "cases":[d["id"] for d in selected],"max_sentences":WIDTH,
             "sampling":"Twelve document IDs per judgment-length quartile, deterministically ranked by hash."}
    if path.is_file():
        assert json.loads(path.read_text())==receipt
    else:
        save_json(path,receipt)
    with ProcessPoolExecutor(4) as pool:
        scores=np.concatenate(list(pool.map(shard,[selected[i::4] for i in range(4)])))
    names=("released","dense control","contextual phrases")
    metrics={name:row_summary(scores[:,i].sum(0)) for i,name in enumerate(names)}
    r={"freeze_sha256":file_hash(path),"cases":len(selected),"metrics":metrics,
       "paired_phrases_to_released":paired(scores[:,2],scores[:,0]),
       "paired_phrases_to_dense":paired(scores[:,2],scores[:,1])}
    save_json(p.RESULT/(TAG+".json"),r)
    print("Four-sentence check",len(selected),[(k,v["rows"],v["macro_f1"],v["accuracy"]) for k,v in metrics.items()],
          r["paired_phrases_to_dense"]["macro_f1"],flush=True)


if __name__=="__main__":
    main()
