"""Replay the frozen phrase readout at several supplied-sentence context budgets."""
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json

import numpy as np

import sentence_phrase_readout as phrase
import sentence_recurrent_inverse as recur
from sentence_document_graph_write import cm,document,file_hash,paired,row_summary,save_json,torch

TAG="sentence-phrase-context"
WIDTHS=(1,4,16,64,None)
ARM="contextual phrases"


def chunk(data):
    torch.set_num_threads(1)
    docs,word=data
    original,cfg,vocab,memory,vectors,weight,head,opt=phrase.load_predictor(ARM)
    result={}
    for w in WIDTHS:
        matrices=[]
        for doc,pre in zip(docs,word):
            step=w or len(pre)
            pieces=[doc["text"][a:b].strip() for a,b in doc["reference_units"]]
            labels=doc["reference_roles"]
            scores=phrase.sparse(vectors,[phrase.canonical(text) for text in pieces])@weight
            cases=np.zeros((2,7,7),np.int64)
            for a in range(0,len(pre),step):
                b=min(len(pre),a+step)
                item=document(doc["id"],[(i,pieces[i],labels[i]) for i in range(a,b)])
                st=original.contextualize(torch.from_numpy(pre[a:b]))
                k,l,base,anchor,x=phrase.dense.dense_features(item,st,original,cfg,vocab,memory)
                bank=([k],[0,len(base)],l,base,anchor,x)
                read=phrase.dense.output(phrase.design(bank,scores[a:b],ARM,opt[0]),head)
                pred=phrase.predict_rows(bank,read,opt) if opt[4] else base
                cases[0]+=cm(l,base)
                cases[1]+=cm(l,pred)
            matrices.append(cases)
        result[str(w)]=np.stack(matrices)
    return result


def main():
    docs,words=recur.prepare("holdout")
    order=sorted(range(len(docs)),key=lambda j:sha256((TAG+"-1|"+docs[j]["id"]).encode()).digest())[:16]
    cohort=[docs[i] for i in order]
    vectors=[words[i] for i in order]
    with ProcessPoolExecutor(4) as pool:
        all_scores=list(pool.map(chunk,[(cohort[i::4],vectors[i::4]) for i in range(4)]))
    freeze=phrase.RESULT/(phrase.TAG+"-frozen.json")
    r={"freeze_sha256":file_hash(freeze),"cases":[d["id"] for d in cohort],
       "rows":sum(len(v) for v in vectors),"profile":ARM,"windows":{}}
    for k in map(str,WIDTHS):
        a=np.concatenate([chunk[k] for chunk in all_scores])
        ref,arm=a[:,0],a[:,1]
        d=r["windows"][k]={"parent":row_summary(ref.sum(0)),"edited":row_summary(arm.sum(0)),
                           "paired":paired(arm,ref)}
        assert d["parent"]["rows"]==d["edited"]["rows"]==r["rows"]
        print(k,d["parent"]["macro_f1"],d["edited"]["macro_f1"],flush=True)
    save_json(phrase.RESULT/(TAG+".json"),r)


if __name__=="__main__":
    main()
