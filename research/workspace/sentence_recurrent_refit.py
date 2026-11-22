"""Refit identical closed-form heads to original and gate-edited representations."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np
from safetensors.numpy import save_file

import sentence_recurrent_inverse as edit
import sentence_dense_readout_write as dense
from sentence_document_graph_write import DONOR,EDGE,OUT,HierModel,file_hash,load_file,paired,row_summary,save_json,torch

TAG="sentence-recurrent-refit"


def plan_sources():
    plan=edit.read(edit.TAG+"-frozen")
    gate=max(edit.ARMS[1:],key=lambda k:(plan["arms"][k]["within_accuracy_floor"],plan["arms"][k]["select_macro_f1"]))
    path=OUT/(edit.TAG+"-"+str(edit.ARMS.index(gate))+".safetensors")
    assert file_hash(path)==plan["arms"][gate]["sha256"]
    selected=dense.read(dense.TAG+"-frozen")["arms"]["multiscale"]["selection"]
    assert selected[3]=="posterior"
    return gate,load_file(str(path)),selected


def gather(task):
    docs,words,patch=task
    torch.set_num_threads(1)
    model=HierModel(DONOR)
    edit.install(model,patch)
    cfg=load_file(str(dense.PARENT))
    vocab=json.loads((EDGE/"vocab.json").read_text())
    lexical=load_file(str(dense.LEXICAL))["lexical"][0]
    rows=[]
    for doc,w in zip(docs,words):
        hidden=model.contextualize(torch.from_numpy(w))
        rows.append(dense.dense_features(doc,hidden,model,cfg,vocab,lexical))
    return rows


def extract(split,patch):
    docs,w=edit.prepare(split)
    with ProcessPoolExecutor(4) as pool:
        pieces=list(pool.map(gather,[(docs[i::4],w[i::4],patch) for i in range(4)]))
    by_key={r[0]:r for q in pieces for r in q}
    rows=[by_key[d["id"]] for d in docs]
    return np.array([r[0] for r in rows]),np.r_[0,np.cumsum([len(r[1]) for r in rows])],*[np.concatenate([r[i] for r in rows]) for i in range(1,5)]


def fit(bank,settings):
    rarity,focus,damping,*_=settings
    _,_,labels,_,anchor,x=bank
    center=x.mean(0,dtype=np.float64).astype(np.float32)
    scale=(1./np.maximum(x.std(0,dtype=np.float64),.025)).astype(np.float32)
    z=np.column_stack(((x-center)*scale,np.ones(len(labels),np.float32)))
    difficulty=(1-anchor[np.arange(len(labels)),labels])**2
    freq=np.maximum(np.bincount(labels,minlength=7)/len(labels),.013)
    norm=(freq[labels]**-rarity*(1+focus*difficulty)).astype(np.float32)
    gram,links=dense.fit_moments(z,labels,anchor,norm)
    reg=np.r_[np.ones(len(gram)-1),.02]*gram.trace()/len(gram)
    beta=np.linalg.solve(gram+np.diag(damping*reg),links).astype(np.float32)
    return {"center":center,"scale":scale,"weight":beta[:-1].copy(),"bias":beta[-1].copy()}


def original_fit(keys):
    with np.load(dense.CACHE/"fit.npz",allow_pickle=False) as raw:
        locations={k:(a,b) for k,a,b in zip(raw["cases"],raw["offsets"],raw["offsets"][1:])}
        route=np.concatenate([np.arange(*locations[i]) for i in keys])
        return keys,None,*(raw[k][route] for k in ("gold","current","anchor","features"))


def main(splits):
    gate,patch,settings=plan_sources()
    gate_fit=extract("fit",patch)
    plain_fit=original_fit(gate_fit[0])
    np.testing.assert_array_equal(plain_fit[2],gate_fit[2])
    parents=(("original state",plain_fit),("edited state",gate_fit))
    models={arm:fit(data,settings) for arm,data in parents}
    folder=edit.RESULT
    hashes={}
    for index,(arm,w) in enumerate(models.items()):
        dest=OUT/(TAG+f"-{index}.safetensors")
        save_file(w,str(dest))
        hashes[arm]=file_hash(dest)
        models[arm]=load_file(str(dest))
    rec=folder/(TAG+"-frozen.json")
    save_json(rec,{"gate":gate,"selection":settings,"fit_cases":gate_fit[0].tolist(),"fit_rows":len(gate_fit[2]),
        "source_gate_plan_sha256":file_hash(folder/(edit.TAG+"-frozen.json")),"weights_sha256":hashes})
    result={"freeze_sha256":file_hash(rec),"splits":{}}
    del parents,plain_fit,gate_fit
    for split in splits:
        states=dense.prepare(split),extract(split,patch)
        for i in range(3):
            np.testing.assert_array_equal(states[0][i],states[1][i])
        mat={}
        for (arm,w),bank in zip(models.items(),states):
            pred=dense.replace(bank[3],bank[4],dense.output(bank[-1],w),*settings[3:])
            mat[arm]=dense.paired_confusion(bank,pred)
        base=mat["original state"]
        result["splits"][split]={"cases":len(base),"paired_gate_vs_original":paired(mat["edited state"],base),
            **{k:row_summary(v.sum(0)) for k,v in mat.items()}}
        save_json(folder/(TAG+".json"),result)
        print(split,result["splits"][split]["original state"]["macro_f1"],result["splits"][split]["edited state"]["macro_f1"],flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("splits",nargs="+")
    main(ap.parse_args().splits)
