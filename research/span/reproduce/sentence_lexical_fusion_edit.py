"""Combine two frozen CPU models with a closed-form sentence score edit."""
import json

import fasttext
import numpy as np
from safetensors.numpy import load_file,save_file

from known_sentence_pair_edit import (
    EDGE,CRF,DONOR,ROOT,OUT,HierModel,KnownSentences,ONE,ZERO,aligned_embeddings,
    cm,document,edit_documents,embeddings,file_hash,gold_cases,independent_validation,
    load_decoder,prepare_score,row_summary,sample_whole,save_json,torch,transition_scores
)
from hier_full import CACHE
from lexical_trial import scores
from predict_role_edge import decode_profile
from sentence_role_retention import paired,measure


def projections(b,x,y):
    n=x.shape[1]
    z=np.column_stack((x,np.ones(len(x))))
    freq=np.bincount(y,minlength=7)/len(y)
    prior=np.eye(7)[y]
    wrong=b.argmax(1)!=y
    r=b.argmax(1)
    ix=np.arange(len(y))
    margin=np.clip(b[ix,r]-b[ix,y]+.6,.5,2.8)
    delta=np.zeros_like(b)
    delta[ix[wrong],y[wrong]]=margin[wrong]*.5
    delta[ix[wrong],r[wrong]]=-margin[wrong]*.5
    for rarity in (.25,.60):
        w=freq[y]**-rarity
        g=z.T@(w[:,None]*z)
        reg=.18*np.trace(g)/len(g)
        regularizer=np.diag([1.]*n+[.04])
        for target,tag in ((delta,"residual"),(prior,"replacement")):
            p=np.linalg.solve(g+reg*regularizer,z.T@(w[:,None]*target))
            if tag=="replacement":
                raw=z@p
                width=lambda u:(u-u.mean(1,keepdims=True)).std()
                p*=width(b)/max(width(raw),.001)
            for amount in (.25,.5,.75,1.):
                yield f"{tag}-{rarity:g}-{amount:g}",(p[:-1]*amount,p[-1]*amount,
                       1. if tag=="residual" else 1.-amount)


def main():
    torch.set_num_threads(2)
    model=HierModel(DONOR)
    base=load_file(str(OUT/"known-sentence-role-pair.safetensors"))
    released=load_file(str(OUT/"known-sentence-label-change.safetensors"))
    vocabulary=json.loads((EDGE/"vocab.json").read_text())
    lexical_file=ROOT/"checkpoints/lexical/model.bin"
    l=fasttext.load_model(str(lexical_file))
    engine=load_decoder()

    def pack(docs,raw):
        copy=[{**d,"units":d["reference_units"]} for d in docs]
        items=[]
        lexical=[]
        for d,x in zip(copy,raw):
            text=d["text"]
            items.append((prepare_score(d,x,{"exact":base}),transition_scores(text,d["units"],vocabulary,base)))
            st=np.stack([scores(l,text[a:b]) for a,b in d["units"]])
            st=(st-st.mean(1,keepdims=True)).clip(-9,9)
            before=np.concatenate((st[:1],st[:-1]))
            after=np.concatenate((st[1:],st[-1:]))
            lexical.append(np.column_stack((st,.5*(before+after))))
        return KnownSentences(items,base,engine).bind(copy),np.concatenate(lexical)

    def control(bank):
        return np.concatenate([decode_profile(d.logits["exact"],e[1:],released,bank.scale)
               for d,e in zip(bank.items,np.split(bank.edge,bank.sizes[1:-1]))])

    def predict(bank,x,name,params):
        if name=="released":
            return control(bank)
        p,b,s=params
        old=bank.logits
        try:
            bank.logits=np.ascontiguousarray(s*old+x@p+b,dtype=np.float64)
            return bank.decode(ZERO,ONE)
        finally:
            bank.logits=old

    train=edit_documents()
    fit,lfit=pack(train,[model.logits(x) for x in embeddings(model,train,"aligned-source-train")])
    candidates={"released":None}
    for g in (.05,.12,.25,.5):
        a=np.zeros((21,7))
        a[7:14]=g*np.eye(7)
        candidates[f"fixed-lexical-{g:g}"]=(a,np.zeros(7),1.)
    candidates.update(projections(fit.logits,np.column_stack((fit.logits,lfit)),fit.gold))
    docs=sample_whole()
    def data(s):
        if s!="test":
            return pack(docs[s],[model.logits(x) for x in aligned_embeddings(model,docs[s],s)])
        documents=[];raw=[]
        for k,rr in gold_cases().items():
            documents.append(document(k,rr))
            pieces=[t for _,t,_ in rr if t.strip()]
            if len(pieces)!=len(rr):
                raw.append(model.logits(model.contextualize(model.words(pieces))))
            else:
                with np.load(CACHE/(k+".npz")) as f:raw.append(f["context"])
        return pack(documents,raw)

    val,ls=data("select")
    x=np.column_stack((val.logits,ls))
    option={name:measure(cm(val.gold,predict(val,x,name,c))).tolist() for name,c in candidates.items()}
    choice=max(option,key=lambda k:(option[k][0],option[k][1],k=="released"))
    print("Fusion choice",choice,option[choice],"baseline",option["released"],flush=True)
    path=OUT/"known-sentence-lexical-fusion.safetensors"
    if choice!="released":
        p,b,a=candidates[choice]
        save_file({**base,"fused.weight":p.astype("f4"),"fused.bias":b.astype("f4"),"fused.scale":np.asarray(a,np.float32)},str(path))
        written=load_file(str(path))
        candidates[choice]=(written["fused.weight"],written["fused.bias"],written["fused.scale"])
    freeze=ROOT/"results/sentence-lexical-fusion-frozen.json"
    save_json(freeze,{"choice":choice,"scores":option,"train_case_ids":[d["id"] for d in train],
                      "selection_cases":[d["id"] for d in docs["select"]],
                      "lexical_sha256":file_hash(lexical_file),"weight_sha256":file_hash(path) if choice!="released" else None})
    used={d["id"] for c in docs.values() for d in c}
    excluded=set(json.loads((EDGE/"overlap.json").read_text())["independent_exclusions_for_text_disjoint_check"])
    docs["independent"]=[d for d in independent_validation(used) if d["id"] not in excluded]
    result={"choice":choice,"freeze_sha256":file_hash(freeze),"splits":{},
        "protocol":"Reuse the already fitted fastText LegalSeg checkpoint as a frozen lexical scorer. Train no neural or lexical features in this experiment. Compare fixed blends and closed-form linear writes on hierarchical and lexical scores, including immediate neighbor lexical scores. All fits use 144 official training judgments, followed by frozen validation selection."}
    for split in ("select","confirm","independent","test") if choice!="released" else ("select","confirm","independent"):
        bank,lx=(val,ls) if split=="select" else data(split)
        x=np.column_stack((bank.logits,lx))
        a=predict(bank,x,choice,candidates[choice])
        b=control(bank)
        case=lambda y:np.stack([cm(bank.gold[i:j],y[i:j]) for i,j in zip(bank.sizes,bank.sizes[1:])])
        a,b=case(a),case(b)
        result["splits"][split]={"cases":len(a),"candidate":row_summary(a.sum(0)),"preceding":row_summary(b.sum(0)),
                                 "paired_to_preceding":paired(a,b)}
        save_json(ROOT/"results/sentence-lexical-fusion-edit.json",result)
        print(split,measure(a.sum(0)).tolist(),measure(b.sum(0)).tolist(),flush=True)


if __name__=="__main__":
    main()
