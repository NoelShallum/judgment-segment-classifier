"""Write a seven-by-seven calibration of frozen sentence evidence scores."""
import json

import numpy as np
from safetensors.numpy import load_file,save_file

from known_sentence_pair_edit import (
    CRF,DONOR,EDGE,ONE,OUT,ROOT,ZERO,HierModel,KnownSentences,aligned_embeddings,
    document,edit_documents,embeddings,file_hash,gold_cases,independent_validation,
    load_decoder,prepare_score,row_summary,sample_whole,save_json,torch,transition_scores
)
from hier_full import CACHE,cm
from sentence_role_retention import measure,paired
from seven_paths import LABELS


I=np.eye(7,dtype=np.float64)
Z=np.zeros(7,dtype=np.float64)
CENTER=I-np.ones_like(I)/7.


def transform(bank,a,b,per_case=False):
    raw=bank.logits
    try:
        bank.logits=np.ascontiguousarray(raw@a+b,dtype=np.float64)
        y=bank.decode(ZERO,ONE)
    finally:
        bank.logits=raw
    if per_case:
        return np.stack([cm(bank.gold[x:z],y[x:z]) for x,z in zip(bank.sizes,bank.sizes[1:])])
    return measure(cm(bank.gold,y))


def discriminant(x,y,prior_weight):
    mu=np.stack([x[y==k].mean(0) for k in range(7)])
    groups=[(x[y==k]-mu[k]) for k in range(7)]
    cov=sum(a.T@a/len(a) for a in groups)/7
    v=np.linalg.solve(cov+.35*np.trace(cov)/7.*I,mu.T)
    b=-.5*np.sum(mu*v.T,axis=1)
    center=np.log(np.bincount(y)/len(y))
    b+=prior_weight*(center-center.mean())
    d=x@v+b
    scl=(x@CENTER).std()/max((d@CENTER).std(),.001)
    return CENTER@v*scl,b*scl


def margin_correction(bank,signal,protected):
    old=bank.decode(ZERO,ONE)
    y=bank.gold
    i=np.arange(len(y))
    selected=np.flatnonzero(old!=y)
    target=np.zeros((len(y),7))
    margin=np.clip(bank.logits[i,old]-bank.logits[i,y]+.4,.25,2.)
    target[selected,y[selected]]=margin[selected]*.5
    target[selected,old[selected]]=-margin[selected]*.5
    prior=np.bincount(y)/len(y)
    weights=np.where(old==y,.75,1.) / prior[y]**.35
    weights[np.isin(y,[5,6])]*=protected
    head=np.column_stack((signal,np.ones(len(y))))
    normal=head.T@(head*weights[:,None])
    # Quantities other than model scores are used only to write the output matrix.
    reg=.25*np.trace(normal)/len(normal)
    matrix=np.linalg.solve(normal+reg*np.eye(len(normal)),head.T@(target*weights[:,None]))
    return CENTER@matrix[:-1],matrix[-1]


def main():
    torch.set_num_threads(2)
    base=load_file(str(OUT/"known-sentence-role-pair.safetensors"))
    model=HierModel(DONOR)
    engine=load_decoder()
    vocab=json.loads((EDGE/"vocab.json").read_text())

    def pack(docs,raw):
        ds=[{**d,"units":d["reference_units"]} for d in docs]
        rows=[(prepare_score(d,x,{"exact":base}),transition_scores(d["text"],d["units"],vocab,base)) for d,x in zip(ds,raw)]
        return KnownSentences(rows,base,engine).bind(ds)

    train=edit_documents()
    tr=pack(train,[model.logits(x) for x in embeddings(model,train,"aligned-source-train")])
    signals=tr.logits@CENTER
    options={"unchanged":(I,Z)}
    for alpha in (0.,.25,1.):
        a,b=discriminant(signals,tr.gold,alpha)
        for mix in (.15,.35,.65,1.):
            options[f"class-moments-{alpha:g}-{mix:g}"]=((1-mix)*CENTER+mix*a,mix*b)
    for alpha in (1.,2.,4.):
        a,b=margin_correction(tr,signals,alpha)
        for gain in (.25,.50,1.,1.5):
            options[f"margin-{alpha:g}-{gain:g}"]=(I+gain*a,gain*b)
    groups=sample_whole()
    def bank(split):
        docs=groups[split]
        return pack(docs,[model.logits(x) for x in aligned_embeddings(model,docs,split)])
    selection=bank("select")
    values={k:transform(selection,*m).tolist() for k,m in options.items()}
    current=np.asarray(values["unchanged"])
    safe=[k for k,v in values.items() if np.all(np.asarray(v)[2:]>=current[2:]-.005)]
    decisions={"macro":max(options,key=lambda k:(values[k][0],values[k][1],k=="unchanged")),
               "retention":max(safe,key=lambda k:(values[k][0],values[k][1],k=="unchanged"))}
    parent_meta={"sha256":file_hash(OUT/"known-sentence-role-pair.safetensors")}
    freeze=ROOT/"results/sentence-emission-calibration-frozen.json"
    save_json(freeze,{"selected":decisions,"grid":values,"benchmark_case_ids":[d["id"] for d in groups["select"]],
                      "train_cases":[d["id"] for d in train],"parent":parent_meta})
    replay={}
    for n,k in decisions.items():
        a,b=options[k]
        expected=OUT/f"known-sentence-emission-calibration-{n}.safetensors"
        save_file({**base,"emission.weight":a.astype("f4"),"emission.bias":b.astype("f4")},str(expected))
        p=load_file(str(expected))
        replay[n]=(p["emission.weight"],p["emission.bias"])
        print("Calibration choice",n,k,values[k],"baseline",current.tolist(),flush=True)
    used={d["id"] for values in groups.values() for d in values}
    excluded=set(json.loads((EDGE/"overlap.json").read_text())["independent_exclusions_for_text_disjoint_check"])
    groups["independent"]=[d for d in independent_validation(used) if d["id"] not in excluded]
    report={"selected":decisions,"freeze_sha256":file_hash(freeze),"splits":{},
            "protocol":"An affine 7-by-7 emission calibration from class moments or residual margin errors, fitted by a regularized linear solve. The current lexical and static transitions remain fixed. For the second selection, the two target roles may lose no more than 0.5 F1 points in development. No gradients."}
    for split in ("select","confirm","independent","test"):
        if split=="test":
            docs=[];text=[]
            for k,rr in gold_cases().items():
                docs.append(document(k,rr))
                pool=[line for _,line,_ in rr if line.strip()]
                if len(pool)==len(rr):
                    with np.load(CACHE/(k+".npz")) as f:text.append(f["context"])
                else:
                    text.append(model.logits(model.contextualize(model.words(pool))))
            data=pack(docs,text)
        else:
            data=selection if split=="select" else bank(split)
        actual=transform(data,I,Z,True)
        entry={}
        for k,proposal in replay.items():
            x=transform(data,*proposal,True)
            entry[k]={"cases":len(x),"candidate":row_summary(x.sum(0)),"preceding":row_summary(actual.sum(0)),
                      "paired_to_preceding":paired(x,actual)}
            print(split,k,measure(x.sum(0)).tolist(),measure(actual.sum(0)).tolist(),flush=True)
        report["splits"][split]=entry
        save_json(ROOT/"results/sentence-emission-calibration.json",report)


if __name__=="__main__":
    main()
