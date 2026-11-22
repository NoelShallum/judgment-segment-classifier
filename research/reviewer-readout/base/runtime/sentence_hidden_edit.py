"""Write sentence readouts from frozen recurrent states by closed-form ridge regression."""
import csv
import json
from itertools import groupby
import random

import numpy as np
from safetensors.numpy import load_file,save_file

from known_sentence_pair_edit import (
    EDGE,CRF,DONOR,ROOT,OUT,HierModel,KnownSentences,ONE,ZERO,aligned_embeddings,
    cm,document,edit_documents,embeddings,file_hash,gold_cases,independent_validation,
    load_decoder,prepare_score,row_summary,sample_whole,save_json,torch,transition_scores
)
from predict_role_edge import decode_profile
from program_trial import TO_ID
from sentence_role_retention import measure,paired


def training_documents():
    base=edit_documents()
    used={int(d["id"]) for d in base}
    extra=set(random.Random(948602).sample(sorted(set(range(4984))-used),240))
    with (ROOT/"references/official/train.csv").open(newline="",encoding="utf-8-sig") as file:
        for k,records in groupby(csv.DictReader(file),key=lambda r:r["Index"]):
            if int(k) in extra:
                rows=[(i,row["Text"],TO_ID[row["Label"]]) for i,row in enumerate(records)]
                base.append(document(k,rows))
    assert len(base)==384
    return base


def transform(x,cfg):
    x=(x-cfg["mean"])*cfg["scale"]
    if "projection" in cfg:
        return np.column_stack((x,np.maximum(x@cfg["projection"]+cfg["offset"],0.)))
    return x


def train_weights(x,y,scores,pred):
    old=np.bincount(y,minlength=7)/len(y)
    dims=x.shape[1]
    mean=x.mean(0).astype("f4")
    scale=(1./np.maximum(x.std(0),.035)).astype("f4")
    basic={"mean":mean,"scale":scale}
    rng=np.random.default_rng(76012)
    random_features={**basic,"projection":(rng.standard_normal((dims,320))/np.sqrt(dims)).astype("f4"),
                    "offset":rng.uniform(-1.,1.,320).astype("f4")}
    i=np.arange(len(y))
    wrong=pred!=y
    margin=np.clip(scores[i,pred]-scores[i,y]+.6,.35,2.8)
    correction=np.zeros_like(scores)
    correction[i[wrong],y[wrong]]=margin[wrong]*.5
    correction[i[wrong],pred[wrong]]=-margin[wrong]*.5
    freq=np.eye(7)[y]
    old_centered=scores-scores.mean(1,keepdims=True)
    for name,c in (("linear",basic),("random_features",random_features)):
        features=transform(x,c)
        z=np.column_stack((features,np.ones(len(features)))).astype(np.float64)
        for weighting in (.25,.5):
            w=old[y]**-weighting
            gram=z.T@(w[:,None]*z)
            diag=np.eye(len(gram))
            diag[-1,-1]=.02
            for ridge in (.04,.18):
                g=gram+diag*ridge*gram.trace()/len(gram)
                residual=np.linalg.solve(g,z.T@(w[:,None]*correction))
                probe=np.linalg.solve(g,z.T@(w[:,None]*freq))
                h=z@probe
                t=old_centered.std()/max((h-h.mean(1,keepdims=True)).std(),.01)
                for gain in (.25,.5,1.):
                    yield f"{name}-residual-{weighting:g}-{ridge:g}-{gain:g}",{
                          **c,"coefficient":(gain*residual[:-1]).astype("f4"),
                          "bias":(gain*residual[-1]).astype("f4"),"original_scale":np.asarray(1.,np.float32)}
                for gain in (.25,.5,1.):
                    # Compare a directional replacement head against a score-preserving residual.
                    yield f"{name}-probe-{weighting:g}-{ridge:g}-{gain:g}",{
                          **c,"coefficient":(gain*t*probe[:-1]).astype("f4"),
                          "bias":(gain*t*probe[-1]).astype("f4"),"original_scale":np.asarray(1-gain,np.float32)}


def edit_decode(bank,x,edit):
    y=bank.logits
    if edit is None:
        return bank.decode(ZERO,ONE)
    try:
        f=transform(x,edit)
        bank.logits=np.ascontiguousarray(y*edit["original_scale"]+f@edit["coefficient"]+edit["bias"],dtype=np.float64)
        return bank.decode(ZERO,ONE)
    finally:
        bank.logits=y


def cm_case(bank,pred):
    return np.stack([cm(bank.gold[i:j],pred[i:j]) for i,j in zip(bank.sizes,bank.sizes[1:])])


def main():
    torch.set_num_threads(2)
    base=load_file(str(OUT/"known-sentence-role-pair.safetensors"))
    released=load_file(str(OUT/"known-sentence-label-change.safetensors"))
    vocab=json.loads((EDGE/"vocab.json").read_text())
    model=HierModel(DONOR)
    engine=load_decoder()

    def package(docs,states):
        docs=[{**d,"units":d["reference_units"]} for d in docs]
        raw=[model.logits(s) for s in states]
        bank=[(prepare_score(d,x,{"exact":base}),transition_scores(d["text"],d["units"],vocab,base))
              for d,x in zip(docs,raw)]
        return KnownSentences(bank,base,engine).bind(docs),np.concatenate(states)

    def control(bank):
        return np.concatenate([decode_profile(row.logits["exact"],edge[1:],released,bank.scale)
               for row,edge in zip(bank.items,np.split(bank.edge,bank.sizes[1:-1]))])

    train=training_documents()
    states=list(embeddings(model,train[:144],"aligned-source-train"))
    for i,d in enumerate(train[144:],1):
        states.append(next(aligned_embeddings(model,[d],"train")))
        if i%48==0:print("Encoded additional official training cases",i,flush=True)
    dataset,states=package(train,states)
    candidates={"released":None}
    candidates.update(train_weights(states,dataset.gold,dataset.logits,control(dataset)))
    cohorts=sample_whole()

    def pack(s):
        docs=cohorts[s]
        return package(docs,list(aligned_embeddings(model,docs,s)))

    def measure_labels(bank,pred):
        return measure(cm(bank.gold,pred)).tolist()

    bank,x=pack("select")
    vals={"released":measure_labels(bank,control(bank))}
    vals.update({k:measure_labels(bank,edit_decode(bank,x,c)) for k,c in candidates.items() if c is not None})
    winner=max(vals,key=lambda k:(vals[k][0],vals[k][1],k=="released"))
    bound=vals["released"][2]-.005
    eligible=[k for k,v in vals.items() if v[1]>=vals["released"][1]-.003 and v[2]>=bound]
    protective=max(eligible,key=lambda k:(vals[k][0],vals[k][1],k=="released"))
    choices={"macro":winner,"retention":protective}
    side={}
    for profile,name in choices.items():
        if name!="released":
            path=OUT/f"known-sentence-hidden-{profile}.safetensors"
            save_file({**base,**{"hidden."+k:v for k,v in candidates[name].items()}},str(path))
            side[profile]=file_hash(path)
        print("Hidden readout choice",profile,name,vals[name],"baseline",vals["released"],flush=True)
    freeze=ROOT/"results/sentence-hidden-edit-frozen.json"
    save_json(freeze,{"choices":choices,"values":vals,"train_cases":[d["id"] for d in train],"train_rows":dataset.units,
                      "train_width":states.shape[1],"selection_cases":[d["id"] for d in cohorts["select"]],"weights_sha256":side})
    excluded={d["id"] for q in cohorts.values() for d in q}
    omitted=set(json.loads((EDGE/"overlap.json").read_text())["independent_exclusions_for_text_disjoint_check"])
    cohorts["independent"]=[d for d in independent_validation(excluded) if d["id"] not in omitted]
    report={"choices":choices,"freeze_sha256":file_hash(freeze),"splits":{},
            "protocol":"Supervised closed-form ridge writes on frozen full judgment representations. Compare a linear head with nonlinear fixed random features, each fitted either as a direct class probe or as a correction to category margins. All neural recurrent weights remain fixed. Selected heads are written before official test. Retention requires accuracy within 0.3 points and reasoning F1 within 0.5 points of the released profile on development."}

    for split in ("select","confirm","independent","test"):
        if split=="test":
            if all(k=="released" for k in choices.values()):
                break
            cohorts[split]=[document(k,r) for k,r in gold_cases().items()]
            docs=cohorts[split]
            vectors=[]
            for i,s in enumerate(aligned_embeddings(model,docs,split),1):
                vectors.append(s)
                if i%120==0:print("Encoded supplied sentences",i,flush=True)
            data,h=package(docs,vectors)
        else:
            data,h=(bank,x) if split=="select" else pack(split)
        baseline=cm_case(data,control(data))
        entries={}
        for profile,k in choices.items():
            if k=="released":
                a=baseline
            else:
                path=OUT/f"known-sentence-hidden-{profile}.safetensors"
                p=load_file(str(path))
                weights={k.removeprefix("hidden."):v for k,v in p.items() if k.startswith("hidden.")}
                a=cm_case(data,edit_decode(data,h,weights))
            entries[profile]={"cases":len(a),"candidate":row_summary(a.sum(0)),
                              "preceding":row_summary(baseline.sum(0)),"paired_to_preceding":paired(a,baseline)}
            print(split,profile,measure(a.sum(0)).tolist(),measure(baseline.sum(0)).tolist(),flush=True)
        report["splits"][split]=entries
        save_json(ROOT/"results/sentence-hidden-edit.json",report)


if __name__=="__main__":
    main()
