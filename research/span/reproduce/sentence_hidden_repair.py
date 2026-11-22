"""Write conditional residual classifiers for mistakes in the released sentence profile."""
from itertools import product
import json

import numpy as np
from safetensors.numpy import load_file,save_file

from sentence_hidden_edit import (
    DONOR,EDGE,OUT,ROOT,HierModel,KnownSentences,aligned_embeddings,cm,cm_case,document,
    embeddings,file_hash,gold_cases,independent_validation,load_decoder,prepare_score,
    row_summary,sample_whole,save_json,torch,training_documents,transition_scores,transform
)
from predict_role_edge import decode_profile
from sentence_role_retention import measure,paired


def fit(states,scores,gold,old,fit_mask):
    x=np.column_stack((states,scores-scores.mean(1,keepdims=True)))
    center=x[fit_mask].mean(0).astype("f4")
    sigma=(1/np.maximum(x[fit_mask].std(0),.034)).astype("f4")
    base={"mean":center,"scale":sigma}
    width=x.shape[1]
    random=np.random.default_rng(706734)
    expansion={**base,"projection":(random.standard_normal((width,220))/np.sqrt(width)).astype("f4"),
               "offset":random.uniform(-1,1,220).astype("f4")}
    frequency=np.bincount(gold[fit_mask],minlength=7)/fit_mask.sum()
    y=np.eye(7)[gold]
    for name,model in (("linear",base),("expanded",expansion)):
        h=transform(x,model)
        for rarity,ridge in product((0.,.25),(.075,.22)):
            individual=[]
            for label in range(7):
                active=fit_mask&(old==label)
                features=np.column_stack((h[active],np.ones(active.sum()))).astype(np.float64)
                w=frequency[gold[active]]**-rarity
                gram=features.T@(features*w[:,None])
                reg=np.eye(len(gram))*ridge*np.trace(gram)/len(gram)
                reg[-1,-1]*=.02
                result=np.linalg.solve(gram+reg,features.T@(y[active]*w[:,None]))
                individual.append(result)
            readout=np.stack(individual).astype("f4")
            # Safetensors serializes the backing bytes, so these views need their own storage.
            yield f"{name}-{rarity:g}-{ridge:g}",{
                **model,"weight":np.ascontiguousarray(readout[:,:-1]),"bias":np.ascontiguousarray(readout[:,-1])
            }


def proposals(h,state,old,weights):
    x=np.column_stack((h,state-state.mean(1,keepdims=True)))
    x=transform(x,weights)
    y=np.empty(len(old),np.int64)
    gap=np.empty(len(old))
    for label in range(7):
        active=old==label
        score=x[active]@weights["weight"][label]+weights["bias"][label]
        selected=score.argmax(1)
        y[active]=selected
        gap[active]=score[np.arange(len(score)),selected]-score[:,label]
    return y,gap


def learned_utility(gold,prev,next_,gap,held):
    """Shrink role-pair statistics toward the class-independent score bin."""
    cut=np.asarray([.06,.15,.28,.45])
    bin_=np.digitize(gap,cut)
    width=len(cut)+1
    active=held&(prev!=next_)
    if not active.any():
        return cut.astype("f4"),np.zeros((7,7,width),np.float32)
    keys=width*(7*prev+next_)+bin_
    gains=(gold==next_).astype(np.int64)-(gold==prev)
    n=np.bincount(keys[active],minlength=49*width).reshape(7,7,width)
    k=np.bincount(keys[active],weights=gains[active],minlength=49*width).reshape(7,7,width)
    base=k.sum(2)/(n.sum(2)+12)
    value=(k+12*base[:,:,None])/(n+12)
    return cut.astype("f4"),value.astype("f4")


def main():
    torch.set_num_threads(2)
    model=HierModel(DONOR)
    released=load_file(str(OUT/"known-sentence-label-change.safetensors"))
    vocabulary=json.loads((EDGE/"vocab.json").read_text())
    engine=load_decoder()
    def package(docs,features):
        cases=[{**d,"units":d["reference_units"]} for d in docs]
        bank=[(prepare_score(d,model.logits(s),{"exact":released}),transition_scores(d["text"],d["units"],vocabulary,released))
              for d,s in zip(cases,features)]
        packed=KnownSentences(bank,released,engine).bind(cases)
        old=np.concatenate([decode_profile(q.logits["exact"],a,released,packed.scale) for q,a in bank]).astype(np.int64)
        return packed,np.concatenate(features),old

    train=training_documents()
    states=list(embeddings(model,train[:144],"aligned-source-train"))
    states.extend(aligned_embeddings(model,train[144:],"train"))
    data,x,old=package(train,states)
    # Samples for judging change reliability are case-disjoint from the ridge fit.
    check_case=np.asarray([int(d["id"])%5==0 for d in train])
    check_rows=np.repeat(check_case,np.diff(data.sizes))
    candidates={}
    for k,w in fit(x,data.logits,data.gold,old,~check_rows):
        alt,gap=proposals(x,data.logits,old,w)
        bins,utility=learned_utility(data.gold,old,alt,gap,check_rows)
        candidates[k]={**w,"margin_bins":bins,"utility":utility}
    groups=sample_whole()
    def prepare(name):
        docs=groups[name]
        return package(docs,list(aligned_embeddings(model,docs,name)))

    val,h,source=prepare("select")
    chart={"released":measure(cm(val.gold,source)).tolist()}
    grid={}
    for name,p in candidates.items():
        a,d=proposals(h,val.logits,source,p)
        ids=np.digitize(d,p["margin_bins"])
        utility=p["utility"][source,a,ids]
        for margin,threshold in product((.05,.14,.25,.38),(-1.,0.,.08)):
            labels=np.where((d>=margin)&(utility>threshold),a,source)
            key=(name,margin,threshold)
            grid[key]=measure(cm(val.gold,labels)).tolist()
    floor=chart["released"][1]-.002
    winner=max((k for k in grid if grid[k][1]>=floor),
               key=lambda k:(grid[k][0],grid[k][1]),default=None)
    if winner is not None and grid[winner][0]<=chart["released"][0]:
        winner=None
    patch_dir=OUT/"known-sentence-hidden-repair.safetensors"
    if winner:
        p={**candidates[winner[0]],"min_margin":np.asarray(winner[1],np.float32),
           "min_utility":np.asarray(winner[2],np.float32)}
        save_file({**released,**{"repair."+k:v for k,v in p.items()}},str(patch_dir))
        physical={k.removeprefix("repair."):v for k,v in load_file(str(patch_dir)).items() if k.startswith("repair.")}
        assert all(np.array_equal(v,physical[k]) for k,v in p.items()),"Readout tensor failed serialization"
    print("Repair choice",winner,grid.get(winner),chart["released"],flush=True)
    path=ROOT/"results/sentence-hidden-repair-frozen.json"
    save_json(path,{"choice":winner,"grid":[[*k,*v] for k,v in grid.items()],"released":chart["released"],
                    "source_cases":[d["id"] for d in train],"reliability_cases":[d["id"] for d in train if int(d["id"])%5==0],
                    "selection_cases":[d["id"] for d in groups["select"]],"patch_sha256":file_hash(patch_dir) if winner else None})
    report={"choice":winner,"freeze_sha256":file_hash(path),"splits":{},
       "protocol":"Fit a closed-form sentence repair classifier conditional on the released predicted role. Use case-disjoint training predictions to estimate genuine benefit for each proposed label change and score-margin bin. Retain each released decision unless the written margin and reliability thresholds permit a change. Validation permits up to 0.2 points of sentence accuracy loss. Zero gradient steps."}
    spent={d["id"] for docs in groups.values() for d in docs}
    omitted=set(json.loads((EDGE/"overlap.json").read_text())["independent_exclusions_for_text_disjoint_check"])
    if winner:
        groups["independent"]=[d for d in independent_validation(spent) if d["id"] not in omitted]
    for s in ("select","confirm","independent","test") if winner else ("select",):
        if s=="test":
            groups[s]=[document(k,r) for k,r in gold_cases().items()]
        bank,h,old=(val,h,source) if s=="select" else prepare(s)
        if winner:
            a,d=proposals(h,bank.logits,old,physical)
            rating=physical["utility"][old,a,np.digitize(d,physical["margin_bins"])]
            new=np.where((d>=physical["min_margin"])&(rating>physical["min_utility"]),a,old)
        else:
            new=old
        a,b=cm_case(bank,new),cm_case(bank,old)
        if s=="select" and winner:
            np.testing.assert_allclose(measure(a.sum(0)),grid[winner],atol=1e-12,rtol=0)
        report["splits"][s]={"cases":len(a),"candidate":row_summary(a.sum(0)),"preceding":row_summary(b.sum(0)),
                              "paired_to_preceding":paired(a,b),"modified":int(np.count_nonzero(new!=old))}
        save_json(ROOT/"results/sentence-hidden-repair.json",report)
        print(s,measure(a.sum(0)).tolist(),measure(b.sum(0)).tolist(),flush=True)


if __name__=="__main__":
    main()
