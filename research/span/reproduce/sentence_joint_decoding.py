"""Write posterior and sequence-consistency weights for LegalSeg sentence roles."""
from itertools import product
import json

import numpy as np
from safetensors.numpy import load_file,save_file

from sentence_hidden_edit import (
    CRF,DONOR,EDGE,ONE,OUT,ROOT,ZERO,HierModel,KnownSentences,aligned_embeddings,cm_case,
    document,embeddings,file_hash,gold_cases,independent_validation,load_decoder,prepare_score,
    row_summary,sample_whole,save_json,torch,training_documents,transition_scores,
)
from sentence_uncertainty_runtime import uncertainty_evidence,probability,apply_uncertainty
from sentence_joint_runtime import distribution,input_arrays,utility
from predict_role_edge import decode_profile
from sentence_role_retention import measure,paired

PARENT=OUT/"known-sentence-uncertainty.safetensors"
CHECKPOINT=OUT/"known-sentence-joint.safetensors"
RATES=(.30,.55,1.)


def package(docs,states,model,meta,vocab,engine):
    ds=[{**d,"units":d["reference_units"]} for d in docs]
    items=[];pieces=[];p=[];old=[];ranks=[]
    rate,mix,margin=meta["uncertainty.settings"]
    weights={k:meta["uncertainty."+k] for k in ("center","scale","weight","bias")}
    for d,state in zip(ds,states):
        raw=prepare_score(d,model.logits(state),{"exact":meta})
        edge=transition_scores(d["text"],d["units"],vocab,meta)
        selected=decode_profile(raw.logits["exact"],edge,meta,float(meta["edge.gain"]))
        lengths=[len(d["text"][a:b].strip()) for a,b in d["units"]]
        order,prior,features=uncertainty_evidence(raw.logits["exact"],edge,meta,float(meta["edge.gain"]),
                                                  selected,lengths,1.)
        estimate=probability(features[order],weights)
        revised=apply_uncertainty(selected,order,prior,estimate,rate,mix,margin)
        marginal=.45*features[:,:7]+.55*features[:,7:14]
        odds=1.-marginal[np.arange(len(marginal)),revised]
        order=np.argsort(-odds,kind="stable")
        rank=np.empty(len(order))
        rank[order]=np.arange(len(order))/len(order)
        items.append((raw,edge))
        pieces.append(features)
        old.append(revised)
        p.append(marginal)
        ranks.append(rank)
    bank=KnownSentences(items,meta,engine).bind(ds)
    current=np.concatenate(old).astype(np.int8)
    marginals=np.concatenate(p)
    structure,states=input_arrays(np.concatenate(pieces),marginals,np.concatenate(states),bank.start_mask)
    return bank,current,marginals,np.concatenate(ranks),structure,states


def weights_for(x,gold,current,kind):
    m=x.mean(0,dtype=np.float64).astype("f4")
    scale=(1/np.maximum(x.std(0,dtype=np.float64),.035)).astype("f4")
    h=((x-m)*scale).astype(np.float64)
    z=np.column_stack((h,np.ones(len(h))))
    y=np.eye(7)[gold]
    for style in (("shared",) if kind=="structure" else ("conditional",)):
        for ridge in (.075,.22):
            heads=[]
            groups=range(7) if style=="conditional" else (None,)
            for label in groups:
                keep=slice(None) if label is None else current==label
                zh=z[keep]
                yt=y[keep]
                gram=zh.T@zh
                d=np.r_[np.ones(gram.shape[0]-1),.035]
                cov=gram+np.diag(d*(ridge*np.trace(gram)/len(gram)))
                heads.append(np.linalg.solve(cov,zh.T@yt))
            w=np.stack(heads).astype("f4")
            yield (kind,style,ridge),{"center":m,"scale":scale,"weight":np.ascontiguousarray(w[:,:-1]),
                                      "bias":np.ascontiguousarray(w[:,-1])}


def train_priors(bank,current):
    matrix=np.bincount(7*bank.gold+current,minlength=49).reshape(7,7)
    predicted=matrix.sum(0)
    gold=matrix.sum(1)
    mean=(predicted+gold)/(2*len(bank.gold))
    f1=2*np.diag(matrix)/(predicted+gold)
    use=~bank.start_mask[1:]
    g=bank.gold
    counts=np.bincount(7*g[:-1][use]+g[1:][use],minlength=49).reshape(7,7)
    smoothing=24*np.maximum(gold/gold.sum(),.003)
    count=counts+smoothing
    q=count/count.sum(1,keepdims=True)
    tendency=np.log(q)-np.log(np.maximum(gold/gold.sum(),.003))
    tendency=(tendency-tendency.mean(1,keepdims=True)).clip(-2.5,2.5)
    # Temperature-independent class proportions and adjacent annotated role statistics.
    return {"denominator":mean.astype("f4"),"ref_f1":f1.astype("f4"),"transition":tendency.astype("f4")}


def predict(bank,current,rank,values,rate,tendency,strength):
    core=bank.transition
    raw=bank.logits
    gain=bank.scale
    try:
        matrix=np.zeros_like(core)
        matrix[3:,3:]=strength*tendency
        bank.transition=matrix
        bank.scale=0.
        if rate<1.:
            costs=np.where((rank[:,None]<rate)|(np.arange(7)[None,:]==current[:,None]),values,-1.e6)
        else:
            costs=values
        bank.logits=np.ascontiguousarray(costs,dtype=np.float64)
        return bank.decode(ZERO,ONE)
    finally:
        bank.transition=core
        bank.logits=raw
        bank.scale=gain


def main():
    torch.set_num_threads(2)
    f=load_file(str(PARENT))
    vocab=json.loads((EDGE/"vocab.json").read_text())
    engine=load_decoder()
    model=HierModel(DONOR)
    fit=training_documents()
    train=list(embeddings(model,fit[:144],"aligned-source-train"))+list(aligned_embeddings(model,fit[144:],"train"))
    data,current,p,rank,structure,h=package(fit,train,model,f,vocab,engine)
    priors=train_priors(data,current)
    readouts={("posterior","fixed",0.):None}
    readouts.update(weights_for(structure,data.gold,current,"structure"))
    readouts.update(weights_for(h,data.gold,current,"context"))
    print("Joint readout fitted",len(fit),len(current),len(readouts),flush=True)
    del train,data,current,p,rank,structure,h
    groups=sample_whole()
    used={d["id"] for cases in groups.values() for d in cases}
    dropped=set(json.loads((EDGE/"overlap.json").read_text())["independent_exclusions_for_text_disjoint_check"])
    groups["independent"]=[d for d in independent_validation(used) if d["id"] not in dropped]

    def prepare(name):
        d=groups[name]
        x=list(aligned_embeddings(model,d,name))
        a=package(d,x,model,f,vocab,engine)
        print("Joint prepared",name,len(d),a[0].units,flush=True)
        return a

    dev,old,marg,rank,struct,hid=prepare("select")
    base=measure(cm_case(dev,old).sum(0)).tolist()
    grid=[];accepted=[];fallback=[]
    caps=(.0,.18,.45,.75)
    for key,edit in readouts.items():
        kind=key[0]
        estimated=None if edit is None else distribution(hid if kind=="context" else struct,old,edit)
        for macro,keep in product(caps,(.015,.075,.16,.26)):
            e=utility(marg,estimated,old,macro,priors,keep)
            for rate,strength in product(RATES,(.0,.055,.13)):
                y=predict(dev,old,rank,e,rate,priors["transition"],strength)
                n=int((y!=old).sum())
                q=measure(cm_case(dev,y).sum(0)).tolist()
                s=(*key,macro,keep,rate,strength)
                grid.append([*s,*q,n])
                if n:
                    item=(q[0],q[1],s)
                    fallback.append(item)
                    if q[1]>=base[1]-.008:
                        accepted.append(item)
    best=max(accepted or fallback)
    choice=best[2]
    tensors={**priors,**(readouts[choice[:3]] or {}),"settings":np.asarray(choice[3:])}
    save_file(tensors,str(CHECKPOINT))
    physical=load_file(str(CHECKPOINT))
    assert all(np.array_equal(v,physical[k]) for k,v in tensors.items())
    frozen=ROOT/"results/sentence-joint-frozen.json"
    save_json(frozen,{"choice":choice,"grid":grid,"prior_accuracy":base,"selected":best[:2],
                      "source_sha256":file_hash(PARENT),"weights_sha256":file_hash(CHECKPOINT),
                      "training_cases":[d["id"] for d in fit],"selection_cases":[d["id"] for d in groups["select"]]})
    print("Joint decoder choice",choice,best[:2],base,flush=True)
    report={"choice":choice,"freeze_sha256":file_hash(frozen),"splits":{},
       "protocol":"Estimate expected per-role F1 utility from marginal role probabilities. Compare hierarchy posteriors with structural and conditional latent closed-form readouts. Add an annotated transition matrix and an incumbent-label retention score, and decode jointly under a validation-selected routing budget of 30, 55, or 100 percent."}
    for name in ("select","confirm","independent","test"):
        if name=="test":
            groups[name]=[document(k,r) for k,r in gold_cases().items()]
        bank,current,marg,rank,s,h=(dev,old,marg,rank,struct,hid) if name=="select" else prepare(name)
        kind=choice[0]
        ph=None if kind=="posterior" else distribution(h if kind=="context" else s,current,physical)
        macro,retention,rate,strength=physical["settings"]
        e=utility(marg,ph,current,macro,physical,retention)
        y=predict(bank,current,rank,e,rate,physical["transition"],strength)
        a,b=cm_case(bank,y),cm_case(bank,current)
        if name=="select":
            np.testing.assert_allclose(measure(a.sum(0))[:2],best[:2],atol=1e-12,rtol=0)
        if name=="test":
            checked=json.loads((ROOT/"results/sentence-uncertainty-qa.json").read_text())["scores"]
            assert row_summary(b.sum(0))==checked
        report["splits"][name]={"cases":len(a),"candidate":row_summary(a.sum(0)),"preceding":row_summary(b.sum(0)),
               "paired_to_preceding":paired(a,b),"modified":int((y!=current).sum()),
               "eligible_sentences":int((rank<rate).sum())}
        save_json(ROOT/"results/sentence-joint-edit.json",report)
        print(name,measure(a.sum(0)).tolist(),measure(b.sum(0)).tolist(),flush=True)


if __name__=="__main__":
    main()
