"""Write multiple cosine prototypes per LegalSeg role and a residual sentence head."""
from itertools import product
import json

import numpy as np
from safetensors.numpy import load_file,save_file

from sentence_hidden_edit import (
    DONOR, EDGE, ONE, OUT, ROOT, ZERO, HierModel, KnownSentences, aligned_embeddings,
    cm_case, document, embeddings, file_hash, gold_cases, independent_validation,
    load_decoder, prepare_score, row_summary, sample_whole, save_json, torch,
    training_documents, transition_scores,
)
from predict_role_edge import decode_profile
from sentence_role_retention import measure, paired


def normalize(x):
    return (x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1.e-6)).astype("f4")


def directions(train, shrink):
    mean=train.mean(0,dtype=np.float64)
    x=train-mean
    if shrink==0:
        mapping=np.diag(1/np.maximum(x.std(0),.04))
    else:
        cov=x.T@x/len(x)
        s,v=np.linalg.eigh(cov)
        floor=shrink*cov.trace()/len(s)
        mapping=(v/np.sqrt(s+floor))@v.T
    return {"center":mean.astype("f4"),"map":mapping.astype("f4")}


def bank_of_centers(x,gold,count,seed):
    rng=np.random.default_rng(seed)
    memory=[]
    for role in range(7):
        h=x[gold==role]
        c=h[rng.choice(len(h),count,replace=False)].copy()
        for it in range(7):
            choose=np.argmax(h@c.T,axis=1)
            hits=np.bincount(choose,minlength=count)
            sums=np.eye(count,dtype=np.float32)[choose].T@h
            bad=hits==0
            sums[bad]=c[bad]
            c=normalize(sums)
        memory.append(c)
    return np.stack(memory)


def representation(states,score,bank):
    center=score-score.mean(1,keepdims=True)
    h=np.column_stack((states,.22*center)).astype("f4")
    h=normalize((h-bank["center"])@bank["map"])
    n=len(h)
    k=bank["memory"].shape[1]
    cos=np.ascontiguousarray(h@bank["memory"].reshape(7*k,-1).T)
    votes=np.partition(cos.reshape(n,7,k),k-2,axis=2)[:,:,-2:].mean(2)
    activation=np.maximum(cos-.15,0.)
    return votes,activation


def packed_decode(bank,meta,logits):
    original=bank.logits
    try:
        bank.logits=np.ascontiguousarray(logits,dtype=np.float64)
        first=bank.decode(ZERO,ONE)
        refined=logits@meta["emission.weight"]+meta["emission.bias"]
        bank.logits=np.ascontiguousarray(refined,dtype=np.float64)
        second=bank.decode(ZERO,ONE)
    finally:
        bank.logits=original
    i=np.arange(len(first))
    gap=refined[i,second]-refined[i,first]
    cell=np.digitize(gap,meta["label_change.margin_bins"])
    value=meta["label_change.utility"][first,second,cell]
    return np.where(value>meta["label_change.threshold"].item(),second,first)


def fit(h,pack,old):
    raw=np.column_stack((h,.22*(pack.logits-pack.logits.mean(1,keepdims=True)))).astype("f4")
    gold=pack.gold
    prior=np.bincount(gold,minlength=7)/len(gold)
    role=np.eye(7)[gold]
    error=gold!=old
    margins=np.clip(pack.logits[np.arange(len(old)),old]-pack.logits[np.arange(len(old)),gold]+.5,.45,2.4)
    delta=np.zeros_like(role)
    i=np.flatnonzero(error)
    delta[i,old[i]]= -.5*margins[i]
    delta[i,gold[i]]= .5*margins[i]
    for shrink,num in product((0.,.3),(12,28)):
        cfg=directions(raw,shrink)
        norm=normalize((raw-cfg["center"])@cfg["map"])
        cfg["memory"]=bank_of_centers(norm,gold,num,271828+num+int(10*shrink))
        vote,activation=representation(h,pack.logits,cfg)
        centered=vote-vote.mean(1,keepdims=True)
        cfg["vote_scale"]=np.asarray(1/max(centered.std(),.025),dtype=np.float32)
        yield (shrink,num,"vote",0.),cfg
        mean=activation.mean(0)
        sigma=1/np.maximum(activation.std(0),.04)
        z=np.column_stack(((activation-mean)*sigma,np.ones(len(activation)))).astype(np.float64)
        for balance in (.15,.45):
            w=prior[gold]**(-balance)
            gram=z.T@(z*w[:,None])
            reg=np.ones(len(gram))
            reg[-1]=.04
            g=gram+np.diag(reg*.12*gram.trace()/len(gram))
            for target,how in ((delta,"residual"),(role,"class")):
                solved=np.linalg.solve(g,z.T@(target*w[:,None]))
                if how=="class":
                    tr=z@solved
                    clean=tr-tr.mean(1,keepdims=True)
                    solved*=1/max(clean.std(),.05)
                yield (shrink,num,how,balance),{**cfg,"activation_center":mean,"activation_scale":sigma,
                                              "weight":solved[:-1].astype("f4"),"bias":solved[-1].astype("f4")}


def additional(h,x,params,mode):
    vote,activation=representation(h,x,params)
    if mode=="vote":
        added=vote*params["vote_scale"]
    else:
        added=((activation-params["activation_center"])*params["activation_scale"])@params["weight"]+params["bias"]
    return (added-added.mean(1,keepdims=True)).astype("f4")


def revised(bank,meta,added,mode,gain):
    raw=bank.logits
    if mode=="class":
        extra=(1-gain)*raw+gain*added
    else:
        extra=raw+gain*added
    return packed_decode(bank,meta,extra)


def main():
    torch.set_num_threads(2)
    model=HierModel(DONOR)
    old_path=OUT/"known-sentence-label-change.safetensors"
    old=load_file(str(old_path))
    vocab=json.loads((EDGE/"vocab.json").read_text())
    engine=load_decoder()

    def pack(docs,h):
        cases=[{**d,"units":d["reference_units"]} for d in docs]
        inputs=[(prepare_score(d,model.logits(x),{"exact":old}),transition_scores(d["text"],d["units"],vocab,old))
                for d,x in zip(cases,h)]
        packed=KnownSentences(inputs,old,engine).bind(cases)
        native=packed_decode(packed,old,packed.logits)
        for index,(item,edge) in enumerate(inputs[:2]):
            y=decode_profile(item.logits["exact"],edge,old,packed.scale)
            lo,hi=packed.sizes[index:index+2]
            np.testing.assert_array_equal(native[lo:hi],y)
        return packed,np.concatenate(h)

    train=training_documents()
    h=list(embeddings(model,train[:144],"aligned-source-train"))+list(aligned_embeddings(model,train[144:],"train"))
    p,x=pack(train,h)
    old_y=packed_decode(p,old,p.logits)
    memory=dict(fit(x,p,old_y))
    print("Prototype weights fitted",len(train),p.units,len(memory),flush=True)
    del p,x,h
    groups=sample_whole()

    def prepare(name):
        docs=groups[name]
        return pack(docs,list(aligned_embeddings(model,docs,name)))

    val,h=prepare("select")
    baseline=packed_decode(val,old,val.logits)
    control=measure(cm_case(val,baseline).sum(0))
    grid=[]
    eligible=[];fallback=[]
    for key,w in memory.items():
        mode=key[2]
        add=additional(h,val.logits,w,mode)
        gains=(.05,.12,.25,.45) if mode=="class" else (.07,.18,.4,.75)
        for gain in gains:
            out=revised(val,old,add,mode,gain)
            numbers=measure(cm_case(val,out).sum(0)).tolist()
            spec=(*key,gain)
            count=int((out!=baseline).sum())
            grid.append([*spec,*numbers,count])
            if count:
                entry=(numbers[0],numbers[1],spec)
                fallback.append(entry)
                if numbers[1]>=control[1]-.004:
                    eligible.append(entry)
    best=max(eligible or fallback)
    choice=best[2]
    select_improved=bool(best[0]>control[0])
    path=OUT/"known-sentence-prototypes.safetensors"
    payload={**memory[choice[:4]],"mode":np.asarray({"vote":0,"residual":1,"class":2}[choice[2]],np.int16),
             "gain":np.asarray(choice[4])}
    save_file({**old,**{"prototype."+k:v for k,v in payload.items()}},str(path))
    loaded=load_file(str(path))
    physical={k.removeprefix("prototype."):v for k,v in loaded.items() if k.startswith("prototype.")}
    assert all(np.array_equal(v,physical[k]) for k,v in payload.items())
    frozen=ROOT/"results/sentence-prototype-frozen.json"
    save_json(frozen,{"choice":choice,"selection_improved":select_improved,"grid":grid,"preceding":control.tolist(),
                "selection_cases":[d["id"] for d in groups["select"]],"training_cases":[d["id"] for d in train],
                "preceding_sha256":file_hash(old_path),"weights_sha256":file_hash(path)})
    print("Prototype choice",choice,best[:2],control.tolist(),flush=True)
    used={d["id"] for seq in groups.values() for d in seq}
    omitted=set(json.loads((EDGE/"overlap.json").read_text())["independent_exclusions_for_text_disjoint_check"])
    groups["independent"]=[d for d in independent_validation(used) if d["id"] not in omitted]
    report={"choice":choice,"freeze_sha256":file_hash(frozen),"splits":{},
        "protocol":"Cluster normalized recurrent state signatures separately for each of seven roles, with 12 or 28 prototypes per role. Compare cosine nearest-prototype voting and a closed-form weighted readout of nonlinear similarities. Keep encoder and CRF frozen. Write prototype matrices and chosen readout before confirmation and official test."}
    for name in ("select","confirm","independent","test"):
        if name=="test":
            groups[name]=[document(k,r) for k,r in gold_cases().items()]
        bp,xx=(val,h) if name=="select" else prepare(name)
        mode=choice[2]
        before=packed_decode(bp,old,bp.logits)
        after=revised(bp,old,additional(xx,bp.logits,physical,mode),mode,float(physical["gain"]))
        a,b=cm_case(bp,after),cm_case(bp,before)
        if name=="select":
            np.testing.assert_allclose(measure(a.sum(0))[:2],best[:2],rtol=0.,atol=1e-12)
        if name=="test":
            expect=json.loads((ROOT/"results/sentence-label-change-qa.json").read_text())["scores"]
            assert row_summary(b.sum(0))==expect
        report["splits"][name]={"cases":len(a),"candidate":row_summary(a.sum(0)),"preceding":row_summary(b.sum(0)),
              "paired_to_preceding":paired(a,b),"modified":int((before!=after).sum())}
        save_json(ROOT/"results/sentence-prototype-edit.json",report)
        print(name,measure(a.sum(0)).tolist(),measure(b.sum(0)).tolist(),flush=True)


if __name__=="__main__":
    main()
