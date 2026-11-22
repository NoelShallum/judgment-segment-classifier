"""Write a role-pair utility table for the frozen lexical sentence edit."""
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import json

import numpy as np
from safetensors.numpy import save_file

from sentence_ngram_residual_write import (
    STEM as LEX,OUT,PARENT,ROOT,EDGE,document,gold_cases,independent_validation,load_file,measure,
    paired,predict,prepare,route_data,row_summary,evidence,sample_whole,save_json,stats,file_hash,torch,
)
from sentence_hidden_edit import training_documents

STEM="sentence-ngram-pair-gate"
CUTS=np.array([-.12,-.04,.025,.075,.15,.26])
BINS=len(CUTS)+1
CHOICES=("unchanged","lexical","guard-reasoning","utility")


def proposals(cases,directions,routes,settings):
    rate,gain,cut=settings
    data=[]
    for (c,_),z,r in zip(cases,directions,routes):
        q=r[:3] if not rate else r[:max(1,int(np.ceil(rate*len(c.current))))]
        before=c.current[q]
        current=c.prior[q]+gain*z[q]
        row=np.arange(len(q))
        gap=current-current[row,before,None]
        bins=np.searchsorted(CUTS,gap,side="right")
        data.append((q,before,gap,bins))
    return data


def utility_tables(cases,data):
    baseline=stats(cases,[c.current for c,_ in cases]).sum(0)
    denom=baseline.sum(0)+baseline.sum(1)
    f=2*baseline.diagonal()/denom
    normal=np.mean(denom)/np.maximum(denom,.035*baseline.sum())
    count=np.zeros((7,7,BINS))
    u=np.zeros_like(count)
    a=np.zeros_like(count)
    for (c,_),(q,before,gap,bins) in zip(cases,data):
        y=c.gold[q]
        left=before==y
        for dst in range(7):
            ids=np.flatnonzero((before!=dst)&(gap[:,dst]>=-.22))
            src=before[ids]
            right=(y[ids]==dst).astype(float)
            old=left[ids].astype(float)
            # First-order delta for both affected class-F1 denominators.
            delta=normal[dst]*(right-.5*f[dst])-normal[src]*(old-.5*f[src])
            ix=(src,np.full(len(ids),dst),bins[ids,dst])
            np.add.at(count,ix,1)
            np.add.at(u,ix,delta)
            np.add.at(a,ix,right-old)
    n=count.sum(2,keepdims=True)
    for smooth in (8.,30.):
        prior_u=u.sum(2,keepdims=True)/(n+12.)
        prior_a=a.sum(2,keepdims=True)/(n+12.)
        yield smooth,{"macro":(u+smooth*prior_u)/(count+smooth),
                      "accuracy":(a+smooth*prior_a)/(count+smooth),
                      "counts":count,"training_confusion":baseline}


def revise(cases,data,mode,table,parameters,settings):
    if mode=="unchanged":
        return [c.current for c,_ in cases]
    if mode in ("lexical","guard-reasoning"):
        guard=mode=="guard-reasoning"
        cut=settings[2]
    else:
        lam,gate,cut,depth=parameters
    predictions=[]
    for (c,_),(q,before,gap,bins) in zip(cases,data):
        y=c.current.copy()
        if mode in ("lexical","guard-reasoning"):
            top=gap.argmax(1)
            ok=gap[np.arange(len(q)),top]>=cut
            if guard:
                ok&=before!=5
        else:
            dst=np.arange(7)
            idx=(before[:,None],dst[None,:],bins)
            benefit=table["macro"][idx]+lam*table["accuracy"][idx]
            strength=benefit+.4*gap
            possible=(gap>=cut)&(before[:,None]!=dst)&(table["counts"][idx]>=4)&(benefit>=gate)
            if depth==1:
                possible&=dst==gap.argmax(1)[:,None]
            selection=np.where(possible,strength,-np.inf)
            top=selection.argmax(1)
            ok=np.isfinite(selection[np.arange(len(q)),top])
        y[q[ok]]=top[ok]
        predictions.append(y)
    return predictions


def packed(cases,head,mode):
    d=evidence(cases,head,mode)
    route=route_data(cases)
    return d,route,proposals(cases,d,route,head["settings"])


def replay(args):
    docs,lex_mode,choice=args
    mem=load_file(str(OUT/(LEX+"-memory.safetensors")))["lexical"]
    head=load_file(str(OUT/(LEX+".safetensors")))
    physical=load_file(str(OUT/(STEM+".safetensors"))) if choice[0]=="utility" else None
    docs_views=prepare(docs,"test",mem)
    d,r,data=packed(docs_views,head,lex_mode)
    new=revise(docs_views,data,choice[0],physical,choice[2:],head["settings"])
    old=predict(docs_views,r,d,*head["settings"])
    return [(c.key,aa,bb,cc,int(np.count_nonzero(x!=c.current))) for (c,_),aa,bb,cc,x in zip(
        docs_views,stats(docs_views,new),stats(docs_views,old),stats(docs_views,[c.current for c,_ in docs_views]),new)]


def main():
    torch.set_num_threads(1)
    root=ROOT/"results"
    incumbent=json.loads((root/(LEX+"-frozen.json")).read_text())
    lex_mode=incumbent["choice"][0]
    memfile=OUT/(LEX+"-memory.safetensors")
    headfile=OUT/(LEX+".safetensors")
    assert file_hash(memfile)==incumbent["memory_sha256"] and file_hash(headfile)==incumbent["head_sha256"]
    mem=load_file(str(memfile))["lexical"]
    head=load_file(str(headfile))
    seen=set(incumbent["memory_training_cases"])|set(incumbent["readout_training_cases"])
    training=[d for d in training_documents() if d["id"] not in seen]
    print("Calibrating on",len(training),"untouched training judgments",flush=True)
    fit=prepare(training,"train",mem)
    _,_,f=packed(fit,head,lex_mode)
    options=dict(utility_tables(fit,f))
    del fit,f
    cohorts=sample_whole()
    dev=prepare(cohorts["select"],"select",mem)
    dz,dr,dp=packed(dev,head,lex_mode)
    bz=measure(stats(dev,[c.current for c,_ in dev]).sum(0))
    candidates=[(x,) for x in CHOICES[:3]]
    candidates.extend(("utility",smooth,*p) for smooth in options for p in product(
        (.15,.55),(-.055,.005,.065),(-.03,.015,.045),(1,7)))
    ranking=[]
    for cfg in candidates:
        table=options[cfg[1]] if cfg[0]=="utility" else None
        y=revise(dev,dp,cfg[0],table,cfg[2:],head["settings"])
        q=measure(stats(dev,y).sum(0))
        feasible=bool(q[2]>=bz[2]-.001 and q[1]>=bz[1]-.0003)
        ranking.append((feasible,float(q[0]),float(q[1]),cfg[0]=="unchanged",cfg,q.tolist()))
    best=max(ranking)
    choice=best[4]
    sha=None
    if choice[0]=="utility":
        out=OUT/(STEM+".safetensors")
        payload={**options[choice[1]],"settings":np.asarray(choice[2:])}
        save_file(payload,str(out))
        table=load_file(str(out))
        for k,v in payload.items():
            np.testing.assert_array_equal(v,table[k])
        sha=file_hash(out)
    else:
        table=None
    freeze=root/(STEM+"-frozen.json")
    save_json(freeze,{"choice":choice,"train_cases":[d["id"] for d in training],
                      "selection_cases":[c.key for c,_ in dev],"score_grid":[[r[4],r[0],r[5]] for r in ranking],
                      "parent_sha256":file_hash(PARENT),"lexical_head_sha256":file_hash(headfile),
                      "lexical_memory_sha256":file_hash(memfile),"gate_sha256":sha,"selection_baseline":bz.tolist()})
    print("Frozen gate",choice,best[5],bz.tolist(),flush=True)
    used={d["id"] for batches in cohorts.values() for d in batches}
    excluded=set(json.loads((EDGE/"overlap.json").read_text())["independent_exclusions_for_text_disjoint_check"])
    cohorts["independent"]=[d for d in independent_validation(used) if d["id"] not in excluded]
    old_scores=json.loads((root/(LEX+".json")).read_text())["splits"]
    result={"freeze_sha256":file_hash(freeze),"splits":{}}
    for split in ("select","confirm","independent","test"):
        if split=="test":
            if choice[0]=="unchanged":
                break
            docs=[document(k,v) for k,v in gold_cases().items()]
            with ProcessPoolExecutor(3) as pool:
                result_rows={k:(a,b,c,n) for rows in pool.map(replay,[(docs[i::3],lex_mode,choice) for i in range(3)])
                             for k,a,b,c,n in rows}
            order=[result_rows[d["id"]] for d in docs]
            a,b,c=(np.stack([r[i] for r in order]) for i in range(3))
            n=sum(r[3] for r in order)
        else:
            cases=dev if split=="select" else prepare(cohorts[split],split,mem)
            z,rr,data=(dz,dr,dp) if split=="select" else packed(cases,head,lex_mode)
            candidate=revise(cases,data,choice[0],table,choice[2:],head["settings"])
            a=stats(cases,candidate)
            b=stats(cases,predict(cases,rr,z,*head["settings"]))
            c=stats(cases,[w.current for w,_ in cases])
            n=sum(int((x!=w.current).sum()) for x,(w,_) in zip(candidate,cases))
        recorded=old_scores[split]
        item={"cases":len(a),"changed":n,"candidate":row_summary(a.sum(0)),"lexical":row_summary(b.sum(0)),
              "baseline":row_summary(c.sum(0)),"paired_to_baseline":paired(a,c),"paired_to_lexical":paired(a,b)}
        assert item["baseline"]==recorded["baseline"] and item["lexical"]==recorded["candidate"]
        result["splits"][split]=item
        save_json(root/(STEM+".json"),result)
        print(split,item["candidate"]["macro_f1"],item["candidate"]["accuracy"],item["candidate"]["per_role_f1"]["reasoning"],n,flush=True)


if __name__=="__main__":
    main()
