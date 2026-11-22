"""Write conditional transition tensors from adjacent LegalSeg sentence pairs."""
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import json

import numpy as np
from safetensors.numpy import load_file,save_file

from sentence_hidden_edit import (
    CRF,DONOR,EDGE,ONE,OUT,ROOT,ZERO,HierModel,KnownSentences,aligned_embeddings,cm_case,
    document,embeddings,file_hash,gold_cases,independent_validation,load_decoder,prepare_score,
    row_summary,sample_whole,save_json,torch,training_documents,transition_scores,
)
from sentence_joint_runtime import joint_inputs
from sentence_role_retention import measure,paired
from predict_role_edge import decode_profile

PARENT=OUT/"known-sentence-joint-portable.safetensors"
DEST=OUT/"known-sentence-conditioned-edges.safetensors"
NAMES=("cosine","agreement","margin","position")
STRENGTHS=(.012,.030,.060,.10,.17)
BINS=5
OFF=~np.eye(7,dtype=bool)


def evidence(states,base,edge):
    norm=np.maximum(np.linalg.norm(states,axis=1),1e-7)
    cosine=(states[:-1]*states[1:]).sum(1)/(norm[:-1]*norm[1:])
    centered=base-base.max(1,keepdims=True)
    p=np.exp(centered)
    p/=p.sum(1,keepdims=True)
    agreement=(np.sqrt(p[:-1]*p[1:])).sum(1)
    interaction=edge.diagonal(axis1=1,axis2=2).mean(1)-edge[:,OFF].mean(1)
    position=np.arange(1,len(states))/len(states)
    return np.column_stack((cosine,agreement,interaction,position))


def package(docs,states,model,meta,vocab,engine):
    ds=[{**d,"units":d["reference_units"]} for d in docs]
    items=[];rows=[];anchor=[];ranks=[];values=[]
    gain=float(meta["edge.gain"])
    for d,h in zip(ds,states):
        raw=prepare_score(d,model.logits(h),{"exact":meta})
        edge=transition_scores(d["text"],d["units"],vocab,meta)
        logits=raw.logits["exact"]
        initial=decode_profile(logits,edge,meta,gain)
        lengths=[len(d["text"][a:b].strip()) for a,b in d["units"]]
        current,rank,value,_=joint_inputs(logits,edge,meta,gain,initial,lengths,h)
        rows.append(np.r_[np.zeros((1,len(NAMES))),evidence(h,logits,edge)])
        items.append((raw,edge))
        anchor.append(current);ranks.append(rank);values.append(value)
    bank=KnownSentences(items,meta,engine).bind(ds)
    current=np.concatenate(anchor)
    rank=np.concatenate(ranks)
    _,_,rate,strength=meta["joint.settings"]
    costs=np.where((rank[:,None]<rate)|(np.arange(7)[None,:]==current[:,None]),np.concatenate(values),-1.e6)
    bank.logits=np.ascontiguousarray(costs,dtype=np.float64)
    bank.transition=np.zeros_like(meta[CRF])
    bank.transition[3:,3:]=strength*meta["joint.transition"]
    bank.scale=1.
    return bank,np.concatenate(rows)


def write_tables(bank,x):
    active=~bank.start_mask[1:]
    keys=7*bank.gold[:-1][active]+bank.gold[1:][active]
    pooled=np.bincount(keys,minlength=49).reshape(7,7)+5.
    p=pooled/pooled.sum()
    cuts=np.ascontiguousarray(np.quantile(x[1:][active],np.arange(1,BINS)/BINS,axis=0).T)
    chain=[];same=[];support=[]
    for col,cut in enumerate(cuts):
        q=np.searchsorted(cut,x[1:,col][active])
        count=np.bincount(49*q+keys,minlength=49*BINS).reshape(BINS,7,7)
        support.append(count.sum((1,2)))
        total=count+400*p
        ratio=np.log(total/total.sum((1,2),keepdims=True))-np.log(p)
        interaction=ratio-ratio.mean(1,keepdims=True)-ratio.mean(2,keepdims=True)+ratio.mean((1,2),keepdims=True)
        stay=total.diagonal(axis1=1,axis2=2).sum(1)
        change=total[:,OFF].sum(1)
        baseline=np.log(np.trace(p)/p[OFF].sum())
        chain.append(interaction.clip(-1.6,1.6))
        same.append(np.eye(7)[None,:,:]*np.clip(np.log(stay/change)-baseline,-1.6,1.6)[:,None,None])
    return {"cuts":cuts,"interaction":np.asarray(chain,dtype="f4"),"stay":np.asarray(same,dtype="f4")},np.array(support)


def mapped(x,w):
    return np.column_stack([np.searchsorted(c,x[:,j]) for j,c in enumerate(w["cuts"])])


def edge_values(indices,w,mode,source):
    table=w[mode]
    if source==len(NAMES):
        return np.mean([table[j,indices[:,j]] for j in range(len(NAMES))],axis=0,dtype=np.float32)
    return table[source,indices[:,source]]


def run(bank,edge,scale):
    bank.edge=np.ascontiguousarray(edge*scale,dtype=np.float32)
    return bank.decode(ZERO,ONE)


def evaluate(bank,x,weights,mode,source,scale):
    parent=run(bank,np.zeros_like(bank.edge),0.)
    test=run(bank,edge_values(mapped(x,weights),weights,mode,source),scale)
    return cm_case(bank,test),cm_case(bank,parent),int((test!=parent).sum())


def official_shard(args):
    docs,mode,source,scale=args
    torch.set_num_threads(1)
    model=HierModel(DONOR)
    meta=load_file(str(PARENT))
    vocab=json.loads((EDGE/"vocab.json").read_text())
    bank,x=package(docs,aligned_embeddings(model,docs,"test"),model,meta,vocab,load_decoder())
    return evaluate(bank,x,load_file(str(DEST)),mode,source,scale)


def main():
    torch.set_num_threads(2)
    meta=load_file(str(PARENT))
    vocab=json.loads((EDGE/"vocab.json").read_text())
    model=HierModel(DONOR)
    engine=load_decoder()
    fit=training_documents()
    states=list(embeddings(model,fit[:144],"aligned-source-train"))+list(aligned_embeddings(model,fit[144:],"train"))
    train,x=package(fit,states,model,meta,vocab,engine)
    weights,support=write_tables(train,x)
    print("Wrote edge tables",len(fit),train.units,support.tolist(),flush=True)
    del train,x,states
    groups=sample_whole()
    used={d["id"] for group in groups.values() for d in group}
    dropped=set(json.loads((EDGE/"overlap.json").read_text())["independent_exclusions_for_text_disjoint_check"])
    groups["independent"]=[d for d in independent_validation(used) if d["id"] not in dropped]

    def prepare(group):
        ds=groups[group]
        h=list(aligned_embeddings(model,ds,group))
        x=package(ds,h,model,meta,vocab,engine)
        print("Conditioned edge prepared",group,len(ds),x[0].units,flush=True)
        return x

    bank,ex=prepare("select")
    neutral=np.zeros_like(bank.edge)
    incumbent=run(bank,neutral,0.)
    reference=measure(cm_case(bank,incumbent).sum(0)).tolist()
    grid=[];best=(reference[0],reference[1],("stay",0,0.))
    bins=mapped(ex,weights)
    for mode,source in product(("stay","interaction"),range(len(NAMES)+1)):
        edges=edge_values(bins,weights,mode,source)
        for scale in STRENGTHS:
            ys=run(bank,edges,scale)
            scores=measure(cm_case(bank,ys).sum(0)).tolist()
            choice=(mode,source,scale)
            changed=int((ys!=incumbent).sum())
            grid.append([*choice,*scores,changed])
            if scores[1]>=reference[1]-.004 and changed:
                best=max(best,(scores[0],scores[1],choice))
    mode,source,scale=best[2]
    weights["settings"]=np.asarray([int(mode=="interaction"),source,scale])
    save_file(weights,str(DEST))
    restored=load_file(str(DEST))
    for name,value in weights.items():
        np.testing.assert_array_equal(value,restored[name],err_msg=name)
    weights=restored
    freeze=ROOT/"results/sentence-conditioned-edges-frozen.json"
    save_json(freeze,{"choice":list(best[2]),"features":NAMES,"grid":grid,"select_parent":reference,
                      "select_candidate":list(best[:2]),"weights_sha256":file_hash(DEST),"parent_sha256":file_hash(PARENT),
                      "train_cases":[q["id"] for q in fit],"selection_cases":[q["id"] for q in groups["select"]],
                      "train_bin_support":support.tolist()})
    print("Frozen conditioned edges",best,reference,flush=True)

    out={"freeze_sha256":file_hash(freeze),"choice":list(best[2]),"splits":{}}
    official=json.loads((ROOT/"results/sentence-joint-qa.json").read_text())["scores"]
    for group in ("select","confirm","independent","test"):
        if group=="test":
            docs=[document(k,r) for k,r in gold_cases().items()]
            with ProcessPoolExecutor(4) as pool:
                parts=list(pool.map(official_shard,[(docs[i::4],mode,source,scale) for i in range(4)]))
            index={case["id"]:i for i,case in enumerate(q for shard in range(4) for q in docs[shard::4])}
            inv=[index[q["id"]] for q in docs]
            a,b=(np.concatenate([q[j] for q in parts])[inv] for j in (0,1))
            changed=sum(q[2] for q in parts)
            assert row_summary(b.sum(0))==official
        else:
            if group!="select":
                bank,ex=prepare(group)
            a,b,changed=evaluate(bank,ex,weights,mode,source,scale)
        if group=="select":
            np.testing.assert_allclose(measure(a.sum(0))[:2],best[:2],rtol=0.,atol=1e-12)
        out["splits"][group]={"cases":len(a),"sentences":int(a.sum()),"changed":changed,
                              "candidate":row_summary(a.sum(0)),"parent":row_summary(b.sum(0)),
                              "paired":paired(a,b)}
        save_json(ROOT/"results/sentence-conditioned-edges.json",out)
        print(group,measure(a.sum(0)),measure(b.sum(0)),flush=True)


if __name__=="__main__":
    main()
