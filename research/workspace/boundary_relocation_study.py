"""Move an existing predicted role transition to an optional internal offset."""
from itertools import product
import hashlib
import json

import numpy as np
from safetensors.numpy import load_file,save_file

from optional_boundary_study import (EDGE, PATCH, RESULT, CRF, DONOR, ROOT, HierModel,
    bootstrap, document, file_hash, full_score, gold_cases, independent_validation,
    metric, package, sample_whole, save_json, torch)
from hier_full import CACHE
from optional_boundary_runtime import decode_blocks

GRID=[None,*product((.4,.8,1.2),(.25,.50),(.2,.7,1.3))]


def best_ordered(options):
    state=np.array([0.])
    positions=np.array([0])
    pointers=[]
    for variants in options:
        offsets,values=variants.T
        score=np.where(offsets[None,:]>positions[:,None],state[:,None]+values[None,:],-np.inf)
        back=score.argmax(0)
        state=score[back,np.arange(len(values))]
        positions=offsets
        pointers.append(back)
    selected=[]
    i=int(state.argmax())
    for variants,back in zip(reversed(options),reversed(pointers)):
        selected.append(int(variants[i,0]))
        i=int(back[i])
    return selected[::-1]


def prepare_role_moves(bank,parent):
    bureau=[]
    for scorer,em,edge,options,coords,_,_ in bank:
        parent_path=decode_blocks(em,edge,parent[CRF],[[] for _ in options])
        roles=np.asarray([r for _,r,_ in parent_path],dtype=np.int8)
        boundaries=np.r_[np.flatnonzero(roles[1:]!=roles[:-1])+1]
        names=np.r_[roles[0],roles[boundaries]]
        choices=[]
        lowers=[]
        for right in boundaries:
            left=right-1
            a,b=roles[[left,right]]
            old=coords[right][0]
            record=[]
            for i in (left,right):
                for prob,ratio,diff,lex,idx in options[i]:
                    mass=(1-ratio)*(em[i,b]-em[i,a]) if i==left else ratio*(em[i,a]-em[i,b])
                    local=diff[a]-diff[b]
                    record.append([idx,mass+.2*lex[a,b],prob,local])
            choices.append(np.asarray(record).reshape(-1,4))
            lowers.append(old)
        bureau.append((scorer,names,lowers,choices))
    return bureau


def score(bureau,spec):
    metric_rows=[]
    total_correct_position=total_moves=0
    for scorer,roles,original,choices in bureau:
        moved=original
        if spec is not None and len(original):
            alpha,prior,cost=spec
            opts=[]
            for old,alts in zip(original,choices):
                risk=alts[:,1]+prior*alts[:,2]+alpha*alts[:,3]-cost
                opts.append(np.vstack(([old,0.],np.column_stack((alts[:,0],risk)))))
            moved=best_ordered(opts)
        y=np.repeat(roles,np.diff([0,*moved,len(scorer.offsets)-1]))
        metric_rows.append(scorer.score(y))
        swaps=[b for a,b in zip(original,moved) if a!=b]
        total_moves+=len(swaps)
        total_correct_position+=sum(scorer.gold_label[x]>=0 for x in swaps)
    return np.stack(metric_rows),total_moves,int(total_correct_position)


def main():
    torch.set_num_threads(2)
    p=load_file(str(EDGE/"role-edge-exact.safetensors"))
    prior=load_file(str(PATCH/"prior.safetensors"))
    cut_vocab=json.loads((PATCH/"vocab.json").read_text())
    lex=json.loads((EDGE/"vocab.json").read_text())
    model=HierModel(DONOR)
    groups=sample_whole()
    def bank_docs(group):
        encoded=[]
        for i,d in enumerate(groups[group],1):
            if group=="test":
                with np.load(CACHE/(d["id"]+".npz")) as f:raw=f["raw"]
            elif group=="independent":
                sig=hashlib.sha256(d["text"].encode()).hexdigest()
                raw=np.load(ROOT/"astra-run-penalty/validation-cache"/(sig+".npy"))
            else:
                raw=model.logits(model.contextualize(model.words([d["text"][a:b] for a,b in d["units"]])))
            encoded.append(package(d,model,p,raw,cut_vocab,prior,lex))
            if i%72==0 or i==len(groups[group]):
                print("Scored relocation features",group,i,flush=True)
        return prepare_role_moves(encoded,p)
    selection=bank_docs("select")
    best=base=metric(score(selection,None)[0].sum(0))
    chosen=None
    grid=[]
    for cfg in GRID:
        q,n,correct=score(selection,cfg)
        m=metric(q.sum(0))
        grid.append([cfg,*m,n,correct])
        if m>best:
            best=m
            chosen=cfg
    freeze=RESULT/"boundary-relocation-frozen.json"
    save_json(freeze,{"chosen":chosen,"grid":grid,"case_ids":[d["id"] for d in groups["select"]]})
    if chosen:
        save_file({**prior,"boundary.relocation":np.asarray(chosen,np.float32)},str(PATCH/"relocation.safetensors"))
    print("Relocation selection",chosen,best,"base",base,flush=True)
    out={"method":"Move an existing predicted role transition up to one original unit left or right. Local left/right frozen evidence, written prior and existing lexical edge score govern the move. Ordered dynamic programming preserves the number and ordering of predicted role runs. No gradient steps.",
         "choice":chosen,"freeze_sha256":file_hash(freeze),"splits":{}}
    scope=("select","confirm","independent","test") if chosen else ("select",)
    used={d["id"] for ls in groups.values() for d in ls}
    excluded=set(json.loads((EDGE/"overlap.json").read_text())["independent_exclusions_for_text_disjoint_check"])
    if chosen:
        groups["independent"]=[d for d in independent_validation(used) if d["id"] not in excluded]
    for s in scope:
        if s=="test":
            groups[s]=[document(k,v) for k,v in gold_cases().items()]
        bank=selection if s=="select" else bank_docs(s)
        q,n,h=score(bank,chosen)
        z,_,_=score(bank,None)
        out["splits"][s]={"cases":len(q),"candidate":full_score(q),"parent":full_score(z),"moves":n,
                          "moves_to_gold_change":h,"gap_ci95":bootstrap(q,z)}
        save_json(RESULT/"boundary-relocation.json",out)
        print(s,metric(q.sum(0)),metric(z.sum(0)),n,h,flush=True)


if __name__=="__main__":
    main()
