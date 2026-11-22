"""Let adjacent short units share a merged-text emission in a frozen seven-role decoder."""
import hashlib
import json
from itertools import product
import re

import numpy as np
from safetensors.numpy import load_file,save_file

from role_edge_interpolation import (EDGE, CRF, DONOR, ROOT, HierModel, bootstrap,
    document, file_hash, full_score, gold_cases, independent_validation, metric,
    sample_whole,save_json,torch)
from discourse_span_edit import text_features
from hier_full import CACHE
from role_cues import cue_features
from role_edge_runtime import transition_scores,decode
from run_entry import prepare_score

RESULT=ROOT/"results"
CHECKPOINT=ROOT/"checkpoints/hierarchical/optional-merge.safetensors"
REGEX=re.compile(r"\S+")
GRID=[None,*product((.25,.50,.75),(0.,.20,.45))]


def pairs(text, bounds):
    lengths=[len(REGEX.findall(text[a:b])) for a,b in bounds]
    return [i for i in range(len(lengths)-1) if min(lengths[i:i+2])<=10 and sum(lengths[i:i+2])<=40]


def features(model, pieces, p):
    return model.logits(model.contextualize(model.words(pieces),independent=True))+cue_features(pieces)@p["cue_head"]


def package(d, model, p, raw, vocab):
    text=d["text"]
    offsets=d["units"]
    cue,position=text_features(d)
    logits=raw+cue@p["cue_head"]+position@p["position_head"]
    edge=transition_scores(text,offsets,vocab,p)
    indices=pairs(text,offsets)
    evidence=np.zeros_like(logits)
    if indices:
        excerpts=[]
        for i in indices:
            a,b=offsets[i]
            z=offsets[i+1][1]
            excerpts.extend((text[a:b],text[b:z],text[a:z]))
        a,b,whole=features(model,excerpts,p).reshape(-1,3,7).transpose(1,0,2)
        change=whole-(a+b)*.5
        evidence[indices]=change-change.mean(1,keepdims=True)
    return prepare_score(d,raw,{}),logits,edge,indices,evidence


def span_paths(logits, steps, transition, linked, deltas, coefficient, cost):
    n=len(logits)
    roles=np.arange(7)
    scores=np.full((n+1,7),-np.inf)
    scores[0]=transition[1,3:]
    back=np.empty((n+1,7),dtype=np.int8)
    interval=np.ones((n+1,7),dtype=np.int8)
    for i,u in enumerate(logits):
        if i==0:
            current=scores[i]
            origin=roles
        else:
            state=scores[i,:,None]+steps[i-1]
            origin=state.argmax(0)
            current=state[origin,roles]
        separate=current+u
        better=separate>scores[i+1]
        scores[i+1,better]=separate[better]
        back[i+1,better]=origin[better]
        interval[i+1,better]=1
        if linked[i]:
            paired=separate+steps[i].diagonal()+logits[i+1]+coefficient*deltas[i]-cost
            better=paired>scores[i+2]
            scores[i+2,better]=paired[better]
            back[i+2,better]=origin[better]
            interval[i+2,better]=2
    y=np.empty(n,dtype=np.int8)
    role=int((scores[n]+transition[3:,2]).argmax())
    used=0
    while n:
        width=int(interval[n,role])
        y[n-width:n]=role
        used+=width==2
        role=int(back[n,role])
        n-=width
    return y,used


def evaluate(bank,parent,spec):
    output=[]
    used=0
    for scorer,raw,edge,indices,evidence in bank:
        gain=float(parent["edge.gain"])
        if spec is None:
            result=decode(raw,parent[CRF],edge,gain)
        else:
            link=np.zeros(len(raw),bool)
            link[indices]=1
            scores=parent[CRF][3:,3:]+gain*edge
            result,n=span_paths(raw,scores,parent[CRF],link,evidence,*spec)
            used+=n
        output.append(scorer.score(result))
    return np.stack(output),used


def main():
    torch.set_num_threads(2)
    parent=load_file(str(EDGE/"role-edge-exact.safetensors"))
    vocab=json.loads((EDGE/"vocab.json").read_text())
    model=HierModel(DONOR)
    groups=sample_whole()
    def make_bank(split):
        output=[]
        for i,d in enumerate(groups[split],1):
            if split=="test":
                with np.load(CACHE/(d["id"]+".npz")) as fp:raw=fp["raw"]
            elif split=="independent":
                sig=hashlib.sha256(d["text"].encode()).hexdigest()
                raw=np.load(ROOT/"astra-run-penalty/validation-cache"/(sig+".npy"))
            else:
                raw=model.logits(model.contextualize(model.words([d["text"][a:b] for a,b in d["units"]])))
            output.append(package(d,model,parent,raw,vocab))
            if i%72==0 or i==len(groups[split]):
                print("Encoded merged windows",split,i,flush=True)
        return output

    selection=make_bank("select")
    choice=None
    grid=[]
    base=best=metric(evaluate(selection,parent,None)[0].sum(0))
    for opt in GRID:
        values,count=evaluate(selection,parent,opt)
        q=metric(values.sum(0))
        grid.append([opt,*q,count])
        if q>best:
            best=q
            choice=opt
    freeze=RESULT/"optional-merge-frozen.json"
    save_json(freeze,{"choice":choice,"grid":grid,"selection_ids":[d["id"] for d in groups["select"]],
                      "parent_sha256":file_hash(EDGE/"role-edge-exact.safetensors")})
    if choice:
        save_file({"merge.policy":np.asarray(choice,np.float32)},str(CHECKPOINT))
    print("Selected",choice,best,"parent",base,flush=True)
    used={x["id"] for g in groups.values() for x in g}
    excluded=set(json.loads((EDGE/"overlap.json").read_text())["independent_exclusions_for_text_disjoint_check"])
    groups["independent"]=[d for d in independent_validation(used) if d["id"] not in excluded]
    report={"selected":choice,"selection_sha256":file_hash(freeze),
            "method":"Two adjacent short units can share a role using a score adjustment from their joint embedding versus separate embeddings. The baseline path remains eligible. Use the fixed donor, parameters frozen on development, no gradient steps.","splits":{}}
    for split in ("select","confirm","independent","test"):
        if split=="test":
            groups["test"]=[document(k,r) for k,r in gold_cases().items()]
        bank=selection if split=="select" else make_bank(split)
        q,opened=evaluate(bank,parent,choice)
        z,_=evaluate(bank,parent,None)
        report["splits"][split]={"candidate":full_score(q),"parent":full_score(z),"cases":len(z),
                                 "options":sum(len(x[3]) for x in bank),"joint_windows":opened,"gap_ci95":bootstrap(q,z)}
        save_json(RESULT/"optional-merge-lattice.json",report)
        print(split,metric(q.sum(0)),metric(z.sum(0)),"joint windows",opened,flush=True)


if __name__=="__main__":
    main()
