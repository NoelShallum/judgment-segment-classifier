"""Directly reweight lexical, neighboring and structural transition evidence."""
import json

import numpy as np
from safetensors.numpy import save_file

from role_edge_interpolation import (EDGE, CRF, DONOR, ROOT, HierModel, bootstrap, decode,
    document, file_hash, full_score, gold_cases, independent_validation, load_file, metric,
    prepare, sample_whole, save_json, torch)
from role_edge_runtime import signatures

FAMILY={"r1":0,"r2":1,"r3":1,"l1":2,"l2":2,"lr":2,"pun":3,"zone":3}
CONTROL=(1.,1.,1.,1.)
CHOICES=(CONTROL, (1.,.5,1.,1.), (1.,1.5,1.,1.), (.5,1.5,1.,1.),
         (1.5,.5,1.,1.), (1.,1.,.5,1.5), (1.,1.,1.5,.5), (1.5,1.5,.5,1.),
         (.5,.5,1.5,1.), (1.,1.,0.,1.), (1.,1.,1.,0.), (.75,.75,.75,.75),
         (1.25,1.25,1.25,1.25))


def collect(docs, raw, vocab, mem, lookup):
    results=[]
    for d,(item,_) in zip(docs,raw):
        index=np.asarray([[vocab.get(k,0) for k in row] for row in signatures(d["text"],d["units"])],dtype=np.int32)
        if not len(index):
            values=np.zeros((0,4,7,7),np.float32)
        else:
            block=mem[index]
            values=np.stack([block[:,lookup==g].sum(1) for g in range(4)],axis=1)
        results.append((item,values))
    return results


def score(bank, profile, f, parent):
    weights=np.asarray(f,np.float32)
    return np.stack([item.score(decode(item.logits[profile],parent[CRF],
                                     (phi*weights[None,:,None,None]).sum(1),.2)) for item,phi in bank])


def main():
    torch.set_num_threads(2)
    deployed=json.loads((EDGE/"deployment.json").read_text())
    parents={k:load_file(str(EDGE/v["file"])) for k,v in deployed["profiles"].items()}
    vocab=json.loads((EDGE/"vocab.json").read_text())
    mem=parents["exact"]["edge.log_ratio"]
    d=sample_whole()
    first=next(strings for doc in d["select"] for strings in signatures(doc["text"],doc["units"]))
    lookup=np.asarray([FAMILY[k.split(":",1)[0]] for k in first])
    model=HierModel(DONOR)

    def bank(split):
        base=prepare(d[split],split,model,parents,vocab,parents["exact"])
        return collect(d[split],base,vocab,mem,lookup)

    selection=bank("select")
    choice={}
    metrics={}
    for k,p in parents.items():
        m={f:metric(score(selection,k,f,p).sum(0)) for f in CHOICES}
        floor=m[CONTROL][1]
        choice[k]=max((f for f in CHOICES if k=="exact" or m[f][1]>=floor),key=lambda f:(m[f],f==CONTROL))
        metrics[k]=[[*f,*v] for f,v in m.items()]
        print(k,choice[k],m[choice[k]],m[CONTROL],flush=True)
    freeze=ROOT/"results/role-edge-feature-mix-frozen.json"
    save_json(freeze,{"selected":choice,"grid":metrics,"feature_to_family":FAMILY,"ids":[x["id"] for x in d["select"]]})
    hashes={}
    for k,p in parents.items():
        scale=np.ones(len(mem),np.float32)
        for key,index in vocab.items():
            scale[index]=choice[k][FAMILY[key.split(":",1)[0]]]
        out=ROOT/"checkpoints/hierarchical"/f"whole-role-edge-feature-mix-{k}.safetensors"
        save_file({**p,"edge.log_ratio":mem*scale[:,None,None]},str(out))
        hashes[k]=file_hash(out)
    used={x["id"] for v in d.values() for x in v}
    excluded=set(json.loads((EDGE/"overlap.json").read_text())["independent_exclusions_for_text_disjoint_check"])
    d["independent"]=[x for x in independent_validation(used) if x["id"] not in excluded]
    result={"method":"Weighted four groups of feature rows in the existing role-pair table. The existing transition runtime can load the written tensors without modifications.",
            "weights_sha256":hashes,"selection_sha256":file_hash(freeze),"selected":choice,"splits":{}}
    for split in ("select","confirm","independent","test"):
        if split=="test":
            d[split]=[document(k,r) for k,r in gold_cases().items()]
        data=selection if split=="select" else bank(split)
        records={}
        for k,p in parents.items():
            q=score(data,k,choice[k],p)
            z=score(data,k,CONTROL,p)
            records[k]={"cases":len(data),"candidate":full_score(q),"parent":full_score(z),
                        "gap_ci95":bootstrap(q,z)}
            print(split,k,metric(q.sum(0)),metric(z.sum(0)),flush=True)
        result["splits"][split]=records
        save_json(ROOT/"results/role-edge-feature-mix.json",result)


if __name__=="__main__":
    main()
