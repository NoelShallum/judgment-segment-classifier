"""Write exact-match and boundary scores over bounded dense legal spans."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import product
import json

import numpy as np
from safetensors.numpy import save_file

from role_edge_interpolation import (
    EDGE, DONOR, HierModel, ROOT, bootstrap, document, file_hash, full_score, gold_cases,
    independent_validation, load_file, metric, prepare, sample_whole, save_json, torch,
)
from hier_full import CACHE
from hier_model import LABELS
from role_pair_weight_search import edit_documents, embeddings, prepare_score, transition_scores
from segment_span_runtime import density_features, density_scores, density_values
from segment_weight_write import engine

MAX_TIER=5


@dataclass
class CaseArcs:
    original: object
    arcs: np.ndarray
    tier: np.ndarray
    features: np.ndarray
    current: np.ndarray


def bank(item,edge,parents,incumbent,tier):
    parts=density_features(item.logits,edge,parents,item.token_role_counts.sum(1),incumbent,tier)
    return CaseArcs(item,*parts)


def targets(d,row):
    l,r,k=row.arcs.T
    item=row.original
    ends=np.full(len(item.offsets),-1,dtype=np.int8)
    positions={x:i for i,x in enumerate(item.offsets)}
    for span in d["gold_spans"]:
        j=positions.get(span["end"])
        if j is not None:
            ends[j]=LABELS.index(span["label"])
    left=item.gold_label[l]==k
    right=ends[r]==k
    exact=left&(item.gold_end[l]==item.offsets[r])
    assert np.all(right[exact])
    return np.column_stack((exact,left,right)).astype("f4")


def learn(docs,rows):
    feat=np.concatenate([r.features for r in rows])
    y=np.concatenate([targets(d,r) for d,r in zip(docs,rows)])
    center=feat.mean(0,dtype=np.float64)
    scale=1/np.maximum(feat.std(0,dtype=np.float64),.04)
    ridge=np.r_[np.ones(len(center)),.04]
    gram=np.zeros((len(ridge),len(ridge)))
    rhs=np.zeros((len(ridge),3))
    for emphasis in (0.,8.):
        gram.fill(0.)
        rhs.fill(0.)
        for i in range(0,len(feat),12000):
            x=feat[i:i+12000]
            v=y[i:i+len(x)]
            z=np.column_stack(((x-center)*scale,np.ones(len(x))))
            importance=1+emphasis*v[:,0]+.65*(x[:,-3]>0)
            weighted=z.T*importance
            gram+=weighted@z
            rhs+=weighted@v
        for damping in (.025,.16):
            w=np.linalg.solve(gram+np.diag(ridge*damping*np.trace(gram)/len(gram)),rhs)
            yield emphasis,damping,{"center":center,"scale":scale,
                "coefficients":np.ascontiguousarray(w[:-1]),"bias":w[-1].copy()}


def scores(row,weights):
    return density_scores(row.features,weights)


def decode_row(row,decoder,incumbent,weights=None,spec=None,prediction=None):
    if spec is None:
        keep=row.tier==0
        utilities=row.current[keep]-float(incumbent["span_cost"])
    else:
        keep=row.tier<=spec[0]
        predicted=prediction if prediction is not None else scores(row,weights)
        utilities=density_values(row.current,row.tier,predicted,spec)[keep]
    labels=decoder(row.arcs[keep],utilities,len(row.original.offsets)-1)
    return row.original.score(labels)


def setup():
    parents={k:load_file(str(EDGE/f"role-edge-{k}.safetensors")) for k in ("exact","balanced")}
    vocab=json.loads((EDGE/"vocab.json").read_text())
    path=ROOT/"checkpoints/hierarchical/whole-segment-weights.safetensors"
    return parents,vocab,path,load_file(str(path))


def replay(args):
    docs,weights,spec=args
    torch.set_num_threads(1)
    parents,vocab,_,incumbent=setup()
    decode=engine()
    ca=[];ba=[];count=hit=0
    tier=int(spec[0]) if spec is not None else 0
    for d in docs:
        with np.load(CACHE/(d["id"]+".npz")) as r:
            item=prepare_score(d,r["raw"],parents)
        edge=transition_scores(d["text"],d["units"],vocab,parents["exact"])
        row=bank(item,edge,parents,incumbent,tier)
        b=decode_row(row,decode,incumbent)
        a=b if spec is None else decode_row(row,decode,incumbent,weights,spec)
        s=row.tier<=tier
        l,r,k=row.arcs[s].T
        count+=int(s.sum())
        hit+=int(((item.gold_label[l]==k)&(item.gold_end[l]==item.offsets[r])).sum())
        ca.append(a)
        ba.append(b)
    return np.stack(ca),np.stack(ba),count,hit


def official(physical):
    cases=[document(k,r) for k,r in gold_cases().items()]
    spec=None if physical is None else physical["settings"]
    with ProcessPoolExecutor(4) as pool:
        parts=list(pool.map(replay,[(cases[i::4],physical,spec) for i in range(4)]))
    candidate=np.concatenate([v[0] for v in parts])
    baseline=np.concatenate([v[1] for v in parts])
    arcs=sum(v[2] for v in parts)
    hits=sum(v[3] for v in parts)
    expect=json.loads((ROOT/"results/segment-weight-qa.json").read_text())["scores"]
    assert full_score(baseline)==expect
    return candidate,baseline,arcs,hits


def run_frozen():
    root=ROOT/"results"
    f=root/"short-span-density-frozen.json"
    report_path=root/"short-span-density-edit.json"
    child=json.loads(report_path.read_text())
    provenance=json.loads(f.read_text())
    assert child["freeze_sha256"]==file_hash(f)
    assert child["choice"]==provenance["choice"]
    assert {"select","confirm","independent"} <= child["splits"].keys()
    physical=None
    if child["choice"]:
        saved=ROOT/"checkpoints/hierarchical/whole-short-span-density.safetensors"
        assert provenance["sha256"]==file_hash(saved)
        physical=load_file(str(saved))
        np.testing.assert_array_equal(physical["settings"],provenance["choice"][2:])
    a,b,arcs,hits=official(physical)
    child["splits"]["test"]={"cases":len(b),"candidate":full_score(a),"parent":full_score(b),
         "gap_ci95":bootstrap(a,b),"candidates":arcs,"covered_gold":hits}
    save_json(report_path,child)
    print("test",metric(a.sum(0)),metric(b.sum(0)),arcs,hits,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-frozen",action="store_true")
    args=parser.parse_args()
    if args.replay_frozen:
        run_frozen()
        return
    torch.set_num_threads(2)
    parents,vocab,path,incumbent=setup()
    engine_=engine()
    model=HierModel(DONOR)

    def compile(bank_):
        return [bank(item,edge,parents,incumbent,MAX_TIER) for item,edge in bank_]

    train=edit_documents()
    states=embeddings(model,train,"raw-emitter-train")
    training=compile([(prepare_score(d,model.logits(x),parents),
                       transition_scores(d["text"],d["units"],vocab,parents["exact"])) for d,x in zip(train,states)])
    weights={(p,r):w for p,r,w in learn(train,training)}
    count=sum(len(r.arcs) for r in training)
    positives=sum(int(targets(d,r)[:,0].sum()) for d,r in zip(train,training))
    print("Dense train",len(train),count,positives,flush=True)
    del training
    groups=sample_whole()
    selection=compile(prepare(groups["select"],"select",model,parents,vocab,parents["exact"]))

    def decode_all(rows,w=None,spec=None,pred=None):
        if pred is None: pred=[None]*len(rows)
        return np.stack([decode_row(r,engine_,incumbent,w,spec,p) for r,p in zip(rows,pred)])

    b=decode_all(selection)
    best=base=metric(b.sum(0))
    grid=[];choice=None
    for kp,w in weights.items():
        predicted=[scores(r,w) for r in selection]
        for spec in product((2,3,4,5),(.45,.85,1.),(0.,.28),(.13,.21,.30,.40),(.0,.065)):
            q=metric(decode_all(selection,w,spec,predicted).sum(0))
            grid.append([*kp,*spec,*q])
            if q>best and q[1]>=base[1]-.015:
                best=q
                choice=(*kp,*spec)
    freeze=ROOT/"results/short-span-density-frozen.json"
    destination=ROOT/"checkpoints/hierarchical/whole-short-span-density.safetensors"
    physical=None
    if choice:
        pack={**weights[choice[:2]],"settings":np.asarray(choice[2:])}
        save_file(pack,str(destination))
        physical=load_file(str(destination))
        assert all(np.array_equal(v,physical[k]) for k,v in pack.items())
    save_json(freeze,{"choice":choice,"grid":grid,"train_cases":[d["id"] for d in train],
        "selection_cases":[d["id"] for d in groups["select"]],"candidates":count,"matches":positives,
        "base":base,"selected":best,"incumbent_sha256":file_hash(path),
        "sha256":file_hash(destination) if choice else None})
    print("Dense choice",choice,best,base,flush=True)
    used={d["id"] for q in groups.values() for d in q}
    excluded=set(json.loads((EDGE/"overlap.json").read_text())["independent_exclusions_for_text_disjoint_check"])
    groups["independent"]=[d for d in independent_validation(used) if d["id"] not in excluded]
    spec=physical["settings"] if choice else None
    doc={"choice":choice,"freeze_sha256":file_hash(freeze),"splits":{},
         "protocol":"Add all raw-unit spans up to a fixed length, limiting each interval to two or three posterior roles. Write three scores by closed-form ridge regression for an exact span, its role starting at the left endpoint, and its role ending at the right endpoint. Compare direct and endpoint-assisted scores against the existing profile. Select all controls on validation before scoring later cohorts."}
    for name in ("select","confirm","independent","test"):
        if name!="test":
            rows=selection if name=="select" else compile(prepare(groups[name],name,model,parents,vocab,parents["exact"]))
            baseline=decode_all(rows)
            candidate=baseline if spec is None else decode_all(rows,physical,spec)
            used_tier=int(spec[0]) if choice else 0
            arcs=sum(int((r.tier<=used_tier).sum()) for r in rows)
            hits=sum(int(targets(d,r)[r.tier<=used_tier,0].sum()) for d,r in zip(groups[name],rows))
        else:
            candidate,baseline,arcs,hits=official(physical)
        if name=="select":
            np.testing.assert_allclose(metric(candidate.sum(0)),best,atol=1e-12,rtol=0)
        doc["splits"][name]={"cases":len(baseline),"candidate":full_score(candidate),"parent":full_score(baseline),
            "gap_ci95":bootstrap(candidate,baseline),"candidates":arcs,"covered_gold":hits}
        save_json(ROOT/"results/short-span-density-edit.json",doc)
        print(name,metric(candidate.sum(0)),metric(baseline.sum(0)),arcs,hits,flush=True)


if __name__=="__main__":
    main()
