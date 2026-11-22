"""Write a segment-level precision score over a filtered rhetorical-role lattice."""
import argparse
import ctypes
import json
import subprocess
from itertools import product

import numpy as np
from safetensors.numpy import save_file

from role_edge_interpolation import (
    EDGE, CRF, DONOR, HierModel, ROOT, bootstrap, document,
    file_hash, full_score, gold_cases, independent_validation, load_file,
    metric, prepare, sample_whole, save_json, torch,
)
from role_edge_posterior_study import PROPOSALS
from role_pair_weight_search import edit_documents, embeddings, prepare_score, transition_scores
from span_posterior_decode import best_path, spans
from segment_span_runtime import learned_scores, span_features

ROLES = 7
TEMPERATURE = .9
MIX = (.0, .35, .7, 1.)
COST = (.02, .07, .12, .18, .24)


def engine():
    source = ROOT / "packed_segment_path.cpp"
    lib = ROOT / "cache" / f"segment-path-{file_hash(source)[:16]}.so"
    if not lib.is_file():
        tmp = lib.with_suffix(".part")
        subprocess.run(["g++", "-std=c++17", "-shared", "-fPIC", "-O3", str(source), "-o", str(tmp)], check=True)
        tmp.replace(lib)
    call = ctypes.CDLL(str(lib)).segment_path
    carray = np.ctypeslib.ndpointer
    call.argtypes = [carray(np.int32, flags="C_CONTIGUOUS"), carray(np.float64, flags="C_CONTIGUOUS"),
                     ctypes.c_size_t, ctypes.c_size_t, carray(np.int8, flags="C_CONTIGUOUS")]
    call.restype = ctypes.c_bool

    def go(intervals, values, n):
        result = np.empty(n, dtype=np.int8)
        values = np.ascontiguousarray(values, dtype=np.float64)
        if not call(intervals, values, len(intervals), n, result):
            raise ValueError("Candidate intervals do not tile the text")
        return result
    return go


def lattice(item, edge, parents, posterior_ends=False):
    packed,support,features,p,base=span_features(
        item.logits,edge,parents,item.token_role_counts.sum(1),PROPOSALS,TEMPERATURE,posterior_ends)
    a,b,k=packed.T
    hits=((item.offsets[b]==item.gold_end[a]) & (k==item.gold_label[a])).astype(np.float64)
    return item,packed,support,features,p,hits,base


def fit(train):
    x = np.concatenate([row[3] for row in train])
    y = np.concatenate([row[5] for row in train])
    m = x.mean(0)
    s = 1/np.maximum(x.std(0),.06)
    z = np.column_stack(((x-m)*s,np.ones(len(x))))
    for penalty in (.08, .35):
        diag = np.r_[np.ones(x.shape[1]),.02]
        gram = z.T @ z
        coef = np.linalg.solve(gram+np.diag(diag*penalty*np.trace(gram)/len(gram)),z.T@y)
        yield penalty,{"center":m,"scale":s,"weights":coef[:-1].copy(),"bias":np.asarray(coef[-1])}


def project(rows, weights):
    return [learned_scores(v[3],weights) for v in rows]


def evaluate(rows, call, probabilities, cost, extended):
    scores = []
    for (item,candidates,support,*_), prob in zip(rows, probabilities):
        keep = slice(None) if extended else support>0
        labels = call(candidates[keep],prob[keep]-cost,len(item.offsets)-1)
        scores.append(item.score(labels))
    return np.stack(scores)


def baseline(rows):
    return np.stack([row[0].score(row[-1]) for row in rows])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posterior-ends",action="store_true")
    args=parser.parse_args()
    experiment="segment-posterior-ends" if args.posterior_ends else "segment-weight"
    torch.set_num_threads(2)
    manifest = json.loads((EDGE/"deployment.json").read_text())
    parents = {label:load_file(str(EDGE/manifest["profiles"][label]["file"])) for label in ("exact","balanced")}
    vocabulary = json.loads((EDGE/"vocab.json").read_text())
    model = HierModel(DONOR)
    groups = sample_whole()
    call = engine()

    def compile(bank, name):
        rows = [lattice(item,edge,parents,args.posterior_ends) for item,edge in bank]
        for item, arcs, support, features, p, gain, base in rows[:2]:
            n = len(base)
            np.testing.assert_array_equal(call(arcs,p-.15,n),best_path(arcs,p-.15,n))
            for a,b,k in spans(base):
                assert np.any(np.all(arcs==[a,b,k],axis=1))
            score = item.score(base)
            assert int(gain[np.isin(np.ravel_multi_index(arcs.T,(n,n+1,7)),
                                    np.ravel_multi_index(spans(base).T,(n,n+1,7)))].sum()) == score[49]
        print("Lattice ready",name,len(rows),sum(len(r[1]) for r in rows),flush=True)
        return rows

    samples=edit_documents()
    train=[]
    for d,state in zip(samples,embeddings(model,samples,"raw-emitter-train")):
        train.append((prepare_score(d,model.logits(state),parents),transition_scores(d["text"],d["units"],vocabulary,parents["exact"])))
    train=compile(train,"train")
    weights=dict(fit(train))
    select=compile(prepare(groups["select"],"select",model,parents,vocabulary,parents["exact"]),"select")
    preceding=metric(baseline(select).sum(0))
    options=[]
    base=[s[4] for s in select]
    best=preceding
    chosen=None
    for ridge,w in weights.items():
        read=project(select,w)
        for extension,share in product((False,True),MIX):
            if not share and ridge!=min(weights):
                continue
            prob=[p+share*(q-p) for p,q in zip(base,read)]
            for cost in COST:
                values=metric(evaluate(select,call,prob,cost,extension).sum(0))
                options.append([ridge,extension,share,cost,*values])
                if values > best and values[1] >= preceding[1]-.012:
                    best=values
                    chosen=[ridge,extension,share,cost]
    freeze=ROOT/"results"/f"{experiment}-frozen.json"
    output=ROOT/"checkpoints/hierarchical"/(
        "whole-segment-posterior-ends-weights.safetensors" if args.posterior_ends else "whole-segment-weights.safetensors")
    physical={}
    if chosen:
        ridge,extended,share,cost=chosen
        pack={**weights[ridge],"extended":np.asarray(extended,dtype=np.int16),
              "share":np.asarray(share),"span_cost":np.asarray(cost),"temperature":np.asarray(TEMPERATURE),
              "proposals":np.asarray(PROPOSALS),
              "posterior_ends":np.asarray(args.posterior_ends,dtype=np.int16)}
        save_file(pack,str(output))
        physical=load_file(str(output))
        assert all(np.array_equal(v,physical[k]) for k,v in pack.items())
    save_json(freeze,{"choice":chosen,"grid":options,"preceding":preceding,"best":best,
        "train_cases":[d["id"] for d in samples],"train_candidates":sum(len(r[1]) for r in train),
        "train_exact_matches":int(sum(r[5].sum() for r in train)),
        "selection_cases":[d["id"] for d in groups["select"]],"temperature":TEMPERATURE,
        "weight_sha256":file_hash(output) if chosen else None})
    print("Selected",chosen,best,"parent",preceding,flush=True)
    del train
    used={d["id"] for items in groups.values() for d in items}
    omitted=set(json.loads((EDGE/"overlap.json").read_text())["independent_exclusions_for_text_disjoint_check"])
    groups["independent"]=[d for d in independent_validation(used) if d["id"] not in omitted]
    report={"choice":chosen,"freeze_sha256":file_hash(freeze),"splits":{},"posterior_ends":args.posterior_ends,
        "method":"Directly write segment hit scores by closed-form ridge regression over exact labeled span indicators from official training judgments. Compare a prior CRF posterior, the learned segment score, and their mixtures. Adjacent roles must differ and intervals must tile the document. A second lattice adds locally recombined boundaries without using input labels. Weights freeze before confirmation and official test. The encoder and CRF remain frozen."}
    for name in ("select","confirm","independent","test"):
        if name=="test":
            groups[name]=[document(k,r) for k,r in gold_cases().items()]
        rows=select if name=="select" else compile(prepare(groups[name],name,model,parents,vocabulary,parents["exact"]),name)
        b=baseline(rows)
        a=b
        if chosen:
            share=float(physical["share"])
            prob=[p+share*(f-p) for (_,_,_,_,p,_,_),f in zip(rows,project(rows,physical))]
            a=evaluate(rows,call,prob,float(physical["span_cost"]),bool(physical["extended"]))
        if name=="select":
            np.testing.assert_allclose(metric(a.sum(0)),best,rtol=0,atol=1e-12)
        oracle={}
        for extension in (False,True):
            all_hits=[r[5] for r in rows]
            optim=evaluate(rows,call,all_hits,.25,extension)
            all_gold=sum(i.truth_run_counts.sum() for i,*_ in rows)
            present=sum(int(((r[2]>0)|extension)@r[5]) for r in rows)
            oracle[str(extension)]={"exact_span_recall_ceiling":float(present/all_gold),
                                    "exact_hit_oracle":full_score(optim)["exact_labeled_spans"]}
        report["splits"][name]={"cases":len(rows),"candidate":full_score(a),"parent":full_score(b),
                               "gap_ci95":bootstrap(a,b),"candidate_oracles":oracle}
        save_json(ROOT/"results"/f"{experiment}-write.json",report)
        print(name,metric(a.sum(0)),metric(b.sum(0)),flush=True)


if __name__=="__main__":
    main()
