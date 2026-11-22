"""Improve selected role confusions without moving text boundaries."""
from itertools import product

import numpy as np
from safetensors.numpy import load_file, save_file
from scipy.linalg import solve
import torch

from discourse_span_edit import text_features
from hier_full import CACHE, gold_cases
from hier_model import CRF, LABELS, HierModel, path_labels
from hier_trial import DONOR, OUT
from preserved_emitter_edit import edit_documents, embeddings
from program_trial import save_json
from raw_emitter_write import paired
from seven_paths import ROOT
from span_eval import counts, document, summarize
from transition_span_edit import sample_whole
from weight_edit import file_hash

PAIRS = ((0,3), (1,5))


def build(doc, logits, parent):
    x, location = text_features(doc)
    evidence = logits + x @ parent["cue_head"] + location @ parent["position_head"]
    y = path_labels(evidence, parent[CRF])
    begins = np.r_[0, np.flatnonzero(y[1:] != y[:-1]) + 1]
    ends = np.r_[begins[1:], len(y)]
    roles = y[begins]
    spans = [(doc["units"][a][0], doc["units"][b-1][1]) for a,b in zip(begins,ends)]
    gold = {(s["start"],s["end"]): LABELS.index(s["label"]) for s in doc["gold_spans"]}
    by_span = []
    for a,b,(l,r) in zip(begins,ends,spans):
        seq = evidence[a:b]
        previous = np.eye(8)[y[a-1] if a else 7]
        following = np.eye(8)[y[b] if b < len(y) else 7]
        by_span.append(np.r_[seq.mean(0),seq[0],seq[-1], x[a:b].mean(0),x[a],x[b-1],
                             np.log1p(b-a), np.log1p(r-l), l/len(doc["text"]), r/len(doc["text"]),
                             previous,following,1.])
    return doc, (begins, ends, roles, np.stack(by_span), np.array([gold.get(s,-1) for s in spans]))


def fit(banks, penalty):
    result = {}
    for current, revised in PAIRS:
        keep, target, rare_errors = [], [], []
        for _,(_,_,roles,x,y) in banks:
            keep.extend(x[(roles == current) & (y == current)])
            target.extend(x[y == revised])
            rare_errors.extend(x[(roles == current) & (y == revised)])
        a = np.asarray(keep)
        b = np.asarray(target + 3 * rare_errors)
        if len(b) < 10:
            raise ValueError(f"Insufficient observations of {current} → {revised}: {len(b)}")
        pos = b.mean(0)
        neg = a.mean(0)
        u,v = a-neg,b-pos
        cov = .5*(u.T@u/len(u)+v.T@v/len(v))
        diag = np.maximum(np.diag(cov), .01*cov.trace()/len(cov))
        diag[-1] = 1.
        w = solve(cov + penalty*np.diag(diag), pos-neg, assume_a="pos")
        w[-1] -= (.5*(pos+neg))@w
        sigma = max(.001, (w @ cov @ w)**.5)
        result[current] = (revised, w/sigma, len(a), len(target), len(rare_errors))
    return result


def relabel(data, weights, threshold):
    begins, ends, original, x, _ = data
    changed = original.copy()
    for a,(b,w,*_) in weights.items():
        if np.isinf(threshold):
            continue
        mask = (original == a) & (x@w > threshold)
        left = np.r_[7, changed[:-1]]
        right = np.r_[changed[1:], 7]
        changed[mask & (left != b) & (right != b)] = b
    return np.repeat(changed, ends-begins)


def score(items, weights, threshold):
    return np.stack([counts(d,d["units"],relabel(v,weights,threshold)) for d,v in items])


def main():
    torch.set_num_threads(2)
    model = HierModel(DONOR)
    parent = load_file(str(OUT/"whole-discourse-exact.safetensors"))
    docs = edit_documents()
    train = [build(d,model.logits(h),parent) for d,h in zip(docs,embeddings(model,docs,"raw-emitter-train"))]
    dev = {k:[build(d,model.logits(h),parent) for d,h in zip(v,embeddings(model,v,"raw-boundary-write-"+k))]
           for k,v in sample_whole().items()}
    reference = {k:score(v,{},np.inf) for k,v in dev.items()}
    def rank(rows):
        v = summarize(rows.sum(0))
        return v["exact_labeled_spans"]["f1"],v["token_accuracy"]
    best = rank(reference["select"])
    preferred = {}
    selected = [None,None,None]
    grid = []
    for damp in (.4, 2.):
        weights = fit(train,damp)
        print("Fit",damp,{str(k):v[2:] for k,v in weights.items()},flush=True)
        for group,thr in product(("petitioner","reasoning","both"), (0.,.75,1.5,2.5)):
            subset = weights if group=="both" else {0 if group=="petitioner" else 1:weights[0 if group=="petitioner" else 1]}
            metric = rank(score(dev["select"],subset,thr))
            grid.append([damp,group,thr,*metric])
            if metric > best:
                best = metric
                preferred = subset
                selected = [damp,group,thr]
    dest = OUT / "whole-discourse-relabel.safetensors"
    params = {**parent,"relabel.pairs":np.asarray([[a,b] for a,(b,*_) in preferred.items()],np.int16).reshape(-1,2),
              "relabel.threshold":np.asarray(selected[2] or 0.,np.float32)}
    for a,(_,w,*_) in preferred.items():
        params[f"relabel.class_{a}"] = w.astype("f4")
    save_file(params,str(dest))
    record = {"selected":selected,"grid":grid,"weight_sha256":file_hash(dest),
              "counts":{str(a):v[2:] for a,v in preferred.items()}}
    freeze = ROOT / "results/role-relabel-frozen.json"
    save_json(freeze,record)
    print("Selection fixed",selected,flush=True)
    physical = load_file(str(dest))
    preferred = {int(a):(int(b),physical[f"relabel.class_{a}"]) for a,b in physical["relabel.pairs"]}
    threshold = float(physical["relabel.threshold"])
    test = []
    for case,rows in gold_cases().items():
        d = document(case,rows)
        with np.load(CACHE/(case+".npz")) as npz:
            test.append(build(d,npz["raw"],parent))
    dev["test"] = test
    reference["test"] = score(test,{},np.inf)
    scores = {}
    for k,v in dev.items():
        a,b = score(v,preferred,threshold),reference[k]
        scores[k] = {"cases":len(a),"candidate":summarize(a.sum(0)),"parent":summarize(b.sum(0)),"gap_ci95":paired(a,b)}
        print(k,rank(a),rank(b),flush=True)
    save_json(ROOT/"results/role-relabel-write.json",{**record,"scores":scores,"freeze_sha256":file_hash(freeze),
              "method":"A regularized linear discriminant is written from train cases where parent span geometry is exact. Gold target anchors are included regardless of which role the parent chose, with extra weight for their rare target confusions. It can relabel None as petitioner argument or facts as reasoning, with prediction-time thresholds and no merging into adjacent spans. No gradients."})


if __name__ == "__main__":
    main()
