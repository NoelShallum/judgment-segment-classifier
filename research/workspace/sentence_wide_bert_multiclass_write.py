"""Write a seven-output residual over the contextual reviewer and document model."""
import argparse
from itertools import product
import json
from typing import NamedTuple

import numpy as np
from safetensors.numpy import load_file, save_file

import sentence_bert_reviewer_gain as gain
import sentence_pairwise_repair as repair
import sentence_phase_readout_write as phase
import sentence_referral_actual_replay as replay
import sentence_referral_utility_write as utility
import sentence_wide_bert_benchmark as wide
from hier_model import LABELS
from sentence_document_graph_write import OUT, cm, file_hash, paired, row_summary, save_json


P = repair.P
B = gain.B
TAG = "sentence-wide-bert-multiclass"
WEIGHTS = OUT / (TAG + ".safetensors")
RATE = .20
EYE = np.eye(7)


class Review(NamedTuple):
    bank: tuple
    base: np.ndarray
    incumbent: np.ndarray
    positions: np.ndarray
    probability: np.ndarray
    prior: np.ndarray
    structure: np.ndarray
    signature: str


def training(context):
    bank,base,old,positions,_,_,_,sig,prob,prior,structure = gain.get_features(
        "fit",context,include_probabilities=True)
    return Review(bank,base,old,positions,prob,prior,structure,sig)


def inputs(split,context,wide_results):
    source = wide_results["splits"][split]
    method = wide_results["method"]
    _,_,_,cfg,policy,dependencies,w = context
    assert method["bert_freeze_sha256"]==dependencies[B.TAG]
    assert method["local_overlay_sha256"]==dependencies[gain.overlay.TAG]
    bank,base,ranks,router_hash = replay.risk(split)
    assert router_hash==method["router_freeze_sha256"]
    union = np.union1d(*(utility.route(bank[1],rank,wide.RATE) for rank in ranks.values()))
    rows = wide.feature_bank(bank,split,union)
    cached,provenance = wide.encode(rows,split,source["embedding_provenance"]["device"],router_hash)
    assert provenance == source["embedding_provenance"]
    mode = cfg["frozen_modes"][gain.overlay.BERT][0]
    scores = B.response(B.design(rows,cached,mode,w),w)
    prior = np.concatenate([r[5] for r in rows])
    structure = np.concatenate([r[4] for r in rows])
    reference = np.concatenate([r[2][r[3]] for r in rows])
    g,t,k = policy["arms"]["BERT only"]["choice"]
    review = gain.overlay.edit(base,union,prior,scores,g,t,reference if k else None)
    positions = utility.route(bank[1],ranks["confidence"],RATE)
    order = np.searchsorted(union,positions)
    np.testing.assert_array_equal(union[order],positions)
    old = base.copy()
    old[positions] = review[positions]
    assert row_summary(cm(bank[2],old)) == source["comparison"]["routes"]["confidence"]["labels"]
    return Review(bank,base,old,positions,scores[order],prior[order],structure[order],provenance["embedding_sha256"])


def evidence(view):
    p, q = view.probability, view.prior
    assert np.all(p>=0) and np.all(q>=0)
    np.testing.assert_allclose(p.sum(1),1.,atol=1e-6)
    np.testing.assert_allclose(q.sum(1),1.,atol=1e-6)
    interaction = (p[:,:,None]*q[:,None,:]).reshape(len(p),49)
    original = EYE[view.base[view.positions]]
    x = np.column_stack((p,q,original,interaction,np.sqrt(p),np.sqrt(q),view.structure))
    return x, .55*p + .45*q


def transform(x,weights):
    return np.column_stack((np.clip((x-weights["center"])*weights["scale"],-5.,5.),np.ones(len(x))))


def writes(view):
    x,mixture = evidence(view)
    base = {"center":x.mean(0), "scale":1/np.maximum(x.std(0),.025)}
    z = transform(x,base)
    gold = view.bank[2][view.positions]
    freq = np.bincount(gold,minlength=7)
    assert min(freq)>0
    target = EYE[gold]-mixture
    for focus in (0.,.35):
        importance = (freq[gold]/freq.mean())**-focus
        importance = importance/importance.mean()
        gram = (z.T*importance)@z
        rhs = (z.T*importance)@target
        for penalty in (.04,.18,.55):
            coefficients = phase.solve(gram,rhs,penalty)
            np.testing.assert_allclose(coefficients.sum(1),0.,atol=1e-6)
            yield (focus,penalty),{**base,"coefficients":np.ascontiguousarray(coefficients)}


def apply(view,mixture,correction,rule):
    gain,threshold,protection,alternative = rule
    loc = view.positions
    old = view.incumbent[loc]
    anchor = view.base[loc]
    row = np.arange(len(loc))
    score = mixture + gain * correction
    if alternative=="two candidates":
        valid = (EYE[anchor]+EYE[view.probability.argmax(1)]+EYE[old]) > 0
        score = np.where(valid,score,-np.inf)
    proposal = score.argmax(1)
    take = score[row,proposal] > score[row,old]+threshold
    if protection:
        take &= old==anchor
    pred = view.incumbent.copy()
    pred[loc[take]] = proposal[take]
    return pred


def diagnostics(view,pred):
    labels = view.bank[2]
    changed = view.incumbent!=pred
    fixed = changed & (pred==labels)
    broken = changed & (view.incumbent==labels)
    original = view.incumbent
    repairs = cm(original[fixed],pred[fixed])
    damage = cm(original[broken],pred[broken])
    roles = {}
    for a,b in ((0,5),(1,2),(3,4)):
        clean = int(repairs[a,b]+repairs[b,a])
        lost = int(damage[a,b]+damage[b,a])
        roles[LABELS[a]+" ↔ "+LABELS[b]]={"fixed":clean,"broken":lost}
    return roles


def choose(context,wide_results):
    train = training(context)
    dev = inputs("select",context,wide_results)
    assert not set(train.bank[0]) & set(dev.bank[0])
    old = row_summary(cm(dev.bank[2],dev.incumbent))
    best = (old["macro_f1"],old["accuracy"],0)
    tensors,pick = None,None
    x,mixture = evidence(dev)
    grid = []
    for params,w in writes(train):
        correction = transform(x,w)@w["coefficients"]
        for rule in product((0.,.3,.6,1.),(.035,.10,.19,.32),(False,True),("two candidates","any role")):
            pred = apply(dev,mixture,correction,rule)
            stats = row_summary(cm(dev.bank[2],pred))
            key = (stats["macro_f1"],stats["accuracy"],-int(np.count_nonzero(dev.incumbent!=pred)))
            grid.append([params,rule,key])
            if stats["accuracy"]>=old["accuracy"]-.001 and key>best:
                best,pick,tensors = key,(params,rule),w
    digest = None
    if tensors is not None:
        save_file(tensors,str(WEIGHTS))
        loaded = load_file(str(WEIGHTS))
        for name,value in tensors.items():
            np.testing.assert_array_equal(loaded[name],value)
        digest = file_hash(WEIGHTS)
    freeze = {"dependencies":context[5],"wide_method":wide_results["method"],"selection":best,
              "rule":pick,"fit_rows":len(train.positions),"fit_cases":train.bank[0].tolist(),
              "development_cases":dev.bank[0].tolist(),"fit_embeddings_sha256":train.signature,
              "gradient_steps":0,"folds":4,"weights_sha256":digest,"grid":grid}
    save_json(repair.receipt(TAG),freeze)
    print("Frozen",pick,best,"baseline",old["macro_f1"],flush=True)
    return freeze,dev


def read_wide():
    return json.loads((P.RESULT/(wide.TAG+".json")).read_text())


def run(action,splits):
    context = gain.plans()
    wide_results = read_wide()
    freeze_file = repair.receipt(TAG)
    if action=="fit":
        freeze,dev = choose(context,wide_results)
    else:
        freeze = json.loads(freeze_file.read_text())
        dev = None
    assert freeze["dependencies"] == context[5] and freeze["wide_method"] == wide_results["method"]
    w = None
    if freeze["weights_sha256"]:
        assert file_hash(WEIGHTS)==freeze["weights_sha256"]
        w = load_file(str(WEIGHTS))
    output = P.RESULT/(TAG+".json")
    result = {"freeze_sha256":file_hash(freeze_file),"splits":{}}
    if action=="evaluate" and output.is_file():
        result = json.loads(output.read_text())
        assert result["freeze_sha256"]==file_hash(freeze_file)
    for split in splits:
        view = dev if split=="select" and dev is not None else inputs(split,context,wide_results)
        assert not set(view.bank[0]) & set(freeze["fit_cases"])
        if split!="select":
            assert not set(view.bank[0]) & set(freeze["development_cases"])
        pred = view.incumbent
        if w is not None:
            x,mixture = evidence(view)
            pred = apply(view,mixture,transform(x,w)@w["coefficients"],freeze["rule"][1])
        new,now,trial = replay.scores(view.bank,view.incumbent,pred,np.flatnonzero(pred!=view.incumbent))
        np.testing.assert_array_equal(new,pred)
        before = P.dense.paired_confusion(view.bank,view.incumbent)
        if split=="select":
            assert list(freeze["selection"])==[trial["labels"]["macro_f1"],trial["labels"]["accuracy"],-trial["changed"]]
        result["splits"][split] = {"cases":len(view.bank[0]),"reviewed":len(view.positions),"candidate":trial,
                                   "parent":row_summary(before.sum(0)),"paired":paired(now,before),
                                   "priority_pairs":diagnostics(view,pred),"embedding_sha256":view.signature}
        save_json(output,result)
        print(split,trial["labels"]["macro_f1"],trial["changed"],trial["fixed"],trial["broken"],flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=("fit","evaluate"))
    parser.add_argument("splits",nargs="*")
    args = parser.parse_args()
    run(args.action,args.splits or ("select","confirm"))
