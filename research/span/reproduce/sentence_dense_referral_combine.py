"""Measure complementary fixes from the dense readout and frozen BERT referrals."""
import argparse
import json

import numpy as np

import sentence_bert_referral_write as bert
import sentence_dense_readout_write as dense
from sentence_document_graph_write import OUT, cm, file_hash, load_file, paired, row_summary, save_json

TAG = "sentence-dense-referral"
EXPERT = "BERT previous two and target"
POLICIES = ("BERT priority","dense priority","protect every BERT referral","agree on referrals")


def views(split):
    bank = dense.prepare(split)
    base = dense.baseline(bank,split)
    br, embeddings, _ = bert.encoded(split,(EXPERT,))
    bert_plan = bert.read(bert.TAG+"-frozen")
    assert bank[0].tolist()==[r[0] for r in br]
    assert np.array_equal(bank[2],np.concatenate([r[1] for r in br]))
    assert np.array_equal(bank[3],np.concatenate([r[2] for r in br]))
    model = OUT/(bert.TAG+"-2.safetensors")
    assert file_hash(model)==bert_plan["checkpoints"][EXPERT]
    w = load_file(str(model))
    variant,*_,gain,cut = bert_plan["frozen_modes"][EXPERT]
    response = bert.response(bert.design(br,embeddings[EXPERT],variant,w),w)
    expert = np.concatenate(bert.apply(br,response,gain,cut))
    assert row_summary(cm(bank[2],expert))==bert.read(bert.TAG)["splits"][split]["arms"][EXPERT]["scores"]
    routed = np.zeros(len(expert),bool)
    for row,start in zip(br,bank[1][:-1]):
        routed[start+row[3]]=True
    return bank,base,expert,routed


def readout(bank,arm,choice):
    p = OUT/(dense.TAG+"-"+str(dense.ARMS.index(arm))+".safetensors")
    assert file_hash(p)==choice["sha256"]
    w = load_file(str(p))
    estimate = dense.output(dense.infer_basis(bank[-1],arm,w),w)
    _,_,_,*action = choice["selection"]
    return dense.replace(bank[3],bank[4],estimate,*action)


def combine(baseline,proposal,expert,routed,rule):
    follow = expert!=baseline
    if rule=="BERT priority":
        return np.where(follow,expert,proposal)
    if rule=="dense priority":
        return np.where(proposal!=baseline,proposal,expert)
    if rule=="protect every BERT referral":
        return np.where(routed,expert,proposal)
    if rule=="agree on referrals":
        return np.where(routed & (proposal!=expert),baseline,proposal)
    assert rule=="BERT alone"
    return expert


def main(splits):
    folder = dense.FOLDER
    plan = dense.read(dense.TAG+"-frozen")
    hashes = {k:file_hash(folder/(k+"-frozen.json")) for k in (dense.TAG,bert.TAG)}
    shape = folder/(TAG+"-frozen.json")
    val,val_parent,val_expert,val_routed = views("select")
    expert,routed = val_expert,val_routed
    base_stats = row_summary(cm(val[2],expert))
    pick = (base_stats["macro_f1"],base_stats["accuracy"],"current scores","BERT alone")
    trials = []
    for arm,options in plan["arms"].items():
        prediction = readout(val,arm,options)
        for rule in POLICIES:
            pred = combine(val[3],prediction,expert,routed,rule)
            metric = row_summary(cm(val[2],pred))
            stats = (metric["macro_f1"],metric["accuracy"],arm,rule)
            if stats[:2]>pick[:2] and metric["accuracy"]>=base_stats["accuracy"]-.001:
                pick = stats
            trials.append([*stats,int(np.count_nonzero(pred!=expert))])
    receipt = {"source_hashes":hashes,"choice":pick[-2:],"selection_cases":val[0].tolist(),
               "selection_macro_f1":pick[0],"trials":trials}
    if shape.exists():
        assert json.loads(shape.read_text())==json.loads(json.dumps(receipt))
    else:
        save_json(shape,receipt)
    arm,rule = pick[-2:]
    output = {"freeze_sha256":file_hash(shape),"splits":{}}
    archive = folder/"known-sentence-numerical-archives.npz"
    source = dense.read("known-sentence-numerical-archives")
    assert file_hash(archive)==source["score_file_sha256"]
    for s in splits:
        bank,base,expert,routed = (val,val_parent,val_expert,val_routed) if s=="select" else views(s)
        prediction = readout(bank,arm,plan["arms"][arm])
        y = combine(bank[3],prediction,expert,routed,rule)
        count = dense.paired_confusion(bank,y)
        before = dense.paired_confusion(bank,expert)
        row = {"cases":len(bank[0]),"candidate":row_summary(count.sum(0)),"bert":row_summary(before.sum(0)),
               "parent":base,"paired_bert":paired(count,before),
               "paired_parent":paired(count,dense.paired_confusion(bank,bank[3])),
               "changed_vs_bert":int(np.count_nonzero(y!=expert))}
        if s=="test":
            with np.load(archive) as data:
                np.testing.assert_array_equal(bank[0],data["cases"])
                author = data[source["best_by_macro_f1"]]
            row["aligned_author"]=row_summary(author.sum(0))
            row["paired_aligned_author"]=paired(count,author)
        output["splits"][s]=row
        print(s,arm,rule,row["candidate"]["macro_f1"],row["candidate"]["accuracy"],row["changed_vs_bert"],flush=True)
    save_json(folder/(TAG+".json"),output)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("splits",nargs="+",choices=("select","confirm","holdout","test"))
    main(ap.parse_args().splits)
