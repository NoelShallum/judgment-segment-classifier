"""Write a gate from cross-fitted InLegalBERT corrections to supplied-sentence labels."""
import argparse
from hashlib import sha256
from itertools import product
import json

import numpy as np
from safetensors.numpy import load_file,save_file

import mixedbread_referral_overlay as overlay
import sentence_minilm_whitened_write as mini
import sentence_pairwise_consensus as consensus
import sentence_pairwise_repair as repair
import sentence_phase_readout_write as phase
from sentence_document_graph_write import OUT,cm,file_hash,paired,row_summary,save_json

P=repair.P
B=overlay.B
TAG="sentence-bert-reviewer-gain"
CHECKPOINT=OUT/(TAG+".safetensors")
SPLITS=("select","confirm","holdout","test")


def plans():
    parent,pair_w,p_sha=consensus.inputs()
    bpath=repair.receipt(B.TAG)
    cfg=json.loads(bpath.read_text())
    frozen=repair.receipt(overlay.TAG)
    policy=json.loads(frozen.read_text())
    dep={repair.TAG:p_sha,B.TAG:file_hash(bpath),overlay.TAG:file_hash(frozen)}
    assert all(policy["sources"][k]==dep[k] for k in (repair.TAG,B.TAG))
    path=OUT/(B.TAG+"-2.safetensors")
    assert cfg["checkpoints"][overlay.BERT]==file_hash(path)
    tensors=load_file(str(path))
    return parent,pair_w,repair.load_recipe(),cfg,policy,dep,tensors


def train_base(parent,pair_w,recipe):
    docs=B.docs_for_fit()
    bank=mini.build(docs,"reviewer-gain-fit")
    lex=mini.independent_lexical(docs)
    p,u,_=repair.incumbent(bank,lex,recipe)["sequence only"]
    mode,scale,top,cutoff,guard=parent["arms"]["sequence only"]["choice"]
    scores=repair.score_pairs(repair.dense_basis(bank,lex),pair_w,mode)
    c=repair.proposals(scores,p,u,scale,top)
    return bank,repair.decide(c,p,np.zeros(len(p),bool),cutoff,guard)


def get_features(split,context,*,include_probabilities=False):
    parent,pair_w,recipe,cfg,policy,_,tensors=context
    rows,vectors,receipt=B.encoded(split,(overlay.BERT,))
    if split=="fit":
        bank,base=train_base(parent,pair_w,recipe)
    else:
        bank,_,base,_=phase.view(split,parent,pair_w,recipe)
    np.testing.assert_array_equal(bank[0],[r[0] for r in rows])
    np.testing.assert_array_equal(bank[2],np.concatenate([r[1] for r in rows]))
    locations=np.concatenate([q+a for a,(_,_,_,q,_,_,_) in zip(bank[1],rows)])
    structure=np.concatenate([r[4] for r in rows])
    prior=bank[4][locations]
    original=np.concatenate([r[2][r[3]] for r in rows])
    mode,rare,damp,*_=cfg["frozen_modes"][overlay.BERT]
    raw=B.design(rows,vectors[overlay.BERT],mode,tensors)
    if split!="fit":
        prob=B.response(raw,tensors)
    else:
        split_ids=np.concatenate([np.repeat(int.from_bytes(sha256(("bert-review-v1|"+str(r[0])).encode()).digest()[:2],"big")%4,len(r[3])) for r in rows])
        truth=bank[2][locations]
        prob=np.empty((len(locations),7),np.float64)
        for key in range(4):
            include=split_ids!=key
            weights=next(patch for value,patch in B.fit(raw[include],truth[include],rare) if value==damp)
            prob[~include]=B.response(raw[~include],weights)
    rule=policy["arms"]["BERT only"]["choice"]
    pre=np.concatenate([r[5] for r in rows])
    p0=overlay.edit(base,locations,pre,prob,*rule[:2],original if rule[2] else None)
    if split!="fit":
        assert row_summary(cm(bank[2],p0))==P.dense.read(overlay.TAG)["splits"][split]["arms"]["BERT only"]["labels"]
    novelty=prob.argmax(1)
    old=base[locations]
    i=np.arange(len(old))
    margin=prob[i,novelty]-prob[i,old]
    features=np.column_stack((structure,prob,prior,np.eye(7)[old],np.eye(7)[novelty],margin,
                              prior[i,novelty]-prior[i,old],
                              (old==original).astype(float),(novelty==original).astype(float)))
    out=bank,base,p0,locations,novelty,margin,features,receipt[overlay.BERT]["features_sha256"]
    if include_probabilities:
        return (*out,prob,pre,structure)
    return out


def targets(bank,current,locations,alternates):
    old=current[locations]
    truth=bank[2][locations]
    freq=cm(bank[2],current)
    tp=freq.diagonal()
    total=freq.sum(0)+freq.sum(1)
    plus=2*(tp+1)/(total+1)-2*tp/total
    minus=2*tp/(total-1)-2*tp/total
    # Approximate the difference between giving the correction and undoing it.
    good=((alternates==truth)&(old!=truth))*(plus[alternates]+minus[old])
    bad=((alternates!=truth)&(old==truth))*(plus[old]+minus[alternates])
    change=(alternates==truth).astype(np.float64)-(old==truth)
    raw=good-bad
    typical=np.mean(np.abs(raw[raw!=0]))
    return np.column_stack((change,np.clip(raw/typical,-4.,4.)))


def predict(x,w):
    return ((x-w["center"])*w["scale"])@w["weights"]+w["bias"]


def operations(view,risk,params):
    bank,base,p0,loc,alt,margin,*_=view
    benefit,margin_min,cut,guard=params
    score=(1-benefit)*risk[:,0]+benefit*risk[:,1]
    current=base[loc]
    revise=(margin>=margin_min)&(score>cut)&(alt!=current)
    candidate=base.copy()
    candidate[loc[revise]]=alt[revise]
    if guard:
        changed=p0!=base
        candidate[changed]=p0[changed]
    return candidate


def fit():
    context=plans()
    train=get_features("fit",context)
    dev=get_features("select",context)
    bank,base,old_base,loc,alt,_,x,hash_train=train
    valid,_,incumbent,_,_,_,z,_=dev
    assert not set(bank[0]) & set(valid[0])
    gold=valid[2]
    original=row_summary(cm(gold,incumbent))
    best=(original["macro_f1"],original["accuracy"],0)
    setting=None
    y=targets(bank,base,loc,alt)
    center=x.mean(0)
    scale=1/np.maximum(x.std(0),.025)
    q=np.column_stack(((x-center)*scale,np.ones(len(x))))
    grid=[]
    for focus in (0.,.30,.65):
        relevance=(np.abs(y[:,0])+np.abs(y[:,1]))**focus if focus else np.ones(len(q))
        relevance=.4+.6*relevance
        gram=q.T@(q*relevance[:,None])
        rhs=q.T@(y*relevance[:,None])
        for damping in (.025,.12,.40):
            coeff=phase.solve(gram,rhs,damping)
            w={"center":center,"scale":scale,"weights":coeff[:-1].copy(),"bias":coeff[-1].copy()}
            est=predict(z,w)
            for cfg in product((0.,.40,.85),(.025,.075,.16,.27),(-.02,.02,.06,.14,.28),(False,True)):
                out=operations(dev,est,cfg)
                summary=row_summary(cm(gold,out))
                grade=(summary["macro_f1"],summary["accuracy"],-int(np.count_nonzero(out!=incumbent)))
                grid.append([[focus,damping,*cfg],grade])
                if grade>best and grade[1]>=original["accuracy"]-.001:
                    save_file(w,str(CHECKPOINT))
                    assert all(np.array_equal(v,load_file(str(CHECKPOINT))[k]) for k,v in w.items())
                    best,setting=grade,(focus,damping,*cfg)
    save_json(repair.receipt(TAG),{"source_sha256":context[5],"fit_cases":bank[0].tolist(),"selection_cases":valid[0].tolist(),
            "train_referrals":len(loc),"feature_train_sha256":hash_train,"crossfit_folds":4,"gradient_steps":0,
            "settings":setting,"selection":best,"weights_sha256":file_hash(CHECKPOINT) if setting else None,"grid":grid})
    print("Frozen",setting,best,flush=True)


def evaluate(splits):
    context=plans()
    freeze=repair.receipt(TAG)
    config=json.loads(freeze.read_text())
    assert config["source_sha256"]==context[5]
    model=None
    if config["settings"]:
        assert config["weights_sha256"]==file_hash(CHECKPOINT)
        model=load_file(str(CHECKPOINT))
    out={"freeze_sha256":file_hash(freeze),"splits":{}}
    for split in splits:
        view=get_features(split,context)
        bank,base,incumbent,loc,alt,_,x,_=view
        assert not set(config["fit_cases"]) & set(bank[0])
        if split!="select":
            assert not set(config["selection_cases"]) & set(bank[0])
        updated=incumbent if model is None else operations(view,predict(x,model),config["settings"][2:])
        assert np.isin(np.flatnonzero(updated!=base),loc).all()
        prior,new=(P.dense.paired_confusion(bank,v) for v in (incumbent,updated))
        rec={"cases":len(bank[0]),"referred":len(loc),"baseline":row_summary(prior.sum(0)),"scores":row_summary(new.sum(0)),
             "paired":paired(new,prior),"changed":int(np.count_nonzero(updated!=incumbent))}
        if split=="select":
            assert [rec["scores"]["macro_f1"],rec["scores"]["accuracy"],-rec["changed"]]==config["selection"]
        out["splits"][split]=rec
        save_json(P.RESULT/(TAG+".json"),out)
        print(split,rec["baseline"]["macro_f1"],rec["scores"]["macro_f1"],rec["paired"]["macro_f1"],rec["changed"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("fit","evaluate"))
    p.add_argument("splits",nargs="*")
    args=p.parse_args()
    if args.action=="fit":
        fit()
    else:
        evaluate(args.splits or SPLITS)
