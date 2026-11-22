"""Write a readout that estimates each sentence's value for a second opinion."""
import argparse
from itertools import product
import json

import numpy as np
from safetensors.numpy import load_file,save_file

import audit_pairwise_referral_coverage as coverage
import sentence_minilm_whitened_write as mini
import sentence_pairwise_consensus as consensus
import sentence_pairwise_repair as repair
import sentence_phase_readout_write as phase
from sentence_document_graph_write import OUT,cm,file_hash,row_summary,save_json

P=repair.P
TAG="sentence-referral-utility"
MODEL=OUT/(TAG+".safetensors")
CONTROL=(1.,0.,0.)
BUDGETS=(.05,.10,.20,.33)
SPLITS=("select","confirm","holdout","test")


def data(split,source,weights,recipe):
    if split=="fit":
        doc=mini.docs(split)
        bank=mini.build(doc,split)
        lex=mini.independent_lexical(doc)
    else:
        bank=P.dense.prepare(split)
        lex=P.cached_scores(split)
    pred,rank=coverage.routing(bank,lex,recipe,weights,source["arms"]["sequence only"]["choice"])
    if split!="fit":
        assert row_summary(cm(bank[2],pred))==P.dense.read(repair.TAG)["splits"][split]["arms"]["sequence only"]["labels"]
    x=repair.dense_basis(bank,lex)
    x=np.column_stack((x,rank["confidence"],rank["margin"],rank["pair dissent"],np.eye(7,dtype=np.float32)[pred]))
    return bank,pred,rank["confidence"],x.astype(np.float32)


def nonlinear(x,w):
    z=np.clip((x-w["center"])*w["scale"],-4.,4.)
    return np.column_stack((z,np.tanh(z@w["projection"]+w["bias"]),np.ones(len(z),np.float32)))


def fit_weights(bank,pred,x,confidence):
    rng=np.random.default_rng(713150)
    width=x.shape[1]
    w={"center":x.mean(0,dtype=np.float64).astype(np.float32),
       "scale":(1/np.maximum(x.std(0,dtype=np.float64),.025)).astype(np.float32),
       "projection":(rng.standard_normal((width,96))/np.sqrt(width)).astype(np.float32),
       "bias":rng.uniform(-.6,.6,96).astype(np.float32),
       "confidence_scale":np.array([max(confidence.std(),.025)],np.float32)}
    z=nonlinear(x,w)
    gold=bank[2]
    conf=cm(gold,pred)
    tp=conf.diagonal()
    den=conf.sum(0)+conf.sum(1)
    gain=(2*(tp+1)/(den+1)-2*tp/den)[gold]+(2*tp/(den-1)-2*tp/den)[pred]
    err=gold!=pred
    gain*=err
    scaled=(gain/max(gain[err].mean(),1.e-9)).clip(0,4)
    y=np.column_stack((err,scaled)).astype(np.float32)
    gram=(z.T@z).astype(np.float64)
    link=(z.T@y).astype(np.float64)
    for strength in (.025,.12,.42):
        beta=phase.solve(gram,link,strength)
        v=z@beta
        scale=np.maximum(v.std(0),.03)
        yield strength,{**w,"matrix":np.ascontiguousarray(beta,dtype=np.float32),"calibration":scale.astype(np.float32)}


def route(off,risk,fraction):
    return np.concatenate([np.argsort(-risk[a:b],kind="stable")[:int(np.ceil(fraction*(b-a)))]+a
                           for a,b in zip(off,off[1:])])


def choices(x,conf,w):
    v=(nonlinear(x,w)@w["matrix"])/w["calibration"]
    normalized=conf/w["confidence_scale"][0]
    return np.column_stack((normalized,v))


def audit(bank,pred,score,rates):
    return {f"{f:.0%}":coverage.ceiling(bank,pred,route(bank[1],score,f)) for f in rates}


def fit():
    prior,pair_w,p_sha=consensus.inputs()
    recipe=repair.load_recipe()
    train,p,train_conf,x=data("fit",prior,pair_w,recipe)
    dev,incumbent,old,u=data("select",prior,pair_w,recipe)
    assert not set(train[0]) & set(dev[0])
    assert not set(train[0]) & set(prior["fit_cases"])
    parent=audit(dev,incumbent,old,(.20,))["20%"]
    best=(parent["ceiling"]["macro_f1"],parent["errors_reachable"])
    pick=(None,CONTROL)
    grid=[]
    for penalty,w in fit_weights(train,p,x,train_conf):
        v=choices(u,old,w)
        for local, f1share in product((.30,.60,1.),(0.,.35,.70,1.)):
            mixture=(1-local,local*(1-f1share),local*f1share)
            risk=v@np.array(mixture)
            c=audit(dev,incumbent,risk,(.20,))["20%"]
            stats=(c["ceiling"]["macro_f1"],c["errors_reachable"])
            grid.append([penalty,mixture,stats])
            if stats>best:
                save_file(w,str(MODEL))
                check=load_file(str(MODEL))
                assert all(np.array_equal(v0,check[k]) for k,v0 in w.items())
                best,pick=stats,(penalty,mixture)
        print(penalty,best,pick,flush=True)
    save_json(repair.receipt(TAG),{"pairwise_freeze_sha256":p_sha,"train_cases":train[0].tolist(),
              "fit_rows":len(p),"selection_cases":dev[0].tolist(),"gradient_steps":0,"selection_budget":.20,
              "pick":pick,"validation_oracle":best,"grid":grid,"patch_sha256":file_hash(MODEL) if pick[0] is not None else None})


def evaluate(splits):
    src=repair.receipt(TAG)
    plan=json.loads(src.read_text())
    prior,pair_w,p_sha=consensus.inputs()
    assert plan["pairwise_freeze_sha256"]==p_sha
    recipe=repair.load_recipe()
    w=None
    if plan["patch_sha256"]:
        assert file_hash(MODEL)==plan["patch_sha256"]
        w=load_file(str(MODEL))
    evidence={"freeze_sha256":file_hash(src),"splits":{}}
    for split in splits:
        bank,pred,conf,x=data(split,prior,pair_w,recipe)
        assert not set(plan["train_cases"]) & set(bank[0])
        if split!="select":
            assert not set(plan["selection_cases"]) & set(bank[0])
        scores=conf if w is None else choices(x,conf,w)@np.array(plan["pick"][1])
        baseline=audit(bank,pred,conf,BUDGETS)
        edited=audit(bank,pred,scores,BUDGETS)
        if split=="select":
            q=edited["20%"]
            assert [q["ceiling"]["macro_f1"],q["errors_reachable"]]==plan["validation_oracle"]
        for key in baseline:
            assert baseline[key]["referred"]==edited[key]["referred"]
        q=evidence["splits"][split]={"cases":len(bank[0]),"baseline":baseline,"edited":edited}
        save_json(P.RESULT/(TAG+".json"),evidence)
        print(split,{z:{k:(v["errors_reachable"],v["ceiling"]["macro_f1"]) for k,v in r.items()}
                     for z,r in (("baseline",baseline),("edited",edited))},flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("fit","evaluate"))
    p.add_argument("splits",nargs="*")
    a=p.parse_args()
    if a.action=="fit":
        fit()
    else:
        evaluate(a.splits or SPLITS)
