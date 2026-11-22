"""Write a multiscale kernel readout over frozen contextual text features."""
import argparse
from itertools import product
import json

import numpy as np
from safetensors.numpy import load_file,save_file
from scipy.linalg import cho_factor,cho_solve

import audit_pairwise_referral_coverage as routing
import sentence_pairwise_consensus as consensus
import sentence_pairwise_repair as pair
from sentence_document_graph_write import OUT, cm, file_hash, measure, paired, row_summary, save_json

P = pair.P
TAG = "sentence-kernel-weight-write"
SOURCE = P.RESULT / (TAG + "-frozen.json")
MODEL = OUT / (TAG + ".safetensors")
SPLITS = ("select","confirm","holdout","phrase-prospect","test")


def features(x, w):
    z = np.clip((x-w["mean"])*w["scale"],-4.,4.)
    theta = z @ w["projection"]
    return np.column_stack((.35*z,np.cos(theta),np.sin(theta),np.ones(len(x),np.float32)))


def train(bank, lex):
    x = pair.dense_basis(bank,lex)
    n = x.shape[1]
    rng = np.random.default_rng(980173)
    w = {"mean":x.mean(0,dtype=np.float64).astype(np.float32),
         "scale":(1/np.maximum(x.std(0,dtype=np.float64),.025)).astype(np.float32),
         "projection":(rng.standard_normal((n,384))*np.repeat([.50,1.,1.8],128)/np.sqrt(n)).astype(np.float32)}
    xx = features(x,w)
    del x
    tags = bank[2]
    hot = np.eye(7,dtype=np.float32)[tags]
    anchor = bank[4]
    output = np.column_stack((hot, hot-anchor))
    by_role = np.maximum(np.bincount(tags,minlength=7)/len(tags),.015)
    n_features = xx.shape[1]
    for rarity in (0.,.25):
        gram = np.zeros((n_features,n_features),np.float64)
        rhs = np.zeros((n_features,14),np.float64)
        sample = (by_role[tags]**-rarity).astype(np.float32)
        for lo in range(0,len(tags),8192):
            x = xx[lo:lo+8192]
            wgt = x*sample[lo:lo+8192,None]
            gram += x.T@wgt
            rhs += wgt.T@output[lo:lo+8192]
        gram = (gram+gram.T)*.5
        diag = np.r_[np.ones(len(gram)-1),.02]*gram.trace()/len(gram)
        for value in (.005,.035,.16):
            linear = gram+np.diag(value*diag)
            coeff = cho_solve(cho_factor(linear),rhs)
            np.testing.assert_allclose(linear@coeff,rhs,rtol=1.e-7,atol=1.e-6)
            yield rarity,value,w,np.ascontiguousarray(coeff,dtype=np.float32)


def views(split, parent, weights, recipe):
    bank = P.dense.prepare(split)
    lex = P.cached_scores(split)
    incumbent, score = routing.routing(bank,lex,recipe,weights,parent["arms"]["sequence only"]["choice"])
    ref=P.dense.read(pair.TAG)["splits"][split]["arms"]["sequence only"]["labels"]
    assert ref==row_summary(cm(bank[2],incumbent))
    return bank,incumbent,score["confidence"],pair.dense_basis(bank,lex)


def route(bank, score, fraction):
    if fraction == 1.:
        return np.ones(len(score),bool)
    flags=np.zeros(len(score),bool)
    for a,b in zip(bank[1],bank[1][1:]):
        idx=np.argsort(-score[a:b],kind="stable")[:int(np.ceil(fraction*(b-a)))]
        flags[a+idx]=True
    return flags


def decide(bank, incumbent, estimated, mode, gain, margin, flags):
    row=np.arange(len(incumbent))
    anchor=.64*bank[4]+.36*np.eye(7,dtype=np.float32)[incumbent]
    if mode=="posterior":
        scores=(1-gain)*anchor+gain*P.dense.positive(estimated[:,:7])
    else:
        scores=anchor+gain*np.clip(estimated[:,7:],-.65,.65)
    p=scores.argmax(1)
    edit=(scores[row,p]-scores[row,incumbent]>=margin)&flags
    return np.where(edit,p,incumbent)


def fit():
    origin,pair_weights,pair_sha=consensus.inputs()
    recipe=pair.load_recipe()
    train_bank=P.dense.prepare("fit")
    train_lex=P.cached_scores("fit")
    valid,base,scores,x=views("select",origin,pair_weights,recipe)
    assert not set(train_bank[0]) & set(valid[0])
    flags={f:route(valid,scores,f) for f in (.10,.20,.33,1.)}
    before=measure(cm(valid[2],base))
    best=(float(before[0]),float(before[1]),0)
    choice=None
    grid=[]
    for r,d,w,coeff in train(train_bank,train_lex):
        response=features(x,w)@coeff
        for mode,gain,margin,fraction in product(("posterior","residual"),(.25,.50,.75,1.),(.02,.075,.15),flags):
            pred=decide(valid,base,response,mode,gain,margin,flags[fraction])
            m,a,*_=measure(cm(valid[2],pred))
            outcome=(float(m),float(a),-int(np.count_nonzero(base!=pred)))
            setting=(r,d,mode,gain,margin,fraction)
            grid.append((setting,outcome))
            if outcome>best and a>=before[1]-.001:
                best,choice=outcome,setting
                tensors={**w,"readout":coeff}
                save_file(tensors,str(MODEL))
                checked=load_file(str(MODEL))
                assert all(np.array_equal(checked[k],v) for k,v in tensors.items())
        print(r,d,best,choice,flush=True)
    record={"pairwise_freeze_sha256":pair_sha,"train_cases":train_bank[0].tolist(),
            "select_cases":valid[0].tolist(),"training_rows":len(train_bank[2]),
            "gradient_steps":0,"settings":choice,"selection":best,
            "checkpoint_sha256":file_hash(MODEL) if choice else None,"grid":grid}
    save_json(SOURCE,record)


def evaluate(splits):
    parent,w,source=consensus.inputs()
    frozen=json.loads(SOURCE.read_text())
    assert frozen["pairwise_freeze_sha256"]==source
    recipe=pair.load_recipe()
    option=frozen["settings"]
    patch=None
    if option:
        assert file_hash(MODEL)==frozen["checkpoint_sha256"]
        patch=load_file(str(MODEL))
    report={"freeze_sha256":file_hash(SOURCE),"splits":{}}
    for split in splits:
        bank,base,conf,x=views(split,parent,w,recipe)
        assert not set(frozen["train_cases"]) & set(bank[0])
        if split!="select":
            assert not set(frozen["select_cases"]) & set(bank[0])
        pred=base
        if option:
            _,_,mode,g,margin,f=option
            eligible=route(bank,conf,f)
            pred=decide(bank,base,features(x,patch)@patch["readout"],mode,g,margin,eligible)
            np.testing.assert_array_equal(pred[~eligible],base[~eligible])
        b,c=(P.dense.paired_confusion(bank,v) for v in (base,pred))
        rec={"cases":len(bank[0]),"baseline":row_summary(b.sum(0)),"edited":row_summary(c.sum(0)),
             "paired":paired(c,b),"changed":int(np.count_nonzero(pred!=base))}
        if split=="select":
            # The selector and replay compare the same metric implementation.
            m,a,*_=measure(c.sum(0))
            assert [float(m),float(a),-rec["changed"]]==frozen["selection"]
        report["splits"][split]=rec
        save_json(P.RESULT/(TAG+".json"),report)
        print(split,rec["edited"]["macro_f1"],rec["edited"]["accuracy"],rec["changed"],flush=True)


if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("action",choices=("fit","evaluate"))
    a.add_argument("splits",nargs="*")
    x=a.parse_args()
    if x.action=="fit":
        fit()
    else:
        evaluate(x.splits or SPLITS)
