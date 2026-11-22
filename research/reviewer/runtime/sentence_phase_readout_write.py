"""Write phase-conditioned role readouts over frozen judgment representations."""
import argparse
from itertools import product
import json

import numpy as np
from safetensors.numpy import load_file,save_file
from scipy.linalg import cho_factor,cho_solve

import sentence_pairwise_consensus as consensus
import sentence_pairwise_repair as repair
import sentence_phrase_sequence as seq
from sentence_document_graph_write import OUT, cm, file_hash, measure, paired, row_summary, save_json

P=repair.P
TAG="sentence-phase-readout"
SPLITS=("select","confirm","holdout","phrase-prospect","test")
MODES=("global","phase","phase and confidence")


def gates(bank):
    where=np.concatenate([(np.arange(b-a)+.5)/(b-a) for a,b in zip(bank[1],bank[1][1:])])
    centers=np.array([.03,.25,.55,.82,.98])
    phase=np.exp(-.5*((where[:,None]-centers)/.16)**2)
    phase/=phase.sum(1,keepdims=True)
    posterior=np.sort(bank[4],axis=1)
    uncertainty=np.clip((.42-(posterior[:,-1]-posterior[:,-2]))/.40,0.,1.)
    context=np.column_stack((phase*(1-uncertainty[:,None]),phase*uncertainty[:,None]))
    return {"global":np.ones((len(where),1),np.float32),"phase":phase.astype(np.float32),
            "phase and confidence":context.astype(np.float32)}


def basis(bank,lex,w):
    x=repair.dense_basis(bank,lex)
    x=np.clip((x-w["center"])*w["scale"],-5.,5.)
    return np.column_stack((x,np.ones(len(x),np.float32)))


def solve(gram,rhs,regularization):
    diagonal=np.r_[np.ones(len(gram)-1),.02]*gram.trace()/len(gram)
    gram=(gram+gram.T)*.5+np.diag(diagonal*regularization)
    coef=cho_solve(cho_factor(gram),rhs)
    np.testing.assert_allclose(gram@coef,rhs,rtol=1.e-7,atol=1.e-6)
    return coef


def fit_weights(train,lex):
    raw=repair.dense_basis(train,lex)
    w={"center":raw.mean(0,dtype=np.float64).astype(np.float32),
       "scale":(1/np.maximum(raw.std(0,dtype=np.float64),.025)).astype(np.float32)}
    del raw
    x=basis(train,lex,w)
    gold=train[2]
    hot=np.eye(7,dtype=np.float32)[gold]
    rhs=np.column_stack((hot,hot-train[4]))
    freqs=np.maximum(np.bincount(gold,minlength=7)/len(gold),.016)
    certainty=train[4][np.arange(len(gold)),gold]
    g=gates(train)
    for rare in (0.,.35):
        priority=(freqs[gold]**-rare*(1+.75*(1-certainty)**2)).astype(np.float32)
        for mode, gate in g.items():
            grams, links=[],[]
            for j in range(gate.shape[1]):
                xt=x*(priority*gate[:,j])[:,None]
                grams.append((x.T@xt).astype(np.float64))
                links.append((xt.T@rhs).astype(np.float64))
            for penalty in (.04,.20):
                coefs=np.stack([solve(a,b,penalty) for a,b in zip(grams,links)])
                yield mode,rare,penalty,{**w,"weights":np.ascontiguousarray(coefs,dtype=np.float32)}


def score(bank,lex,w,mode):
    x=basis(bank,lex,w)
    gate=gates(bank)[mode]
    out=np.zeros((len(x),14),np.float32)
    for j,matrix in enumerate(w["weights"]):
        out+=gate[:,j,None]*(x@matrix)
    return out


def view(split,parent,weights,recipe):
    bank=P.dense.prepare(split)
    lex=P.cached_scores(split)
    base,unary,_=repair.incumbent(bank,lex,recipe)["sequence only"]
    mode,scale,top,threshold,guard=parent["arms"]["sequence only"]["choice"]
    est=repair.score_pairs(repair.dense_basis(bank,lex),weights,mode)
    p=repair.proposals(est,base,unary,scale,top)
    incumbent=repair.decide(p,base,np.zeros(len(base),bool),threshold,guard)
    assert row_summary(cm(bank[2],incumbent))==P.dense.read(repair.TAG)["splits"][split]["arms"]["sequence only"]["labels"]
    return bank,lex,incumbent,unary


def edit(bank,current,unary,output,params):
    target,weight,threshold,guard=params
    i=np.arange(len(current))
    u=unary.copy()
    u[i,current]=np.maximum(u[i,current],u.max(1)+.005)
    if target=="posterior":
        q=seq.phrase.dense.positive(output[:,:7])
        adjusted=u+weight*np.log(np.maximum(q,.015))
    else:
        adjusted=u+weight*np.clip(output[:,7:],-.60,.60)
    pred=adjusted.argmax(1)
    accept=(adjusted[i,pred]-adjusted[i,current]>=threshold)
    if guard=="keep issue":
        accept&=(pred==2)==(current==2)
    return np.where(accept,pred,current).astype(np.int8)


def fit():
    parent,pair_w,pair_sha=consensus.inputs()
    recipe=repair.load_recipe()
    train=P.dense.prepare("fit")
    lex=P.cached_scores("fit")
    dev,val_lex,base,unary=view("select",parent,pair_w,recipe)
    assert not set(train[0]) & set(dev[0])
    m,a,*_=measure(cm(dev[2],base))
    initial=[float(m),float(a),0]
    best={k:{"settings":None,"selection":initial} for k in MODES}
    grid={k:[] for k in MODES}
    for mode,rare,damp,w in fit_weights(train,lex):
        response=score(dev,val_lex,w,mode)
        for cfg in product(("posterior","residual"),(.22,.50,.85,1.3,2.),(.02,.075,.16,.27),("all","keep issue")):
            prediction=edit(dev,base,unary,response,cfg)
            m,a,*_=measure(cm(dev[2],prediction))
            stat=[float(m),float(a),-int(np.count_nonzero(prediction!=base))]
            grid[mode].append([[rare,damp,*cfg],stat])
            if stat>best[mode]["selection"] and a>=initial[1]-.0015:
                path=OUT/(TAG+f"-{MODES.index(mode)}.safetensors")
                save_file(w,str(path))
                q=load_file(str(path))
                assert all(np.array_equal(v,q[k]) for k,v in w.items())
                best[mode]={"settings":[rare,damp,*cfg],"selection":stat,"sha256":file_hash(path)}
        print(mode,rare,damp,best[mode],flush=True)
    receipt={"repair_freeze_sha256":pair_sha,"fit_cases":train[0].tolist(),"select_cases":dev[0].tolist(),
             "training_rows":len(train[2]),"gradient_steps":0,"baseline":initial,"arms":best,"grid":grid}
    save_json(repair.receipt(TAG),receipt)


def evaluate(splits):
    parent,pair_w,pair_sha=consensus.inputs()
    plan=repair.receipt(TAG)
    freeze=json.loads(plan.read_text())
    assert freeze["repair_freeze_sha256"]==pair_sha
    recipe=repair.load_recipe()
    report={"freeze_sha256":file_hash(plan),"splits":{}}
    for split in splits:
        bank,lex,base,unary=view(split,parent,pair_w,recipe)
        assert not set(freeze["fit_cases"]) & set(bank[0])
        if split!="select":
            assert not set(freeze["select_cases"]) & set(bank[0])
        old=P.dense.paired_confusion(bank,base)
        spans=seq.span_tally(bank,base)
        q=report["splits"][split]={"cases":len(bank[0]),"baseline":row_summary(old.sum(0)),
                                  "baseline_spans":seq.span_info(spans),"arms":{}}
        for j,(mode,cfg) in enumerate(freeze["arms"].items()):
            pred=base
            if cfg["settings"]:
                path=OUT/(TAG+f"-{j}.safetensors")
                assert file_hash(path)==cfg["sha256"]
                estimated=score(bank,lex,load_file(str(path)),mode)
                pred=edit(bank,base,unary,estimated,cfg["settings"][2:])
            mat=P.dense.paired_confusion(bank,pred)
            pspan=seq.span_tally(bank,pred)
            n=int(np.count_nonzero(pred!=base))
            rec=q["arms"][mode]={"labels":row_summary(mat.sum(0)),"paired":paired(mat,old),"changed":n,
                                 "spans":seq.span_info(pspan),"paired_spans":seq.span_intervals(pspan,spans)}
            if split=="select":
                m,a,*_=measure(mat.sum(0))
                assert [float(m),float(a),-n]==cfg["selection"]
        save_json(P.RESULT/(TAG+".json"),report)
        print(split,{k:(r["labels"]["macro_f1"],r["labels"]["accuracy"],r["changed"]) for k,r in q["arms"].items()},flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("action",choices=("fit","evaluate"))
    cli.add_argument("splits",nargs="*")
    args=cli.parse_args()
    if args.action=="fit":
        fit()
    else:
        evaluate(args.splits or SPLITS)
