"""Write stationary and context-conditioned phrase decoder weights in closed form."""
import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file,save_file

import sentence_phrase_readout as phrase
from role_pair_weight_search import load_decoder
from sentence_document_graph_write import OUT,cm,file_hash,measure,paired,row_summary,save_json

TAG="sentence-phrase-sequence"
ARMS=("stationary pair","learned boundary","learned role pair")
SPLITS=("select","confirm","holdout","prospective","role-guard","phrase-prospect","test")
N=7
OFF=~np.eye(N,dtype=bool)
LEVEL=1


def rows(bank):
    begin=np.zeros(len(bank[2]),bool)
    begin[bank[1][:-1]]=True
    return np.flatnonzero(~begin)


def pair_features(bank,lex,index):
    phrase_score=lex[:,LEVEL*14:LEVEL*14+14]
    x=bank[-1]
    a,b=bank[4][index-1],bank[4][index]
    l,r=phrase_score[index-1],phrase_score[index]
    current=bank[3]
    phase=np.column_stack((x[index,31:33],x[index,-1]))
    agreement=np.column_stack(((a*b).sum(1),np.max(a,axis=1),np.max(b,axis=1),
                               current[index-1]==current[index]))
    return np.column_stack((a,b,np.abs(a-b),a*b,l,r,np.abs(l-r),phase,agreement)).astype(np.float32)


def train():
    bank=phrase.dense.prepare("fit")
    lex=phrase.cached_scores("fit")
    idx=rows(bank)
    left=bank[2][idx-1]
    right=bank[2][idx]
    gold=7*left+right
    counts=np.bincount(gold,minlength=49).reshape(7,7).astype(np.float32)+24
    prior=counts/counts.sum()
    x=pair_features(bank,lex,idx)
    center=x.mean(0,dtype=np.float64).astype(np.float32)
    scale=(1./np.maximum(x.std(0,dtype=np.float64),.025)).astype(np.float32)
    z=np.column_stack(((x-center)*scale,np.ones(len(x),dtype=np.float32)))
    y=np.column_stack((np.eye(49,dtype=np.float32)[gold],left!=right)).astype(np.float32)
    gram=z.T@z
    penalty=np.r_[np.ones(z.shape[1]-1),.02]*gram.trace()/len(gram)
    w=np.linalg.solve(gram+np.diag(.08*penalty),z.T@y)
    output={"mean":center,"scale":scale,"weight":np.ascontiguousarray(w[:-1],dtype=np.float32),
            "bias":np.ascontiguousarray(w[-1],dtype=np.float32),"prior":np.ascontiguousarray(prior)}
    return output,int(len(gold)),int(np.count_nonzero(left!=right)),bank[0].tolist()


def association(p):
    p=p/np.maximum(p.sum(axis=(-2,-1),keepdims=True),1.e-12)
    interesting=np.log(np.maximum(p,1.e-9))
    independent=np.log(np.maximum(p.sum(axis=-1,keepdims=True),1.e-9))+np.log(np.maximum(p.sum(axis=-2,keepdims=True),1.e-9))
    return np.clip(interesting-independent,-2.5,2.5)


def edge_arrays(bank,lex,mode,w):
    n=len(bank[2])
    idx=rows(bank)
    prior=w["prior"]
    stationary=association(prior)
    e=np.zeros((n,7,7),dtype=np.float32)
    if mode=="stationary pair":
        e[idx]=stationary
        return e
    h=(pair_features(bank,lex,idx)-w["mean"])*w["scale"]
    q=h@w["weight"]+w["bias"]
    if mode=="learned boundary":
        observed=float(prior[OFF].sum())
        change=np.clip(q[:,-1],.02,.98)
        strength=np.clip(np.log(change/(1-change))-np.log(observed/(1-observed)),-2.5,2.5)
        e[idx]=stationary[None,:,:]+strength[:,None,None]*OFF
    else:
        prob=np.maximum(q[:,:49],0.).reshape(-1,7,7)
        prob/=np.maximum(prob.sum(axis=(1,2),keepdims=True),1.e-8)
        e[idx]=association(.8*prob+.2*prior)
    return e


def source_unary(bank,lex):
    f=phrase.RESULT/(phrase.TAG+"-frozen.json")
    src=json.loads(f.read_text())["arms"]["target phrases"]
    path=OUT/(phrase.TAG+"-1.safetensors")
    assert file_hash(path)==src["weights_sha256"]
    opt=src["choice"]
    output=phrase.dense.output(phrase.design(bank,lex,"target phrases",opt[0]),load_file(str(path)))
    base=phrase.predict_rows(bank,output,opt)
    if opt[3]=="residual":
        score=bank[4]+opt[4]*np.clip(output[:,7:],-.55,.55)
    else:
        score=(1-opt[4])*bank[4]+opt[4]*phrase.dense.positive(output[:,:7])
    # Preserve the existing deterministic tie and margin handling in the zero-edge control.
    unary=np.log(np.maximum(score,.002))
    u=np.arange(len(base))
    unary[u,base]=np.maximum(unary[u,base],unary.max(1)+.005)
    return base,np.ascontiguousarray(unary,dtype=np.float64)


def decode(engine,offset,unary,base,edge,strength,stay,guard):
    x=unary.copy()
    if guard:
        allowed=(base[:,None]==5)==(np.arange(7)[None,:]==5)
        x[~allowed]=-1.e8
    matrix=np.zeros((10,10),np.float32)
    matrix[3:,3:]=np.eye(7,dtype=np.float32)*stay
    scales=np.full((7,7),strength,np.float32)
    p=np.empty(len(base),np.int8)
    engine(x,edge,offset,len(offset)-1,matrix,scales,p)
    return p


def runs(offset,labels):
    b=np.empty(len(labels),dtype=bool)
    b[0]=True
    b[1:]=labels[1:]!=labels[:-1]
    b[offset[:-1]]=True
    a=np.flatnonzero(b)
    e=np.r_[a[1:],len(labels)]
    return a,e,labels[a]


def shape_metrics(offset,gold,pred):
    gs,ge,gr=runs(offset,gold)
    ps,pe,pr=runs(offset,pred)
    same=np.intersect1d(gs,ps,assume_unique=True,return_indices=True)
    _,a,b=same
    keep=(ge[a]==pe[b])&(gr[a]==pr[b])
    return np.array([np.count_nonzero(keep),len(ps),len(gs),len(a)-len(offset)+1,
                     len(ps)-len(offset)+1,len(gs)-len(offset)+1],dtype=np.int64)


def f1(a):
    return 2*a[0]/(a[1]+a[2]) if a[1]+a[2] else 1.


def span_tally(bank,y):
    return np.stack([shape_metrics(np.array([0,b-a]),bank[2][a:b],y[a:b]) for a,b in zip(bank[1],bank[1][1:])])


def span_info(tally):
    numbers=tally.sum(0)
    return {"exact_labeled_span_f1":f1(numbers[:3]),"role_boundary_f1":f1(numbers[3:]),
            "matched_spans":int(numbers[0]),"predicted_spans":int(numbers[1]),"gold_spans":int(numbers[2])}


def span_intervals(a,b):
    rng=np.random.default_rng(55231)
    out=[]
    for idx in rng.integers(len(a),size=(1500,len(a))):
        x,y=a[idx].sum(0),b[idx].sum(0)
        out.append((f1(x[:3])-f1(y[:3]),f1(x[3:])-f1(y[3:])))
    q=np.quantile(out,(.025,.975),axis=0).T
    return {"exact_labeled_span_f1":q[0].tolist(),"role_boundary_f1":q[1].tolist()}


def tune():
    bank=phrase.dense.prepare("select")
    lex=phrase.cached_scores("select")
    base,u=source_unary(bank,lex)
    offset=np.ascontiguousarray(bank[1],dtype=np.int64)
    engine=load_decoder()
    w,n,changes,ids=train()
    wpath=OUT/(TAG+".safetensors")
    save_file(w,str(wpath))
    saved=load_file(str(wpath))
    assert all(np.array_equal(w[k],v) for k,v in saved.items())
    control=decode(engine,offset,u,base,np.zeros((len(base),7,7),np.float32),0.,0.,False)
    np.testing.assert_array_equal(base,control)
    y=bank[2]
    base_stat=measure(cm(y,base))
    incumbent_span=f1(shape_metrics(offset,y,base)[:3])
    standard=(float(base_stat[0]),float(base_stat[1]),float(incumbent_span),"none",0.,0.,False)
    rules={"roles":standard,"spans":standard}
    grid=[]
    for mode in ARMS:
        edges=edge_arrays(bank,lex,mode,saved)
        for g,p,guard in product((.025,.060,.12,.22,.38),(-.12,0.,.12),(False,True)):
            est=decode(engine,offset,u,base,edges,g,p,guard)
            m,acc,*_=measure(cm(y,est))
            f=f1(shape_metrics(offset,y,est)[:3])
            rule=(float(m),float(acc),float(f),mode,g,p,guard)
            grid.append(rule)
            if acc>=base_stat[1]-.0015 and rule[:3]>rules["roles"][:3]:
                rules["roles"]=rule
            current=rules["spans"]
            if acc>=base_stat[1]-.003 and m>=base_stat[0]-.005 and (f,m,acc)>(current[2],current[0],current[1]):
                rules["spans"]=rule
    choices={k:v[3:] for k,v in rules.items()}
    receipt={"train_cases":ids,"training_edges":n,"training_role_changes":changes,"gradient_steps":0,
             "packed_decoder_sha256":file_hash(phrase.ROOT/"packed_role_decode.cpp"),
             "source_phrase_sha256":file_hash(phrase.RESULT/(phrase.TAG+"-frozen.json")),
             "checkpoint_sha256":file_hash(wpath),"base_selection":standard[:3],"rules":choices,"grid":grid,
             "selected_metrics":{k:v[:3] for k,v in rules.items()}}
    save_json(phrase.RESULT/(TAG+"-frozen.json"),receipt)
    print("Frozen",rules,"baseline",standard,flush=True)


def evaluate(splits):
    plan=phrase.RESULT/(TAG+"-frozen.json")
    cfg=json.loads(plan.read_text())
    path=OUT/(TAG+".safetensors")
    assert file_hash(path)==cfg["checkpoint_sha256"]
    w=load_file(str(path))
    assert cfg["source_phrase_sha256"]==file_hash(phrase.RESULT/(phrase.TAG+"-frozen.json"))
    engine=load_decoder()
    raw=phrase.RESULT/"known-sentence-numerical-archives.npz"
    meta=phrase.dense.read("known-sentence-numerical-archives")
    assert file_hash(raw)==meta["score_file_sha256"]
    results={"freeze_sha256":file_hash(plan),"splits":{}}
    for split in splits:
        bank=phrase.dense.prepare(split)
        lex=phrase.cached_scores(split)
        base,x=source_unary(bank,lex)
        offsets=np.ascontiguousarray(bank[1],dtype=np.int64)
        correct=phrase.dense.paired_confusion(bank,base)
        runs_base=span_tally(bank,base)
        scores=row_summary(correct.sum(0))
        assert scores==phrase.dense.read(phrase.TAG)["splits"][split]["arms"]["target phrases"]["scores"]
        q=results["splits"][split]={"cases":len(bank[0]),"baseline":{"labels":scores,"spans":span_info(runs_base)},"arms":{}}
        if split=="test":
            with np.load(raw,allow_pickle=False) as f:
                np.testing.assert_array_equal(f["cases"],bank[0])
                author=f[meta["best_by_macro_f1"]]
            q["aligned_author"]=row_summary(author.sum(0))
        for a,(mode,g,p,protect) in cfg["rules"].items():
            if mode=="none":
                y=base
            else:
                edge=edge_arrays(bank,lex,mode,w)
                y=decode(engine,offsets,x,base,edge,g,p,protect)
            stat=phrase.dense.paired_confusion(bank,y)
            spans=span_tally(bank,y)
            q["arms"][a]={"labels":row_summary(stat.sum(0)),"spans":span_info(spans),
                          "paired_to_baseline":paired(stat,correct),"span_intervals":span_intervals(spans,runs_base),
                          "changed":int(np.count_nonzero(base!=y))}
            if split=="test":
                q["arms"][a]["paired_to_aligned_author"]=paired(stat,author)
            if split=="select":
                v=q["arms"][a]
                assert v["labels"]["macro_f1"]==cfg["selected_metrics"][a][0]
                assert v["spans"]["exact_labeled_span_f1"]==cfg["selected_metrics"][a][2]
        save_json(phrase.RESULT/(TAG+".json"),results)
        print(split,scores["macro_f1"],q["baseline"]["spans"]["exact_labeled_span_f1"],
              {a:(z["labels"]["macro_f1"],z["labels"]["accuracy"],z["spans"]["exact_labeled_span_f1"],z["changed"])
               for a,z in q["arms"].items()},flush=True)


def raw_document(text, key="input"):
    from span_model import units
    offsets=units(text)
    return {"id":key,"text":text,"units":offsets,"reference_units":offsets,
            "reference_roles":[0]*len(offsets),
            "gold_spans":[{"start":0,"end":len(text),"label":"none"}]}


def run_input(text,is_raw,profile,revise=None,*,revise_with_context=None):
    from hier_model import LABELS,merge
    from sentence_document_graph_write import torch

    model,cfg,vocab,memory,vectors,weight,*_=phrase.load_predictor("target phrases")
    if is_raw:
        item=raw_document(text)
        offsets=item["units"]
    else:
        if not isinstance(text,list) or not text or any(not isinstance(t,str) or not t.strip() for t in text):
            raise ValueError("Supply a JSON array of nonblank sentence strings")
        item=phrase.dense.document("input",[(i,t,0) for i,t in enumerate(text)])
        offsets=item["reference_units"]
    inputs=[item["text"][a:b].strip() for a,b in offsets]
    torch.set_num_threads(2)
    z=phrase.sparse(vectors,[phrase.canonical(t) for t in inputs])@weight
    h=model.contextualize(model.words(inputs))
    k,roles,y,anchor,feat=phrase.dense.dense_features(item,h,model,cfg,vocab,memory)
    ends=np.array([0,len(roles)],dtype=np.int64)
    bank=([k],ends,roles,y,anchor,feat)
    base,unary=source_unary(bank,z)
    freeze=phrase.RESULT/(TAG+"-frozen.json")
    plan=json.loads(freeze.read_text())
    path=OUT/(TAG+".safetensors")
    assert file_hash(path)==plan["checkpoint_sha256"]
    assert plan["source_phrase_sha256"]==file_hash(phrase.RESULT/(phrase.TAG+"-frozen.json"))
    mode,g,p,guard=plan["rules"][profile]
    if mode=="none":
        prediction=base
    else:
        links=edge_arrays(bank,z,mode,load_file(str(path)))
        prediction=decode(load_decoder(),ends,unary,base,links,g,p,guard)
    if revise is not None:
        prediction=revise(bank,z,unary,prediction)
    if revise_with_context is not None:
        prediction=revise_with_context(bank,z,unary,prediction,item,h,model,cfg,vocab)
    if is_raw:
        return {"spans":merge(offsets,prediction,LABELS)}
    start,end,v=runs(ends,prediction)
    return {"labels":[LABELS[i] for i in prediction],"spans":[
        {"start_sentence":int(a),"end_sentence":int(b),"label":LABELS[r]} for a,b,r in zip(start,end,v)]}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("tune","evaluate","predict"))
    parser.add_argument("splits",nargs="*")
    parser.add_argument("--input",type=Path)
    parser.add_argument("--raw",action="store_true")
    parser.add_argument("--profile",choices=("roles","spans"),default="spans")
    a=parser.parse_args()
    if a.mode=="predict":
        if a.input is None:
            parser.error("predict requires --input")
        data=a.input.read_bytes().decode("utf-8")
        print(json.dumps(run_input(data if a.raw else json.loads(data),a.raw,a.profile),ensure_ascii=False))
    elif a.mode=="tune":
        tune()
    else:
        evaluate(a.splits or SPLITS)
