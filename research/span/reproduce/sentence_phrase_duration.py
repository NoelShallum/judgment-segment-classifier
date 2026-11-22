"""Fit role-duration hazards and benchmark exact span decoding against first order."""
import ctypes
from itertools import product
import json
import subprocess

import numpy as np
from safetensors.numpy import load_file,save_file

import sentence_phrase_sequence as seq
from sentence_document_graph_write import OUT,cm,file_hash,measure,paired,row_summary,save_json

P=seq.phrase
TAG="sentence-phrase-duration"
AGES=12


def load_engine():
    source=P.ROOT/"packed_duration_decode.cpp"
    dest=P.ROOT/"cache"/("packed-duration-"+file_hash(source)[:16]+".so")
    if not dest.is_file():
        tmp=dest.with_suffix(".part")
        subprocess.run(["g++","-std=c++17","-shared","-fPIC","-O3","-ffp-contract=off",str(source),"-o",str(tmp)],check=True)
        tmp.replace(dest)
    fn=ctypes.CDLL(str(dest)).decode_durations
    v=np.ctypeslib.ndpointer
    fn.argtypes=[v(np.float64,flags="C_CONTIGUOUS"),v(np.float32,flags="C_CONTIGUOUS"),
                 v(np.int64,flags="C_CONTIGUOUS"),ctypes.c_size_t,ctypes.c_size_t,
                 v(np.float32,flags="C_CONTIGUOUS"),v(np.float32,flags="C_CONTIGUOUS"),v(np.int8,flags="C_CONTIGUOUS")]
    fn.restype=None
    return fn


def train():
    with np.load(P.dense.CACHE/"fit.npz",allow_pickle=False) as f:
        left,right,label=seq.runs(f["offsets"],f["gold"])
        ids=f["cases"].tolist()
    event=np.zeros((7,AGES),np.float64)
    exposure=np.zeros_like(event)
    for n,r in zip(right-left,label):
        exposure[r,:min(n,AGES)]+=1
        exposure[r,-1]+=max(0,n-AGES)
        event[r,min(n,AGES)-1]+=1
    h=event.sum(1)/exposure.sum(1)
    a=np.clip((event+64*h[:,None])/(exposure+64),.02,.98)
    base=np.clip(h[:,None],.02,.98)
    data={"stop":np.ascontiguousarray(np.log(a/base),dtype=np.float32),
          "keep":np.ascontiguousarray(np.log((1-a)/(1-base)),dtype=np.float32),
          "hazard":np.ascontiguousarray(a,dtype=np.float32)}
    return data,ids,{"gold_runs":int(event.sum()),"exposed_sentences":int(exposure.sum())}


def predict(engine,offset,unary,base,pair,strength,stay,guard,w,gain):
    if guard:
        unary=unary.copy()
        unary[(base[:,None]==5)!=(np.arange(7)[None,:]==5)]=-1.e8
    x=np.ascontiguousarray(pair*strength,dtype=np.float32)
    i=np.arange(7)
    x[:,i,i]+=stay
    p=np.empty(len(base),np.int8)
    engine(unary,x,offset,len(offset)-1,AGES,w["stop"]*gain,w["keep"]*gain,p)
    return p


def verify_engine(engine):
    rng=np.random.default_rng(394)
    for n in (1,2,4,5):
        un=rng.normal(size=(n,7))
        edge=rng.normal(scale=.21,size=(n,7,7)).astype(np.float32)
        w={k:rng.normal(scale=.15,size=(7,AGES)).astype(np.float32) for k in ("keep","stop")}
        path=predict(engine,np.array([0,n]),un,np.zeros(n,np.int8),edge,1.,0.,False,w,1.)
        def score(y):
            s=un[0,y[0]]
            r,a=y[0],0
            for i in range(1,len(y)):
                t=y[i]
                s+=un[i,t]+edge[i,r,t]
                if t==r:
                    s+=w["keep"][r,a]
                    a=min(a+1,AGES-1)
                else:
                    s+=w["stop"][r,a]
                    r,a=t,0
            return s+w["stop"][r,a]
        best=max(score(q) for q in product(range(7),repeat=n))
        assert abs(score(path)-best)<1e-7
    return [1,2,4,5]


def main():
    store=P.RESULT
    engine=load_engine()
    checked=verify_engine(engine)
    previous=seq.load_decoder()
    w,ids,fit=train()
    path=OUT/(TAG+".safetensors")
    save_file(w,str(path))
    loaded=load_file(str(path))
    assert all(np.array_equal(loaded[k],v) for k,v in w.items())
    seq_f=store/(seq.TAG+"-frozen.json")
    frozen=json.loads(seq_f.read_text())
    edge_path=OUT/(seq.TAG+".safetensors")
    assert file_hash(edge_path)==frozen["checkpoint_sha256"]
    links=load_file(str(edge_path))
    choices={}
    output={}
    score_grid={}
    for split in seq.SPLITS:
        bank=P.dense.prepare(split)
        lex=P.cached_scores(split)
        base,unary=seq.source_unary(bank,lex)
        offset=np.ascontiguousarray(bank[1],dtype=np.int64)
        q=output[split]={"cases":len(bank[0]),"arms":{}}
        for profile,(mode,strength,stay,guard) in frozen["rules"].items():
            edge=seq.edge_arrays(bank,lex,mode,links) if mode!="none" else np.zeros((len(base),7,7),np.float32)
            original=seq.decode(previous,offset,unary,base,edge,strength,stay,guard)
            shadow=predict(engine,offset,unary,base,edge,strength,stay,guard,loaded,0.)
            np.testing.assert_array_equal(original,shadow)
            g=bank[2]
            if split=="select":
                prior=measure(cm(g,original))
                prior_spans=seq.f1(seq.shape_metrics(offset,g,original)[:3])
                best=(float(prior[0]),float(prior[1]),float(prior_spans),0.)
                score_grid[profile]=[]
                for alpha in (.06,.14,.30,.55,.85):
                    z=predict(engine,offset,unary,base,edge,strength,stay,guard,loaded,alpha)
                    m,a,*_=measure(cm(g,z))
                    f=seq.f1(seq.shape_metrics(offset,g,z)[:3])
                    candidate=(float(m),float(a),float(f),alpha)
                    score_grid[profile].append(candidate)
                    if profile=="roles" and a>=prior[1]-.0015 and candidate[:3]>best[:3]:
                        best=candidate
                    if profile=="spans" and a>=prior[1]-.003 and m>=prior[0]-.005 and (f,m,a)>(best[2],best[0],best[1]):
                        best=candidate
                choices[profile]=best[-1]
            z=predict(engine,offset,unary,base,edge,strength,stay,guard,loaded,choices[profile])
            a,b=(P.dense.paired_confusion(bank,y) for y in (z,original))
            x,y=(seq.span_tally(bank,p) for p in (z,original))
            q["arms"][profile]={"incumbent_labels":row_summary(b.sum(0)),"incumbent_spans":seq.span_info(y),
               "labels":row_summary(a.sum(0)),"spans":seq.span_info(x),"paired_labels":paired(a,b),
               "paired_spans":seq.span_intervals(x,y)}
        if split=="select":
            plan=store/(TAG+"-frozen.json")
            save_json(plan,{"source_sequence_sha256":file_hash(seq_f),"age_cap":AGES,"fit_cases":ids,"fit":fit,
                      "weights_sha256":file_hash(path),"choices":choices,"grid":score_grid,"exhaustive_paths_checked_for_lengths":checked,
                      "cpp_sha256":file_hash(P.ROOT/"packed_duration_decode.cpp"),"gradient_steps":0})
            print("Frozen duration scales",choices,flush=True)
        save_json(store/(TAG+".json"),{"freeze_sha256":file_hash(plan),"splits":output})
        print(split,{k:(v["labels"]["macro_f1"],v["spans"]["exact_labeled_span_f1"]) for k,v in q["arms"].items()},flush=True)


if __name__=="__main__":
    main()
