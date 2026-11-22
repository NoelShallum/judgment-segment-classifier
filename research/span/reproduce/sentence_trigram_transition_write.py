"""Write second-order transition weights from LegalSeg role sequences."""
import argparse
import ctypes
from itertools import product
import json
import subprocess
from tempfile import NamedTemporaryFile

import numpy as np
from safetensors.numpy import load_file,save_file

import sentence_pairwise_repair as repair
import sentence_phrase_sequence as seq
from sentence_document_graph_write import OUT,cm,file_hash,measure,paired,row_summary,save_json

P=repair.P
TAG="sentence-trigram-transition"
SOURCE=P.ROOT/"packed_role_trigram.cpp"
PATCH=OUT/(TAG+".safetensors")
SPLITS=("select","confirm","holdout","phrase-prospect","test")


def engine():
    lib=P.ROOT/"cache"/("packed-trigram-"+file_hash(SOURCE)[:16]+".so")
    if not lib.is_file():
        with NamedTemporaryFile(dir=lib.parent,suffix=".part",delete=False) as f:
            tmp=P.ROOT/"cache"/f.name
        try:
            subprocess.run(["g++","-std=c++17","-O3","-ffp-contract=off","-shared","-fPIC",str(SOURCE),"-o",str(tmp)],check=True)
            tmp.replace(lib)
        finally:
            tmp.unlink(missing_ok=True)
    dec=ctypes.CDLL(str(lib)).decode_trigrams
    nd=lambda t:np.ctypeslib.ndpointer(t,flags="C_CONTIGUOUS")
    dec.argtypes=[nd(np.float64),nd(np.float32),nd(np.int64),ctypes.c_size_t,nd(np.int8),nd(np.float32),
                  ctypes.c_float,ctypes.c_float,ctypes.c_float,nd(np.int8)]
    dec.restype=None
    return dec


def position(offset):
    return np.concatenate([(np.arange(b-a)*3//(b-a)).astype(np.int8) for a,b in zip(offset,offset[1:])])


def train(bank):
    off=bank[1]
    g=bank[2].astype(np.int64)
    pos=position(off)
    start=np.concatenate([np.arange(a+2,b) for a,b in zip(off,off[1:])])
    key=49*g[start-2]+7*g[start-1]+g[start]
    local=np.bincount(343*pos[start].astype(np.int64)+key,minlength=3*343).reshape(3,7,7,7)
    mass=local.sum(0).astype(np.float64)
    prior=mass.sum(0)+2
    prior/=prior.sum(1,keepdims=True)
    normal=mass+96*prior[None,:,:]
    normal/=normal.sum(2,keepdims=True)
    global_edit=np.log(normal)-np.log(prior)[None,:,:]
    phase=local+180*normal[None,:,:,:]
    phase/=phase.sum(3,keepdims=True)
    phase_pair=local.sum(1)+240*prior[None,:,:]
    phase_pair/=phase_pair.sum(2,keepdims=True)
    edit=np.log(phase)-np.log(phase_pair)[:,None,:,:]
    return {"global":np.ascontiguousarray(np.clip(global_edit,-1.8,1.8)[None],dtype=np.float32),
            "phase":np.ascontiguousarray(np.clip(edit,-1.8,1.8),dtype=np.float32)},len(start)


def predict(dec,off,x,edge,pos,weights,mode,gain,stay,triple):
    p=np.empty(len(x),dtype=np.int8)
    sec=pos if mode=="phase" else np.zeros(len(x),np.int8)
    dec(x,edge,off,len(off)-1,sec,weights[mode],gain,stay,triple,p)
    return p


def verify(dec):
    rng=np.random.default_rng(31879)
    for n in (1,2,3,4,5):
        x=rng.normal(size=(n,7))
        p_edge=rng.normal(0,1,(n,7,7)).astype(np.float32)
        weights=rng.normal(0,1,(3,7,7,7)).astype(np.float32)
        pos=position(np.array([0,n]))
        edge_scale,repeat,triple=.08,-.12,.3
        found=predict(dec,np.array([0,n]),x,p_edge,pos,{"phase":weights},"phase",edge_scale,repeat,triple)
        def path_score(y):
            a=sum(v[k] for v,k in zip(x,y))
            for i in range(1,n):
                p,c=y[i-1],y[i]
                a += p_edge[i,p,c]*edge_scale + repeat*(p==c)
                if i>=2:
                    a+=weights[pos[i],y[i-2],p,c]*triple
            return a
        expected=max(product(range(7),repeat=n),key=path_score)
        np.testing.assert_array_equal(found,expected)


def data(split,recipe):
    bank=P.dense.prepare(split)
    z=P.cached_scores(split)
    base,x=seq.source_unary(bank,z)
    mode,gain,stay,guard=recipe["rules"]["spans"]
    assert not guard
    edge=seq.edge_arrays(bank,z,mode,recipe["weights"])
    pred=seq.decode(recipe["engine"],bank[1],x,base,edge,gain,stay,guard)
    assert row_summary(cm(bank[2],pred))==P.dense.read(seq.TAG)["splits"][split]["arms"]["spans"]["labels"]
    return bank,x,edge,pred,gain,stay


def fit():
    dec=engine()
    verify(dec)
    fit_bank=P.dense.prepare("fit")
    w,n=train(fit_bank)
    save_file(w,str(PATCH))
    stored=load_file(str(PATCH))
    assert all(np.array_equal(stored[k],v) for k,v in w.items())
    bank,x,edge,base,gain,stay=data("select",repair.load_recipe())
    off=bank[1]
    pos=position(off)
    gold=bank[2]
    original=(*measure(cm(gold,base))[:2],seq.f1(seq.shape_metrics(off,gold,base)[:3]))
    np.testing.assert_array_equal(predict(dec,off,x,edge,pos,stored,"global",gain,stay,0.),base)
    metric0=list(map(float,original))
    results={k:{"choice":["global",0.],"selection":metric0} for k in ("roles","spans")}
    grid=[]
    for mode,k in product(("global","phase"),(.01,.025,.05,.1,.16,.24,.35)):
        p=predict(dec,off,x,edge,pos,stored,mode,gain,stay,k)
        m,a,*_=measure(cm(gold,p))
        f=seq.f1(seq.shape_metrics(off,gold,p)[:3])
        candidate=list(map(float,(m,a,f)))
        grid.append([[mode,k],candidate])
        for target in results:
            old=results[target]["selection"]
            wins=candidate>old if target=="roles" else (f,m,a)>(old[2],old[0],old[1]) and m>=metric0[0]-.002
            if wins and a>=metric0[1]-.0015:
                results[target]={"choice":[mode,k],"selection":candidate}
    save_json(repair.receipt(TAG),{"patch_sha256":file_hash(PATCH),"source_seq_sha256":file_hash(repair.receipt(seq.TAG)),
        "cpp_sha256":file_hash(SOURCE),"fit_cases":fit_bank[0].tolist(),"selection_cases":bank[0].tolist(),
        "labeled_triples":n,"gradient_steps":0,"exhaustive_lengths_verified":[1,2,3,4,5],"profiles":results,"grid":grid})
    print("Frozen",results,flush=True)


def evaluate(splits):
    dec=engine()
    recipe=repair.load_recipe()
    path=repair.receipt(TAG)
    plan=json.loads(path.read_text())
    assert plan["cpp_sha256"]==file_hash(SOURCE) and plan["patch_sha256"]==file_hash(PATCH)
    assert plan["source_seq_sha256"]==file_hash(repair.receipt(seq.TAG))
    w=load_file(str(PATCH))
    report={"freeze_sha256":file_hash(path),"splits":{}}
    for split in splits:
        bank,x,edges,base,gain,stay=data(split,recipe)
        assert not set(plan["fit_cases"]) & set(bank[0])
        if split!="select":
            assert not set(plan["selection_cases"]) & set(bank[0])
        off=bank[1]
        pos=position(off)
        np.testing.assert_array_equal(predict(dec,off,x,edges,pos,w,"global",gain,stay,0.),base)
        original=P.dense.paired_confusion(bank,base)
        old_spans=seq.span_tally(bank,base)
        r=report["splits"][split]={"cases":len(bank[0]),"baseline":row_summary(original.sum(0)),
                                  "baseline_spans":seq.span_info(old_spans),"profiles":{}}
        for name,entry in plan["profiles"].items():
            mode,g=entry["choice"]
            p=predict(dec,off,x,edges,pos,w,mode,gain,stay,g)
            cf=P.dense.paired_confusion(bank,p)
            spans=seq.span_tally(bank,p)
            q=r["profiles"][name]={"labels":row_summary(cf.sum(0)),"spans":seq.span_info(spans),"paired":paired(cf,original),
                                   "paired_spans":seq.span_intervals(spans,old_spans),"changed":int(np.count_nonzero(p!=base))}
            if split=="select":
                m,a,*_=measure(cf.sum(0))
                assert [float(m),float(a),float(q["spans"]["exact_labeled_span_f1"])]==entry["selection"]
        save_json(P.RESULT/(TAG+".json"),report)
        print(split,{k:(q["labels"]["macro_f1"],q["spans"]["exact_labeled_span_f1"],q["changed"])
                      for k,q in r["profiles"].items()},flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("fit","evaluate"))
    p.add_argument("splits",nargs="*")
    a=p.parse_args()
    if a.action=="fit":
        fit()
    else:
        evaluate(a.splits or SPLITS)
