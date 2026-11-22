"""Fit out-of-fold lexical role and residual scores, then write a dense context readout."""
import argparse
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import re

import numpy as np
from safetensors.numpy import load_file,save_file
from scipy.linalg import cho_factor,cho_solve
from scipy.sparse import diags,hstack
from sklearn.feature_extraction.text import CountVectorizer

import sentence_dense_readout_write as dense
from sentence_document_graph_write import OUT,ROOT,cm,file_hash,measure,paired,row_summary,save_json

TAG="sentence-phrase-readout"
CACHE=ROOT/"cache"/TAG
RESULT=ROOT/"results"
FOLDS=4
SPLITS=("fit","select","confirm","holdout","prospective","role-guard","phrase-prospect","test")
VOCABS=(("word",(1,3),3072),("char_wb",(3,5),1024))
NUMBER=re.compile(r"\d+")


def canonical(text):
    return NUMBER.sub(" __num__ ",text.casefold())


def docs(split):
    if split=="fit":
        return dense.fit_documents()
    if split in ("prospective","role-guard","phrase-prospect"):
        return dense.prospective_documents(split)
    return dense.groups_for(split)


def corpus(split):
    with np.load(dense.CACHE/(split+".npz"),allow_pickle=False) as f:
        ids,offsets,labels,anchor=(f[k] for k in ("cases","offsets","gold","anchor"))
    ds=docs(split)
    np.testing.assert_array_equal(ids,[d["id"] for d in ds])
    source=[canonical(d["text"][a:b].strip()) for d in ds for a,b in d["reference_units"]]
    assert len(source)==len(labels)
    return ids,offsets,labels,anchor,source


def vectorizers(vocab=None):
    return [CountVectorizer(analyzer=mode,ngram_range=span,max_features=capacity,min_df=4,binary=True,
                            dtype=np.float32,lowercase=False,token_pattern=r"(?u)\b\w+\b",
                            vocabulary=None if vocab is None else vocab[i])
            for i,(mode,span,capacity) in enumerate(VOCABS)]


def sparse(vectors,source):
    x=hstack([a.transform(source) for a in vectors],format="csr",dtype=np.float32)
    x.sum_duplicates()
    d=np.diff(x.indptr)
    x.data /= np.repeat(np.sqrt(np.maximum(d,1)).astype(np.float32),d)
    return x


def solve(gram,link,smoothing):
    a=gram.copy()
    a.flat[::len(a)+1]+=smoothing
    return cho_solve(cho_factor(a,overwrite_a=True,check_finite=False),link,check_finite=False)


def moments(x,y,weight):
    d=diags(weight,format="csr",dtype=np.float32)@x
    return (x.T@d).toarray(),np.asarray(d.T@y,dtype=np.float32)


def fit_lexical():
    CACHE.mkdir(exist_ok=True)
    ids,offsets,labels,anchor,source=corpus("fit")
    vectors=vectorizers()
    for a in vectors:
        a.fit(source)
    vocab=[{k:int(v) for k,v in a.vocabulary_.items()} for a in vectors]
    save_json(OUT/(TAG+"-vocabulary.json"),vocab)
    x=sparse(vectors,source)
    del source
    hot=np.eye(7,dtype=np.float32)[labels]
    targets=np.column_stack((hot,hot-anchor))
    weight=(1.+1.5*(1-anchor[np.arange(len(labels)),labels])**2).astype(np.float32)
    fold=np.concatenate([np.repeat(int.from_bytes(sha256(("phrase-v1|"+str(i)).encode()).digest()[:2],"big")%FOLDS,b-a)
                         for i,a,b in zip(ids,offsets,offsets[1:])])
    shape=x.shape[1]
    total=np.zeros((shape,shape),np.float32)
    rhs=np.zeros((shape,14),np.float32)
    partial=[]
    for k in range(FOLDS):
        q=fold==k
        gram,link=moments(x[q],targets[q],weight[q])
        total+=gram
        rhs+=link
        partial.append((gram,link))
        print("Fold moments",k,int(q.sum()),flush=True)
    scales=(.75,5.)
    fitted=np.empty((len(labels),len(scales)*14),np.float32)
    matrices=[]
    for j,s in enumerate(scales):
        full=solve(total,rhs,s)
        matrices.append(full)
        for k,(gram,link) in enumerate(partial):
            q=fold==k
            delta=solve(total-gram,rhs-link,s)
            fitted[q,j*14:(j+1)*14]=x[q]@delta
    coefficients=OUT/(TAG+"-lexical.safetensors")
    weight=np.ascontiguousarray(np.concatenate(matrices,axis=1))
    save_file({"weight":weight},str(coefficients))
    np.testing.assert_array_equal(load_file(str(coefficients))["weight"],weight)
    save_json(RESULT/(TAG+"-lexical-frozen.json"),{"fit_cases":ids.tolist(),"fit_rows":len(labels),"folds":FOLDS,
              "ridge":list(scales),"columns":shape,"nnz":x.nnz,"weights_sha256":file_hash(coefficients),
              "vocabulary_sha256":file_hash(OUT/(TAG+"-vocabulary.json"))})
    np.save(CACHE/"fit.npy",fitted,allow_pickle=False)
    print("Fitted",shape,"lexical dimensions",flush=True)


def cached_scores(split):
    if split=="fit":
        return np.load(CACHE/"fit.npy",allow_pickle=False)
    path=CACHE/(split+".npy")
    if path.is_file():
        return np.load(path,allow_pickle=False)
    _,_,_,_,source=corpus(split)
    vpath=OUT/(TAG+"-vocabulary.json")
    patch=OUT/(TAG+"-lexical.safetensors")
    proof=json.loads((RESULT/(TAG+"-lexical-frozen.json")).read_text())
    assert proof["vocabulary_sha256"]==file_hash(vpath) and proof["weights_sha256"]==file_hash(patch)
    model=vectorizers(json.loads(vpath.read_text()))
    w=load_file(str(patch))["weight"]
    y=np.concatenate([sparse(model,source[j:j+8192])@w for j in range(0,len(source),8192)])
    stage=path.with_suffix(".tmp")
    with stage.open("wb") as f:
        np.save(f,y,allow_pickle=False)
    stage.replace(path)
    print("Scored lexical rows",split,len(source),flush=True)
    return y


ARMS=("dense control","target phrases","contextual phrases")
LEXICAL=("small ridge","large ridge")


def design(bank,z,mode,level):
    x=bank[-1]
    if mode=="dense control":
        return x.copy()
    raw=z[:,level*14:(level+1)*14]
    features=[x,raw,raw[:,:7]*x[:,40:47]]
    if mode=="contextual phrases":
        context=np.empty((len(x),8*14),np.float32)
        bounds=bank[1]
        for a,b in zip(bounds,bounds[1:]):
            context[a:b]=np.column_stack(tuple(dense.windows(raw[a:b])))
        features.append(context)
    return np.column_stack(features)


def predict_rows(bank,score,setting):
    _,_,_,target,g,margin,guard=setting
    result=dense.replace(bank[3],bank[4],score,target,g,margin,guard!="all")
    return dense.preserve_issues(bank[3],result) if guard=="reasoning and issue" else result


def learn_fusion():
    train=dense.prepare("fit")
    val=dense.prepare("select")
    z_train,z_val=(cached_scores(s) for s in ("fit","select"))
    original=dense.baseline(val,"select")
    roles=train[2]
    counts=np.maximum(np.bincount(roles,minlength=7)/len(roles),.013)
    difficulty=(1.-train[4][np.arange(len(roles)),roles])**2
    lexical_plan=RESULT/(TAG+"-lexical-frozen.json")
    prospective=RESULT/"sentence-dense-readout-phrase-prospect.json"
    frozen={"lexical_freeze_sha256":file_hash(lexical_plan),"prospect_sha256":file_hash(prospective),
             "fit_rows":len(roles),"validation_cases":val[0].tolist(),"gradient_steps":0,"arms":{}}
    for index,mode in enumerate(ARMS):
        peak=(-1.,)
        chosen=None
        grid=[]
        for level in range(1 if mode=="dense control" else len(LEXICAL)):
            x=design(train,z_train,mode,level)
            v=design(val,z_val,mode,level)
            center=x.mean(0,dtype=np.float64).astype(np.float32)
            scale=(1./np.maximum(x.std(0,dtype=np.float64),.025)).astype(np.float32)
            x-=center
            x*=scale
            z=np.column_stack((x,np.ones(len(x),np.float32)))
            del x
            for rarity in (0.,.18,.35):
                weights=(counts[roles]**-rarity*(1+1.5*difficulty)).astype(np.float32)
                gram,links=dense.fit_moments(z,roles,train[4],weights)
                penalty=np.r_[np.ones(len(gram)-1),.02]*gram.trace()/len(gram)
                for damping in (.02,.09):
                    w=np.linalg.solve(gram+np.diag(damping*penalty),links).astype(np.float32)
                    trained={"center":center,"scale":scale,"weight":w[:-1].copy(),"bias":w[-1].copy()}
                    est=dense.output(v,trained)
                    limits=product(("posterior","residual"),(0.,.30,.60,.85),(.035,.10),("all","reasoning","reasoning and issue"))
                    for target,g,margin,guard in limits:
                        opt=(level,rarity,damping,target,g,margin,guard)
                        ans=predict_rows(val,est,opt) if g else val[3]
                        macro,acc,*_=measure(cm(val[2],ans))
                        grade=(macro,acc,opt)
                        if grade>peak and acc>=original["accuracy"]-.001:
                            peak,chosen=grade,trained
                        grid.append([opt,float(macro),float(acc)])
        path=OUT/(TAG+f"-{index}.safetensors")
        save_file(chosen,str(path))
        from_disk=load_file(str(path))
        for k,v in chosen.items():
            np.testing.assert_array_equal(v,from_disk[k])
        frozen["arms"][mode]={"choice":peak[2],"select_macro_f1":float(peak[0]),"select_accuracy":float(peak[1]),
                              "weights_sha256":file_hash(path),"width":len(chosen["center"]),"grid":grid}
        print("Frozen",mode,peak,flush=True)
    save_json(RESULT/(TAG+"-frozen.json"),frozen)


def evaluate(splits):
    path=RESULT/(TAG+"-frozen.json")
    frozen=json.loads(path.read_text())
    assert frozen["prospect_sha256"]==file_hash(RESULT/"sentence-dense-readout-phrase-prospect.json")
    assert frozen["lexical_freeze_sha256"]==file_hash(RESULT/(TAG+"-lexical-frozen.json"))
    ref=dense.read("known-sentence-numerical-archives")
    ref_file=RESULT/"known-sentence-numerical-archives.npz"
    assert file_hash(ref_file)==ref["score_file_sha256"]
    output={"freeze_sha256":file_hash(path),"splits":{}}
    for split in splits:
        base=dense.prepare(split)
        words=cached_scores(split)
        prior=dense.baseline(base,split)
        matrices={"parent":dense.paired_confusion(base,base[3])}
        batch=output["splits"][split]={"cases":len(base[0]),"parent":prior,"arms":{}}
        if split=="test":
            with np.load(ref_file,allow_pickle=False) as f:
                np.testing.assert_array_equal(f["cases"],base[0])
                author=f[ref["best_by_macro_f1"]]
        for index,(name,v) in enumerate(frozen["arms"].items()):
            model=OUT/(TAG+f"-{index}.safetensors")
            assert file_hash(model)==v["weights_sha256"]
            w=load_file(str(model))
            action=v["choice"]
            score=dense.output(design(base,words,name,action[0]),w)
            pred=predict_rows(base,score,action) if action[4] else base[3]
            mat=matrices[name]=dense.paired_confusion(base,pred)
            batch["arms"][name]={"scores":row_summary(mat.sum(0)),"paired_parent":paired(mat,matrices["parent"]),
               "paired_dense":paired(mat,matrices[ARMS[0]]),"changed":int(np.count_nonzero(pred!=base[3]))}
            if split=="test":
                batch["arms"][name]["paired_author"]=paired(mat,author)
            if split=="select":
                assert batch["arms"][name]["scores"]["macro_f1"]==v["select_macro_f1"]
        save_json(RESULT/(TAG+".json"),output)
        print(split,"parent",prior["macro_f1"],{k:(v["scores"]["macro_f1"],v["scores"]["accuracy"])
                    for k,v in batch["arms"].items()},flush=True)


def load_predictor(arm):
    from sentence_document_graph_write import DONOR,EDGE,HierModel
    plan=json.loads((RESULT/(TAG+"-frozen.json")).read_text())
    lex=json.loads((RESULT/(TAG+"-lexical-frozen.json")).read_text())
    v=OUT/(TAG+"-vocabulary.json")
    w=OUT/(TAG+"-lexical.safetensors")
    head=OUT/(TAG+f"-{ARMS.index(arm)}.safetensors")
    spec=plan["arms"][arm]
    for path,expected in ((v,lex["vocabulary_sha256"]),(w,lex["weights_sha256"]),(head,spec["weights_sha256"])):
        assert file_hash(path)==expected
    base=HierModel(DONOR)
    decoder=load_file(str(dense.PARENT))
    vocabulary=json.loads((EDGE/"vocab.json").read_text())
    lexical=load_file(str(dense.LEXICAL))["lexical"][0]
    return base,decoder,vocabulary,lexical,vectorizers(json.loads(v.read_text())),load_file(str(w))["weight"],load_file(str(head)),spec["choice"]


def predict(texts,arm,state):
    from hier_model import LABELS
    if not isinstance(texts,list) or not texts or any(not isinstance(t,str) or not t.strip() for t in texts):
        raise ValueError("Supply a nonempty JSON array of nonblank sentence strings")
    model,decoder,vocabulary,lexical,vectors,weights,head,opt=state
    item=dense.document("input",[(i,t,0) for i,t in enumerate(texts)])
    h=model.contextualize(model.words([item["text"][a:b] for a,b in item["reference_units"]]))
    k,l,base,anchor,x=dense.dense_features(item,h,model,decoder,vocabulary,lexical)
    z=sparse(vectors,[canonical(t.strip()) for t in texts])@weights
    bank=([k],[0,len(l)],l,base,anchor,x)
    y=predict_rows(bank,dense.output(design(bank,z,arm,opt[0]),head),opt) if opt[4] else base
    return {"labels":[LABELS[i] for i in y]}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("op",choices=("lexical","cache","learn","evaluate","predict"))
    parser.add_argument("splits",nargs="*")
    parser.add_argument("--arm",choices=ARMS,default="contextual phrases")
    parser.add_argument("--input",type=Path)
    a=parser.parse_args()
    if a.op=="predict":
        if a.input is None:
            parser.error("predict requires --input")
        from sentence_document_graph_write import torch
        torch.set_num_threads(2)
        print(json.dumps(predict(json.loads(a.input.read_text()),a.arm,load_predictor(a.arm)),ensure_ascii=False))
    elif a.op=="lexical":
        fit_lexical()
    elif a.op=="learn":
        learn_fusion()
    elif a.op=="evaluate":
        evaluate(a.splits)
    else:
        for s in a.splits:
            cached_scores(s)
