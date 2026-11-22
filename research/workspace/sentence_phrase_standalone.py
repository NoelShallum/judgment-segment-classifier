"""Compare phrase-only sentence and neighboring-sentence classifiers with no encoder."""
from itertools import product

import numpy as np

import sentence_phrase_readout as phrase
from sentence_document_graph_write import cm,file_hash,measure,paired,row_summary,save_json

TAG="sentence-phrase-standalone"
ARMS=("sentence only","two neighbors per side")


def load(split):
    with np.load(phrase.dense.CACHE/(split+".npz"),allow_pickle=False) as f:
        meta=f["cases"],f["offsets"],f["gold"]
    return meta,phrase.cached_scores(split)


def transform(meta,raw,level):
    q=phrase.dense.positive(raw[:,level*14:level*14+7])
    context=np.empty_like(q)
    for a,b in zip(meta[1],meta[1][1:]):
        left,right,*_=phrase.dense.windows(q[a:b])
        context[a:b]=(left+right)/np.maximum((left.sum(1,keepdims=True)>0).astype(np.int8)+(right.sum(1,keepdims=True)>0),1)
    return q,context


def predictions(q,context,frequency,strength,adjust):
    score=(1-strength)*q+strength*context
    return np.argmax(np.log(np.maximum(score,.003))-adjust*frequency,axis=1)


def confusion(meta,answer):
    _,ends,y=meta
    return np.stack([cm(y[a:b],answer[a:b]) for a,b in zip(ends,ends[1:])])


def main(splits):
    val,z=load("select")
    data,_=load("fit")
    frequency=np.log(np.maximum(np.bincount(data[2],minlength=7)/len(data[2]),.012))
    saved={}
    grid={}
    for arm in ARMS:
        best=(-1.,)
        grid[arm]=[]
        for ridge in range(2):
            q,context=transform(val,z,ridge)
            for strength,bias in product((0.,) if arm==ARMS[0] else (.10,.25,.45),(-.15,0.,.18,.36,.55,.8)):
                p=predictions(q,context,frequency,strength,bias)
                v=measure(cm(val[2],p))
                item=(float(v[0]),float(v[1]),ridge,strength,bias)
                grid[arm].append(item)
                best=max(best,item)
        saved[arm]=best
    f=phrase.RESULT/(TAG+"-frozen.json")
    source=phrase.RESULT/(phrase.TAG+"-lexical-frozen.json")
    save_json(f,{"lexical_freeze_sha256":file_hash(source),"options":saved,"grid":grid,"log_frequency":frequency.tolist()})
    summary={"freeze_sha256":file_hash(f),"splits":{}}
    old=phrase.dense.read("isolated-role-readout")
    for group in splits:
        meta,z=load(group)
        out=summary["splits"][group]={"cases":len(meta[0]),"rows":len(meta[2]),"modes":{}}
        if group in old["splits"]:
            out["previous_isolated"]=old["splits"][group]["candidate"]
            assert out["previous_isolated"]["rows"]==len(meta[2])
        if group=="test":
            archive=phrase.RESULT/"known-sentence-numerical-archives.npz"
            assert file_hash(archive)==phrase.dense.read("known-sentence-numerical-archives")["score_file_sha256"]
            with np.load(archive,allow_pickle=False) as data:
                np.testing.assert_array_equal(data["cases"],meta[0])
                aligned=data["standalone"]
            out["aligned_isolated_author"]=row_summary(aligned.sum(0))
        previous=None
        for arm in ARMS:
            *_,level,strength,bias=saved[arm]
            a,b=transform(meta,z,level)
            p=predictions(a,b,frequency,strength,bias)
            t=confusion(meta,p)
            out["modes"][arm]={"scores":row_summary(t.sum(0))}
            if previous is None:
                if group=="test":
                    out["modes"][arm]["paired_to_isolated_author"]=paired(t,aligned)
                previous=t
            else:
                out["modes"][arm]["paired_to_isolated"]=paired(t,previous)
        print(group,{k:round(v["scores"]["macro_f1"],7) for k,v in out["modes"].items()},flush=True)
        save_json(phrase.RESULT/(TAG+".json"),summary)


if __name__=="__main__":
    main(("select","confirm","holdout","phrase-prospect","test"))
