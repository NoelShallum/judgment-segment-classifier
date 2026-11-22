"""Independently replay the lexical sentence edit and score the aligned author archive."""
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np

from sentence_ngram_residual_write import (
    STEM,OUT,PARENT,ROOT,ROUTE,cm,document,file_hash,gold_cases,load_file,
    ordered,paired,prepare,row_summary,save_json,
)


def replay(docs):
    receipt=json.loads((ROOT/"results"/(STEM+"-frozen.json")).read_text())
    arm=receipt["choice"][0]
    head=load_file(str(OUT/(STEM+".safetensors")))
    memory=load_file(str(OUT/(STEM+"-memory.safetensors")))["lexical"]
    rate,gain,threshold=head["settings"]
    records=[]
    for c,lex in prepare(docs,"test",memory):
        routes=ordered(c,ROUTE)
        k=min(len(routes),max(1,int(np.ceil(rate*len(c.gold))))) if rate else min(3,len(routes))
        at=routes[:k]
        x=c.structure[at]
        if arm!="hierarchy":
            l=lex[at]
            size=7 if arm=="unigram-bigram" else 21
            x=np.concatenate((x,l[:,:size],l[:,:7]*x[:,:7]),axis=1)
        y=c.current.copy()
        # Evaluate the materialized affine map directly on the referred subset.
        w=head["scale"][:,None]*head["matrix"]
        b=head["bias"]-head["center"]@w
        z=np.clip(x@w+b,-.65,.65)
        p=c.prior[at]+gain*z
        top=p.argmax(1)
        delta=p[np.arange(k),top]-p[np.arange(k),y[at]]
        changed=delta>=threshold
        y[at[changed]]=top[changed]
        records.append((c.key,cm(c.gold,y),cm(c.gold,c.current),int((y!=c.current).sum())))
    return records


def main():
    evidence=ROOT/"results"
    freeze=evidence/(STEM+"-frozen.json")
    receipt=json.loads(freeze.read_text())
    assert receipt["parent_sha256"]==file_hash(PARENT)
    for suffix,field in (("","head_sha256"),("-memory","memory_sha256")):
        assert file_hash(OUT/(STEM+suffix+".safetensors"))==receipt[field]
    assert set(receipt["memory_training_cases"]).isdisjoint(receipt["readout_training_cases"])
    docs=[document(k,r) for k,r in gold_cases().items()]
    with ProcessPoolExecutor(3) as pool:
        entries={i:(a,b,c) for pack in pool.map(replay,[docs[i::3] for i in range(3)]) for i,a,b,c in pack}
    assert len(entries)==len(docs)
    ref=json.loads((evidence/"known-sentence-numerical-archives.json").read_text())
    compressed=evidence/"known-sentence-numerical-archives.npz"
    assert file_hash(compressed)==ref["score_file_sha256"]
    best=ref["best_by_macro_f1"]
    with np.load(compressed,allow_pickle=False) as o:
        np.testing.assert_array_equal(o["cases"],[d["id"] for d in docs])
        author=o[best]
    a,b=(np.stack([entries[d["id"]][t] for d in docs]) for t in (0,1))
    n=sum(c for a0,b0,c in entries.values())
    info={"cases":len(docs),"candidate":row_summary(a.sum(0)),"baseline":row_summary(b.sum(0)),
          "changed":n,"aligned_archive_name":best,"aligned_archive":row_summary(author.sum(0)),
          "paired_to_baseline":paired(a,b),"paired_to_aligned_archive":paired(a,author),
          "freeze_sha256":file_hash(freeze)}
    old=json.loads((evidence/"sentence-joint-qa.json").read_text())["scores"]
    assert info["baseline"]==old
    path=evidence/(STEM+".json")
    if path.is_file() and "test" in (record:=json.loads(path.read_text()))["splits"]:
        assert all(info[f]==record["splits"]["test"][f] for f in ("cases","candidate","baseline","changed"))
    save_json(evidence/(STEM+"-qa.json"),info)
    print("Official replay",len(docs),n,info["candidate"]["macro_f1"],
          info["paired_to_baseline"]["macro_f1"],info["paired_to_aligned_archive"]["macro_f1"],flush=True)


if __name__=="__main__":
    main()
