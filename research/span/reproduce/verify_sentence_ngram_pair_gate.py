"""Replay the reasoning guard as a direct affine-map edit and compare author predictions."""
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np

from sentence_ngram_residual_write import (
    STEM as LEX,OUT,PARENT,ROOT,ROUTE,cm,document,file_hash,gold_cases,load_file,
    ordered,paired,prepare,row_summary,save_json,
)
from sentence_ngram_pair_gate import STEM


def shard(docs):
    head=load_file(str(OUT/(LEX+".safetensors")))
    memory=load_file(str(OUT/(LEX+"-memory.safetensors")))["lexical"]
    rate,gain,threshold=head["settings"]
    w=head["scale"][:,None]*head["matrix"]
    b=head["bias"]-head["center"]@w
    rows=[]
    for c,lex in prepare(docs,"test",memory):
        route=ordered(c,ROUTE)
        at=route[:max(1,int(np.ceil(rate*len(c.gold))))] if rate else route[:3]
        structure=c.structure[at]
        l=lex[at,:7]
        x=np.column_stack((structure,l,l*structure[:,:7]))
        value=c.prior[at]+gain*np.clip(x@w+b,-.65,.65)
        k=np.arange(len(at))
        top=value.argmax(1)
        yes=value[k,top]-value[k,c.current[at]]>=threshold
        lexical=c.current.copy()
        lexical[at[yes]]=top[yes]
        result=np.where(c.current==5,c.current,lexical)
        rows.append((c.key,cm(c.gold,result),cm(c.gold,lexical),cm(c.gold,c.current),int((result!=c.current).sum())))
    return rows


def main():
    root=ROOT/"results"
    gate=root/(STEM+"-frozen.json")
    f=json.loads(gate.read_text())
    base=json.loads((root/(LEX+"-frozen.json")).read_text())
    assert f["choice"]==["guard-reasoning"] and base["choice"][0]=="unigram-bigram"
    assert f["parent_sha256"]==file_hash(PARENT)
    for fkey,suffix in (("lexical_head_sha256",""),("lexical_memory_sha256","-memory")):
        assert file_hash(OUT/(LEX+suffix+".safetensors"))==f[fkey]
    assert set(f["train_cases"]).isdisjoint(base["memory_training_cases"])
    assert set(f["train_cases"]).isdisjoint(base["readout_training_cases"])
    docs=[document(k,r) for k,r in gold_cases().items()]
    with ProcessPoolExecutor(3) as workers:
        found={k:(a,b,c,n) for shard_rows in workers.map(shard,[docs[i::3] for i in range(3)])
               for k,a,b,c,n in shard_rows}
    assert len(found)==len(docs)
    out=[found[d["id"]] for d in docs]
    a,b,c=(np.stack([q[i] for q in out]) for i in range(3))
    name_file=json.loads((root/"known-sentence-numerical-archives.json").read_text())
    path=root/"known-sentence-numerical-archives.npz"
    assert file_hash(path)==name_file["score_file_sha256"]
    name=name_file["best_by_macro_f1"]
    with np.load(path,allow_pickle=False) as o:
        np.testing.assert_array_equal(o["cases"],[d["id"] for d in docs])
        author=o[name]
    q={"freeze_sha256":file_hash(gate),"cases":len(out),"changed":sum(m[3] for m in out),
       "candidate":row_summary(a.sum(0)),"lexical":row_summary(b.sum(0)),"baseline":row_summary(c.sum(0)),
       "paired_to_baseline":paired(a,c),"paired_to_lexical":paired(a,b),
       "paired_to_aligned_archive":paired(a,author),"aligned_archive":row_summary(author.sum(0)),
       "aligned_archive_name":name}
    assert q["baseline"]==json.loads((root/"sentence-joint-qa.json").read_text())["scores"]
    source=root/(STEM+".json")
    if source.is_file() and "test" in (s:=json.loads(source.read_text()))["splits"]:
        assert all(q[k]==s["splits"]["test"][k] for k in ("candidate","lexical","baseline","changed","cases"))
    save_json(root/(STEM+"-qa.json"),q)
    print("Paired archive replay",q["cases"],q["candidate"]["macro_f1"],q["paired_to_baseline"]["macro_f1"],
          q["paired_to_aligned_archive"]["macro_f1"],flush=True)

if __name__=="__main__":
    main()
