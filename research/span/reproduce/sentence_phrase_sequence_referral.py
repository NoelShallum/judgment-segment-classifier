"""Compose the frozen span-focused sequence edit with BERT and issue-protected anchors."""
import json

import numpy as np
from safetensors.numpy import load_file

import sentence_phrase_sequence as seq
import sentence_dense_referral_combine as fusion
from sentence_document_graph_write import OUT,file_hash,paired,row_summary,save_json

P=seq.phrase
TAG="sentence-phrase-sequence-referral"


def main():
    source=P.RESULT/(seq.TAG+"-frozen.json")
    seq_plan=json.loads(source.read_text())
    mode,gain,bonus,guard=seq_plan["rules"]["spans"]
    assert mode!="none"
    path=OUT/(seq.TAG+".safetensors")
    assert file_hash(path)==seq_plan["checkpoint_sha256"]
    w=load_file(str(path))
    dense_plan=P.dense.read(P.dense.TAG+"-frozen")
    engine=seq.load_decoder()
    freeze=P.RESULT/(TAG+"-frozen.json")
    save_json(freeze,{"source_sequence_sha256":file_hash(source),
       "dense_freeze_sha256":file_hash(P.RESULT/(P.dense.TAG+"-frozen.json")),
       "bert_freeze_sha256":file_hash(P.RESULT/(fusion.bert.TAG+"-frozen.json")),
       "rule":"Use the frozen span decoder and target-phrase scores anchored to each incumbent. Protect every BERT referral and all issue membership assignments."})
    output={"freeze_sha256":file_hash(freeze),"splits":{}}
    archive=P.RESULT/"known-sentence-numerical-archives.npz"
    aligned=P.dense.read("known-sentence-numerical-archives")
    assert file_hash(archive)==aligned["score_file_sha256"]
    for split in ("select","confirm","holdout","test"):
        bank,_,bert,routed=fusion.views(split)
        lex=P.cached_scores(split)
        _,emissions=seq.source_unary(bank,lex)
        offsets=np.ascontiguousarray(bank[1],dtype=np.int64)
        links=seq.edge_arrays(bank,lex,mode,w)
        proposed=fusion.readout(bank,"nonlinear",dense_plan["arms"]["nonlinear"])
        hybrid=np.where(routed,bert,P.dense.preserve_issues(bank[3],proposed))
        if split in ("holdout","test"):
            expected=P.dense.read("sentence-dense-issue-guard")["splits"][split]["combined"]["scores"]
            assert row_summary(P.dense.paired_confusion(bank,hybrid).sum(0))==expected
        q=output["splits"][split]={"cases":len(bank[0]),"arms":{}}
        if split=="test":
            with np.load(archive,allow_pickle=False) as a:
                np.testing.assert_array_equal(a["cases"],bank[0])
                author=a[aligned["best_by_macro_f1"]]
        for name,base in (("BERT referral",bert),("issue-protected hybrid",hybrid)):
            x=emissions.copy()
            i=np.arange(len(base))
            x[i,base]=np.maximum(x[i,base],x.max(1)+.005)
            allowed=(base[:,None]==2)==(np.arange(7)[None,:]==2)
            allowed[routed]=(base[:,None]==np.arange(7)[None,:])[routed]
            x[~allowed]=-1.e8
            y=seq.decode(engine,offsets,x,base,links,gain,bonus,guard)
            np.testing.assert_array_equal(y[routed],base[routed])
            np.testing.assert_array_equal(y==2,base==2)
            mat=P.dense.paired_confusion(bank,y)
            old=P.dense.paired_confusion(bank,base)
            sp,prev=seq.span_tally(bank,y),seq.span_tally(bank,base)
            trial={"labels":row_summary(mat.sum(0)),"baseline":row_summary(old.sum(0)),
                   "spans":seq.span_info(sp),"baseline_spans":seq.span_info(prev),
                   "paired_labels":paired(mat,old),"paired_spans":seq.span_intervals(sp,prev),
                   "changed":int(np.count_nonzero(y!=base))}
            if split=="test":
                trial["paired_author"]=paired(mat,author)
            q["arms"][name]=trial
        print(split,{k:(v["baseline"]["macro_f1"],v["labels"]["macro_f1"],v["spans"]["exact_labeled_span_f1"])
                    for k,v in q["arms"].items()},flush=True)
        save_json(P.RESULT/(TAG+".json"),output)


if __name__=="__main__":
    main()
