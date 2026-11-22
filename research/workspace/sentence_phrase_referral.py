"""Combine each frozen phrase write with a fixed BERT referral without a new search."""
import json

import numpy as np

import sentence_phrase_readout as phrase
import sentence_dense_referral_combine as referral
from sentence_document_graph_write import OUT,file_hash,load_file,paired,row_summary,save_json

TAG="sentence-phrase-referral"


def main():
    results=phrase.RESULT
    recipe=results/(phrase.TAG+"-frozen.json")
    known=json.loads(recipe.read_text())
    archive=results/"known-sentence-numerical-archives.npz"
    report=phrase.dense.read("known-sentence-numerical-archives")
    assert file_hash(archive)==report["score_file_sha256"]
    freeze=results/(TAG+"-frozen.json")
    save_json(freeze,{"phrase_freeze_sha256":file_hash(recipe),
                     "bert_freeze_sha256":file_hash(results/(referral.bert.TAG+"-frozen.json")),
                     "rule":"Use the frozen BERT decision on all referred rows. Apply the frozen phrase decision on other rows.",
                     "variants":list(phrase.ARMS[1:])})
    ans={"freeze_sha256":file_hash(freeze),"splits":{}}
    for split in ("select","confirm","holdout","test"):
        bank,_,bert,routed=referral.views(split)
        parent=phrase.dense.paired_confusion(bank,bert)
        rows=ans["splits"][split]={"cases":len(bank[0]),"bert":row_summary(parent.sum(0)),"arms":{}}
        lex=phrase.cached_scores(split)
        if split=="test":
            with np.load(archive,allow_pickle=False) as f:
                np.testing.assert_array_equal(f["cases"],bank[0])
                author=f[report["best_by_macro_f1"]]
            rows["author"]=row_summary(author.sum(0))
        for arm in phrase.ARMS[1:]:
            model=OUT/(phrase.TAG+f"-{phrase.ARMS.index(arm)}.safetensors")
            spec=known["arms"][arm]
            assert file_hash(model)==spec["weights_sha256"]
            opt=spec["choice"]
            score=phrase.dense.output(phrase.design(bank,lex,arm,opt[0]),load_file(str(model)))
            p=phrase.predict_rows(bank,score,opt)
            full=np.where(routed,bert,p)
            result=phrase.dense.paired_confusion(bank,full)
            q={"scores":row_summary(result.sum(0)),"paired_bert":paired(result,parent),
               "changed_vs_bert":int(np.count_nonzero(full!=bert))}
            if split=="test":
                q["paired_author"]=paired(result,author)
            rows["arms"][arm]=q
        print(split,{k:(d["scores"]["macro_f1"],d["scores"]["accuracy"]) for k,d in rows["arms"].items()},flush=True)
        save_json(results/(TAG+".json"),ans)


if __name__=="__main__":
    main()
