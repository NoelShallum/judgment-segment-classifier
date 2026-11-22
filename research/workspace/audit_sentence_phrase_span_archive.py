"""Pair the new supplied-boundary span decoder with aligned author predictions."""
import csv
import json

import numpy as np
from safetensors.numpy import load_file

import sentence_phrase_sequence as seq
from hier_full import gold_cases
from sentence_document_graph_write import OUT,file_hash,save_json
from span_eval import counts,document,summarize
from raw_emitter_write import paired

P=seq.phrase
TAG="sentence-phrase-span-archive-audit"


def archive_counts(bank,predicted):
    official_path=P.RESULT/"archived-sentence-span-references.json"
    releases=json.loads(official_path.read_text())["scored_release_files"]
    top=max(releases,key=lambda k:releases[k]["scores"]["exact_labeled_spans"]["f1"])
    assert top=="previous_predicted_role"
    src=P.ROOT/"references/all-predictions/LegalSeg_InLegalBERT_Predictions/InLegalBert(i-1, label_p, i).csv"
    digest=file_hash(src)
    assert releases[top]["bytes_sha256"]==digest
    with src.open(newline="",encoding="utf-8-sig") as f:
        aligned=np.array([(int(r["true"]),int(r["pred"])) for r in csv.DictReader(f)])
    known=gold_cases()
    docs=[document(i,v) for i,v in known.items()]
    np.testing.assert_array_equal(bank[0],[d["id"] for d in docs])
    before=[]
    after=[]
    for a,b,d in zip(bank[1],bank[1][1:],docs):
        relevant=[row for row,text,_ in known[d["id"]] if text.strip()]
        np.testing.assert_array_equal(aligned[relevant,0],bank[2][a:b])
        cut=d["reference_units"]
        before.append(counts(d,cut,aligned[relevant,1]))
        after.append(counts(d,cut,predicted[a:b]))
    before,after=np.stack(before),np.stack(after)
    assert releases[top]["scores"]==summarize(before.sum(0))
    np.testing.assert_array_equal(after[:,49:],seq.span_tally(bank,predicted))
    return before,after,top,digest


def main():
    bank=P.dense.prepare("test")
    lex=P.cached_scores("test")
    f=P.RESULT/(seq.TAG+"-frozen.json")
    plan=json.loads(f.read_text())
    mode,g,b,guard=plan["rules"]["spans"]
    weights=OUT/(seq.TAG+".safetensors")
    assert plan["checkpoint_sha256"]==file_hash(weights)
    baseline,unary=seq.source_unary(bank,lex)
    edge=seq.edge_arrays(bank,lex,mode,load_file(str(weights)))
    predicted=seq.decode(seq.load_decoder(),bank[1],unary,baseline,edge,g,b,guard)
    before,after,top,digest=archive_counts(bank,predicted)
    scores=summarize(after.sum(0))
    previous=summarize(before.sum(0))
    result={"cases":len(bank[0]),"archive":top,"archive_csv_sha256":digest,"sequence_freeze_sha256":file_hash(f),
            "published_projection":previous,"edited_projection":scores,
            "paired_to_archive":paired(after,before),"char_and_sentence_span_tallies_equal":True}
    save_json(P.RESULT/(TAG+".json"),result)
    print("Exact labeled-span projections",previous["exact_labeled_spans"]["f1"],scores["exact_labeled_spans"]["f1"],
          result["paired_to_archive"],flush=True)


if __name__=="__main__":
    main()
