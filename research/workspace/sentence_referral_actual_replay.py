"""Compare frozen review routes using the same actual predictions at matched cost."""
import argparse
import csv
import json

import numpy as np
from safetensors.numpy import load_file

import compare_broad_edits_to_bert as broad
import sentence_minilm_whitened_write as mini
import sentence_pairwise_consensus as consensus
import sentence_pairwise_repair as repair
import sentence_phrase_sequence as seq
import sentence_referral_utility_write as utility
from hier_full import gold_cases
from sentence_document_graph_write import OUT,cm,file_hash,paired,row_summary,save_json

P=repair.P
TAG="sentence-referral-actual-replay"
SUBSET_BUDGETS=(.05,.10,.15)
ARCHIVES={
    "previous_predicted_role":"InLegalBert(i-1, label_p, i).csv",
    "two_previous":"InLegalBert(i-2, i-1, i)_predictions.csv",
}


def risk(split):
    parent,pair_w,pair_sha=consensus.inputs()
    ufile=repair.receipt(utility.TAG)
    cfg=json.loads(ufile.read_text())
    assert cfg["pairwise_freeze_sha256"]==pair_sha
    assert cfg["patch_sha256"]==file_hash(utility.MODEL)
    w=load_file(str(utility.MODEL))
    recipe=repair.load_recipe()
    bank,pred,conf,x=utility.data(split,parent,pair_w,recipe)
    evidence=utility.choices(x,conf,w)@np.array(cfg["pick"][1])
    return bank,pred,{"confidence":conf,"utility":evidence},file_hash(ufile)


def archive_labels(bank):
    meta=P.dense.read("known-sentence-numerical-archives")
    rows=gold_cases()
    np.testing.assert_array_equal(list(rows),bank[0])
    ordered=sorted(row for d in rows.values() for row in d)
    assert all(row[0]==i for i,row in enumerate(ordered))
    all_gold=[r[2] for r in ordered]
    selected=[r[0] for doc in rows.values() for r in doc if r[1].strip()]
    np.testing.assert_array_equal(bank[2],np.array(all_gold)[selected])
    out={}
    for name,filename in ARCHIVES.items():
        source=P.ROOT/"references"/"all-predictions"/"LegalSeg_InLegalBERT_Predictions"/filename
        sha=file_hash(source)
        assert sha==meta["archives"][name]["sha256"]
        with source.open(newline="",encoding="utf-8-sig") as f:
            records=list(csv.DictReader(f))
        np.testing.assert_array_equal([int(r["true"]) for r in records],all_gold)
        pred=np.array([int(r["pred"]) for r in records],np.int8)[selected]
        assert row_summary(cm(bank[2],pred))==meta["archives"][name]["scores"]
        out[name]=(pred,sha)
    return out


def scores(bank,base,review,positions):
    result=base.copy()
    result[positions]=review[positions]
    counts=P.dense.paired_confusion(bank,result)
    gold=bank[2]
    changed=result!=base
    return result,counts,{"labels":row_summary(counts.sum(0)),"referred":len(positions),
                         "changed":int(changed.sum()),"fixed":int((changed&(result==gold)).sum()),
                         "broken":int((changed&(base==gold)).sum()),
                         "referred_disagreement":int(np.count_nonzero(review[positions]!=base[positions])),
                         "spans":seq.span_info(seq.span_tally(bank,result))}


def report(bank,base,review,ranks,rate,available=None):
    positions=[]
    off=bank[1]
    for rank in ranks.values():
        if available is None:
            positions.append(utility.route(off,rank,rate))
        else:
            positions.append(np.concatenate([x[np.argsort(-rank[x],kind="stable")[:int(np.ceil(rate*(b-a)))]]
                                             for a,b,x in zip(off,off[1:],available)]))
    assert len(positions[0])==len(positions[1])
    old=P.dense.paired_confusion(bank,base)
    entries={}
    matrices=[]
    for key,p in zip(ranks,positions):
        _,counts,metrics=scores(bank,base,review,p)
        entries[key]={**metrics,"paired_base":paired(counts,old)}
        matrices.append(counts)
    return {"overlap":int(np.intersect1d(*positions,assume_unique=True).size),
            "paired_utility_minus_confidence":paired(matrices[1],matrices[0]),"routes":entries}


def external():
    bank,base,ranks,src=risk("test")
    overlay_plan=repair.receipt(broad.overlay.TAG)
    policy=json.loads(overlay_plan.read_text())
    origin=broad.B.read(broad.B.TAG+"-frozen")
    bert=broad.bert_reference("test",bank,base,origin,policy["arms"]["BERT only"]["choice"])
    protected=bert!=base
    out={"router_freeze_sha256":src,"cases":len(bank[0]),"sentences":len(base),
         "baseline":row_summary(cm(bank[2],base)),"reviewers":{},"local_overlay_freeze_sha256":file_hash(overlay_plan),
         "local_overlay_baseline":row_summary(cm(bank[2],bert)),"local_overlay_protected_rows":int(protected.sum())}
    for name,(pred,digest) in archive_labels(bank).items():
        record={"source_sha256":digest,"standalone":row_summary(cm(bank[2],pred)),"rates":{}}
        for rate in utility.BUDGETS:
            record["rates"][f"{rate:.0%}"]=report(bank,base,pred,ranks,rate)
        protective=np.where(protected,bert,pred)
        record["protect_local_overlay_at_20_percent"]=report(bank,bert,protective,ranks,.20)
        out["reviewers"][name]=record
    save_json(P.RESULT/(TAG+"-archive.json"),out)
    for name,r in out["reviewers"].items():
        print(name,{k:[v["routes"][route]["labels"]["macro_f1"] for route in ranks] for k,v in r["rates"].items()},flush=True)


def cached(splits):
    cfg,origin,directory=mini.source()
    main_freeze=repair.receipt(mini.TAG+"-active")
    frozen=json.loads(main_freeze.read_text())
    assert origin==frozen["miniLM_source_freeze_sha256"]
    ref=P.dense.read(mini.TAG)
    assert ref["freeze_sha256"]==file_hash(repair.receipt(mini.TAG))
    parent,pair_w,pair_sha=consensus.inputs()
    assert pair_sha==frozen["pair_freeze_sha256"]
    recipe=repair.load_recipe()
    key=max(frozen["arms"],key=lambda n:frozen["arms"][n]["selection"])
    review_cfg=frozen["arms"][key]
    checkpoint=OUT/(mini.TAG+f"-active-{mini.ARMS.index(key)}.safetensors")
    assert file_hash(checkpoint)==review_cfg["sha256"]
    w=load_file(str(checkpoint))
    out={"reviewer_freeze_sha256":file_hash(main_freeze),"reviewer":key,"splits":{}}
    for s in splits:
        bank,base,ranks,src=risk(s)
        out["router_freeze_sha256"]=src
        view=mini.study(s,directory,recipe,pair_w,parent,trusted=ref["splits"][s]["layout_sha256"])
        np.testing.assert_array_equal(view[0][0],bank[0])
        np.testing.assert_array_equal(view[4],base)
        predicted=pair_prob(view,key,w,review_cfg["settings"][-2:])
        expected=P.dense.read(mini.TAG+"-active")["splits"][s]["arms"][key]["scores"]
        assert row_summary(cm(bank[2],predicted))==expected
        available=[q+a for a,q in zip(bank[1],view[1])]
        result=out["splits"][s]={"cases":len(bank[0]),"cached_positions":len(view[3]),
                                 "baseline":row_summary(cm(bank[2],base)),"rates":{}}
        for rate in SUBSET_BUDGETS:
            result["rates"][f"{rate:.0%}"]=report(bank,base,predicted,ranks,rate,available)
        save_json(P.RESULT/(TAG+"-minilm.json"),out)
        print(s,{k:[v["routes"][route]["labels"]["macro_f1"] for route in ranks]
                 for k,v in result["rates"].items()},flush=True)


def pair_prob(view,arm,w,setting):
    prob=repair.fusion.bert.response(mini.features(view,arm,w),w)
    return mini.edit(view,prob,*setting)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("kind",choices=("archive","minilm"))
    p.add_argument("splits",nargs="*")
    a=p.parse_args()
    if a.kind=="archive":
        external()
    else:
        cached(a.splits or ("select","confirm","test"))
