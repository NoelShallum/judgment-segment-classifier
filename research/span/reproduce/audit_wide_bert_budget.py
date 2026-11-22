"""Measure the actual classification trade-off at nested InLegalBERT review budgets."""
import argparse
import json

import numpy as np
from safetensors.numpy import load_file

import mixedbread_referral_overlay as overlay
import sentence_bert_referral_write as bert
import sentence_pairwise_repair as repair
import sentence_referral_actual_replay as replay
import sentence_referral_utility_write as utility
import sentence_wide_bert_benchmark as wide
from sentence_document_graph_write import OUT, cm, file_hash, paired, row_summary, save_json

P=repair.P
TAG="sentence-wide-bert-budget"
RATES=(.03,.05,.075,.10,.15,.20)


def study(split,committed):
    core=committed["method"]
    ref=committed["splits"][split]
    policy_file=repair.receipt(overlay.TAG)
    assert core["local_overlay_sha256"]==file_hash(policy_file)
    policy=json.loads(policy_file.read_text())
    cfg=bert.read(bert.TAG+"-frozen")
    weights=OUT/(bert.TAG+"-2.safetensors")
    assert file_hash(weights)==cfg["checkpoints"][overlay.BERT]
    assert core["bert_freeze_sha256"]==file_hash(repair.receipt(bert.TAG))
    bank,base,ranks,risk_hash=replay.risk(split)
    assert risk_hash==core["router_freeze_sha256"]
    assert row_summary(cm(bank[2],base))==ref["baseline"]
    union=np.union1d(*(utility.route(bank[1],risk,wide.RATE) for risk in ranks.values()))
    rows=wide.feature_bank(bank,split,union)
    cached,receipt=wide.encode(rows,split,ref["embedding_provenance"]["device"],risk_hash)
    assert receipt==ref["embedding_provenance"]
    weights=load_file(str(weights))
    hidden=bert.design(rows,cached,cfg["frozen_modes"][overlay.BERT][0],weights)
    prob=bert.response(hidden,weights)
    local_prior=np.concatenate([r[5] for r in rows])
    initial=np.concatenate([r[2][r[3]] for r in rows])
    g,t,k=policy["arms"]["BERT only"]["choice"]
    review=overlay.edit(base,union,local_prior,prob,g,t,initial if k else None)
    return bank,base,ranks,review,receipt


def policies(bank,base,ranks,review):
    results={}
    for name,rank in ranks.items():
        previous=None
        variants={}
        for rate in RATES:
            places=utility.route(bank[1],rank,rate)
            if previous is not None:
                assert np.isin(previous,places).all()
            previous=places
            _,counts,data=replay.scores(bank,base,review,places)
            variants[rate]=(counts,data)
        final=variants[.20][0]
        old=P.dense.paired_confusion(bank,base)
        results[name]={str(r):{**record,"paired_to_base":paired(c,old),"paired_to_full":paired(c,final)}
                       for r,(c,record) in variants.items()}
    return results


def select(results,base):
    picks={}
    for key,curve in results.items():
        target=curve["0.2"]["labels"]["macro_f1"]
        gain=target-base["macro_f1"]
        minimum=base["macro_f1"]+.90*max(gain,0)
        # Choose the least review cost meeting the predeclared retention target.
        satisfying=[r for r in RATES if curve[str(r)]["labels"]["macro_f1"]>=minimum
                    and curve[str(r)]["labels"]["accuracy"]>=base["accuracy"]-.001]
        picks[key]=satisfying[0] if satisfying else None
    return picks


def main(splits):
    source=P.RESULT/(wide.TAG+".json")
    committed=json.loads(source.read_text())
    allowed=set(committed["splits"])
    assert "select" in allowed
    assert all(k in allowed for k in splits),(splits,allowed)
    cases=list(dict.fromkeys(("select",*splits)))
    result={"source_method":committed["method"],"gain_fraction_for_selection":.90,"splits":{}}
    destination=P.RESULT/(TAG+".json")
    if destination.is_file():
        previous=json.loads(destination.read_text())
        assert previous["source_method"]==result["source_method"]
        assert previous["gain_fraction_for_selection"]==result["gain_fraction_for_selection"]
        result=previous
    for split in cases:
        old=result["splits"].get(split)
        if old is not None:
            upstream=committed["splits"][split]
            for key in ("input_sha256","embedding_sha256"):
                assert old[key]==upstream["embedding_provenance"][key]
            for route,curve in old["routes"].items():
                assert curve["0.2"]["labels"]==upstream["comparison"]["routes"][route]["labels"]
            if split=="select":
                assert select(old["routes"],old["baseline"])==result["development_chosen_rates"]
            print(split,"Reused verified budget receipt",flush=True)
            continue
        bank,base,ranks,review,receipt=study(split,committed)
        out=policies(bank,base,ranks,review)
        baseline=row_summary(cm(bank[2],base))
        for name in out:
            assert out[name]["0.2"]["labels"]==committed["splits"][split]["comparison"]["routes"][name]["labels"]
        if split=="select":
            result["development_chosen_rates"]=select(out,baseline)
        picked={k:(baseline if val is None else out[k][str(val)]["labels"])
                for k,val in result["development_chosen_rates"].items()}
        result["splits"][split]={"cases":len(bank[0]),"baseline":baseline,"routes":out,"chosen":picked,
                                 "input_sha256":receipt["input_sha256"],"embedding_sha256":receipt["embedding_sha256"]}
        save_json(destination,result)
        print(split,result["development_chosen_rates"],
              {k:[round(d["labels"]["macro_f1"],5) for d in q.values()] for k,q in out.items()},flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("splits",nargs="*")
    args=parser.parse_args()
    main(args.splits or ("confirm",))
