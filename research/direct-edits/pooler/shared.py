"""Evaluate frozen pooler edits in the actual 20-percent confidence route."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

import experiment as study
from experiment import HERE, ROOT
import mixedbread_referral_overlay as overlay
import predict_pairwise_judgment as pair
import sentence_bert_referral_write as bert
import sentence_phrase_sequence as sequence
import sentence_referral_utility_write as utility
from openvino_worker import embed_ids, get_engine
from program_trial import TO_ID


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size",type=int,choices=(16,48),default=16)
    parser.add_argument("--device",default="GPU")
    args=parser.parse_args()
    manifest=HERE.parent/"shared-confirmation.json"
    manifest_sha=study.file_hash(manifest)
    assert manifest_sha=="86c94b34849ec0afa6c58fa3dac911b068465cd1dd855aa5c3ce953471797957"
    cohort=json.loads(manifest.read_text())
    ids=cohort["extended_validation_judgment_ids"][:args.size]
    split=ROOT/"references/official/val.csv"
    assert study.file_hash(split)==cohort["official_csv_sha256"]
    frozen_path=HERE/"freeze.json"
    frozen=json.loads(frozen_path.read_text())
    assert set(ids).isdisjoint(frozen["fit_ids"])
    digest=study.file_hash(frozen_path)
    rows=defaultdict(list)
    with split.open(newline="",encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["Index"] in ids and r["Text"].strip():
                rows[r["Index"]].append((r["Text"],TO_ID[r["Label"]]))
    directory=HERE/"shared-cases"
    directory.mkdir(exist_ok=True)
    plan=frozen["arms"]["document"]
    head=study.params()
    patches={k:load_file(str(HERE/rec["file"])) for k,rec in plan["matrices"].items()}
    config=bert.read("mixedbread-referral-overlay-frozen")
    rule=config["arms"]["BERT only"]["choice"]
    reclassify=pair.predictor("direct")
    engine=None
    tokenizer=None
    results=[]
    for key in ids:
        path=directory/(key+".json")
        if path.is_file():
            case=json.loads(path.read_text())
            assert case["freeze_sha256"]==digest and case["manifest_sha256"]==manifest_sha
            if case.get("precision")=="float64-pooler-v1":
                results.append(case)
                continue
        if engine is None:
            engine,tokenizer=get_engine(args.device)
        candidate={}
        def revise(bank,lex,unary,initial,item,h,model,cfg,vocab):
            base=reclassify(bank,lex,unary,initial)
            confidence=np.exp(unary-unary.max(1,keepdims=True))
            confidence/=confidence.sum(1,keepdims=True)
            positions=utility.route(bank[1],1-confidence[np.arange(len(base)),base],.20)
            live=bert.case_view(item,h,model,cfg,vocab,positions)
            np.testing.assert_array_equal(live[2],bank[3])
            windows=bert.pack_ids(tokenizer,[live],bert.MODES[1])
            vec=embed_ids(engine,windows,max_length=bert.LIMIT)
            for arm in ("unchanged","targeted","shuffled"):
                feat = vec if arm=="unchanged" else study.activation(vec,patches[arm],plan["gain"])
                score=bert.response(bert.design([live],feat,"hidden",head),head)
                g,cut,veto=rule
                pred=overlay.edit(base,positions,live[5],score,g,cut,live[2][positions] if veto else None)
                candidate[arm]=pred.astype(int).tolist()
            candidate["positions"]=positions.tolist()
            return np.asarray(candidate["unchanged"])
        text,gold=zip(*rows[key])
        out=sequence.run_input(list(text),False,"spans",revise_with_context=revise)
        assert out["labels"]==[study.LABELS[i] for i in candidate["unchanged"]]
        case={"id":key,"freeze_sha256":digest,"manifest_sha256":manifest_sha,"rows":len(gold),"predictions":candidate,
              "precision":"float64-pooler-v1"}
        part=path.with_suffix(".part")
        part.write_text(json.dumps(case))
        part.replace(path)
        print("shared judgment",key,len(gold),len(candidate["positions"]),flush=True)
        results.append(case)
    output={"manifest_sha256":manifest_sha,"pooler_freeze_sha256":digest,"official_csv_sha256":cohort["official_csv_sha256"],
            "sample":"All nonblank sentences from the first %d uniformly hash-ranked official validation cases."%len(results),
            "overlap_with_historical_pooler_gain_selection":sorted(set(ids)&set(frozen["select_ids"])),
            "ids":ids,"arms":{}}
    allgold=np.concatenate([[y for _,y in rows[d]] for d in ids])
    lengths=[len(rows[d]) for d in ids]
    counts={}
    predictions={}
    for arm in ("unchanged","targeted","shuffled"):
        pred=np.concatenate([r["predictions"][arm] for r in results])
        predictions[arm]=pred
        cm=study.cms(allgold,pred,lengths)
        counts[arm]=cm
        spans=sum((sequence.shape_metrics(np.asarray([0,len(rows[d])]),np.asarray([y for _,y in rows[d]]),
                        np.asarray(case["predictions"][arm])) for d,case in zip(ids,results)), start=np.zeros(6,dtype=int))
        info=study.summary(cm.sum(0))
        info.update(sequence.span_info(spans[None,:]))
        if arm!="unchanged":
            base=predictions["unchanged"]
            changed=base!=pred
            info.update(delta_vs_incumbent_ci95=study.interval(counts["unchanged"],cm), corrected=int(np.sum(changed&(allgold==pred))),
                        broken=int(np.sum(changed&(allgold==base))),changed=int(changed.sum()))
        output["arms"][arm]=info
    output["rows"]=len(allgold)
    output["reviewed"]=sum(len(c["predictions"]["positions"]) for c in results)
    study.write("shared-"+str(args.size)+".json",output)
    print("aggregate",output["rows"],{a:val["macro_f1"] for a,val in output["arms"].items()},flush=True)


if __name__=="__main__":
    main()
