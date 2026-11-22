import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
def load(name): return json.loads((HERE/name).read_text())
def metrics(x): return {k:x[k] for k in ("macro_f1","accuracy")}
def write():
    out={"task":"One of seven human roles per existing sentence, entire judgment context","source":"L-NLProc/LegalSeg_Hier_BiLSTM-CRF","base_parameters":16816110,
    "official_test_scored_in_this_lane":(HERE/"uniform-retention-test-score.json").exists(),"gradient_steps":0,"training_counts":load("counts.json"),"trials":[]}
    shared=load("../shared-confirmation.json")["extended_validation_judgment_ids"]
    for p in sorted(HERE.glob("select-*.json")):
        if not p.stem.split("-")[-1].isdigit(): continue
        obj=load(p.name)
        arm={k:dict(**metrics(v),changed_word_rows=v["changed_rows"],changed_float_entries=v["changed_entries"],L2=v["delta_L2"],tensor_sha256=v["written_sha256"],paired_macro_f1_95=v["paired_bootstrap95"]) for k,v in obj.items() if isinstance(v,dict) and "changes" in v}
        out["trials"].append(dict(trial="Direct token-to-role writes in donor word embeddings",cohort=p.stem,rows=obj["rows"],baseline=metrics(obj["baseline"]),arms=arm,
            controls="Unchanged donor; sign reversal; frequency-binned permutation of lexical identities (subset varies by cohort).",changed_keys=["sent_encoder.emb.weight"],result_path=p.name,
            exclusions=sorted(set(obj["cases"])&set(shared))))
    for p in sorted(HERE.glob("native-*-240-71.json")):
        obj=load(p.name)
        if "weights_sha256" not in obj: continue
        out["trials"].append(dict(trial=p.stem,cohort="Historical selection, excluded new 48",rows=obj["rows_selection"],judgments=len(obj["select_cases"]),metrics=metrics(obj["scores"][obj["chosen"]]),changed_keys=list(obj["tensors"]),
              changed_float_entries=sum(t["changed_entries"] for t in obj["tensors"].values()),result_path=p.name,checkpoint_sha256=obj["weights_sha256"],controls="Unchanged native hierarchy and alternative closed-form head-only checkpoint",gradient_steps=0))
    for p in sorted(HERE.glob("replay-*.json")):
        obj=load(p.name);out["trials"].append(dict(trial=p.stem,cohort=obj["split"],rows=obj["rows"],judgments=len(obj["cases"]),metrics=metrics(obj["edited"]),baseline=metrics(obj["base"]),changed=obj["changed"],corrected=obj["corrected"],harmed=obj["harmed"],intervals=obj["paired_judgment_95_macro_F1_accuracy"],result_path=p.name,checkpoint_sha256=obj["tensor_checkpoint_sha256"],controls="Native donor on identical sentences",changed_keys=list(load(p.stem.removeprefix("replay-").split("-"+obj["split"]+"-")[0]+".json")["tensors"])))
    for p in sorted(HERE.glob("fused-*.json")):
        if p.stem.endswith("freeze"):continue
        obj=load(p.name);first=next(iter(obj.values()));out["trials"].append(dict(trial=p.stem,cohort=p.stem.split("-")[-1],rows=first["rows"],judgments=len(first["cases"]),baseline=metrics(first["baseline"]),arms={k:dict(**metrics(v),changed=v["changed"],corrected=v["corrected"],harmed=v["harmed"],paired95=v["paired95_f1_accuracy"]) for k,v in obj.items()},result_path=p.name,controls="Exact frozen 20% InLegalBERT reviewer and native head-only evidence gate",changed_keys="No additional tensor in fusion; native weights described in linked checkpoint and explicit conditional decision rule"))
    if (HERE/"widened-frozen.json").exists():
        obj=load("widened-frozen.json");out["trials"].append(dict(trial="Function-preserving widening of both native directional LSTMs",cohort="Historical selection, excluded new 48",rows=obj["select_rows"],judgments=len(obj["select_cases"]),metrics=metrics(obj["scores"][obj["chosen"]]),parameters=obj["model_parameters"],result_path="widened-frozen.json",tensor_ledger="widened-tensors.json",checkpoint_sha256=obj["weights_sha256"],changed_keys=[k for k in obj["shapes"] if k!="crf.transitions"],controls="All incumbent units preserved and train-fitted seven-way readout-only arm. Native end-to-end forward parity included."))
    for p in sorted(HERE.glob("widened-*.json")):
        if p.name in ("widened-frozen.json","widened-tensors.json"): continue
        d=load(p.name)
        out["trials"].append(dict(trial=p.stem,cohort=p.stem.split("-")[1],rows=d["rows"],judgments=len(d["cases"]),arms={k:metrics(v) for k,v in d["metrics"].items()},controls="Native donor, head-only control and current 20% reviewer when available",changed_keys="See native 19,234,190-parameter checkpoint in widened-frozen.json",paired=d["paired"],result_path=p.name))
    for p in sorted(HERE.glob("production-word-table-*.json")):
        d=load(p.name)
        out["trials"].append(dict(trial=p.stem,cohort=d["splits"],rows=d["rows"],judgments=len(d["keys"]),controls="Original production pairwise model on identical input plus a frequency-matched permutation; 20% reviewer included where aligned",changed_keys=[d["tensor"]],arms={k:dict(**metrics(v),changed=v["changed"],corrected=v["corrected"],harmed=v["harmed"]) for k,v in d["arms"].items()},incumbent_20pct={k:d["incumbent_20pct"][k] for k in ("macro_f1","accuracy")} if "incumbent_20pct" in d else None,result_path=p.name))
    for p in sorted(HERE.glob("conditional-widened-*.json")):
        if p.stem.endswith("freeze"):continue
        d=load(p.name);first=d["widened"];out["trials"].append(dict(trial=p.stem,cohort=p.stem.split("-")[-1],rows=first["rows"],judgments=len(first["cases"]),baseline=metrics(first["baseline"]),arms={k:dict(**metrics(v),changed=v["changed"],corrected=v["corrected"],harmed=v["harmed"]) for k,v in d.items()},controls="Same residual arbiter without widened dictionary states; full 20% incumbent",changed_keys=["readout.weight","readout.mean","readout.scale","widened.weight","widened.mean","widened.scale"],result_path=p.name))
    for p in sorted(HERE.glob("full-review-*-*.json")):
        d=load(p.name);out["trials"].append(dict(trial=p.stem,cohort=d.get("cohort","Shared validation"),rows=d["rows"],judgments=len(d["keys"]),baseline=metrics(d["incumbent"]),arms={k:dict(**metrics(v),changed=v["changed"],corrected=v["corrected"],harmed=v["harmed"],paired95=v["paired95"]) for k,v in d["arms"].items()},controls="Identical 20% reviewer with completely recalculated route; unchanged donor and matched vocabulary shuffling",changed_keys=(d["tensor"] if isinstance(d["tensor"],list) else [d["tensor"]]),result_path=p.name))
    for p in sorted(HERE.glob("forget-*.json")):
        if p.name=="forget-frozen.json":continue
        d=load(p.name)
        out["trials"].append(dict(trial="Program document LSTM memory retention – "+d["split"],cohort=d["split"],rows=d["rows"],judgments=len(d["cases"]),arms={k:dict(**metrics(v),changed=v["changed"],corrected=v["correct"],harmed=v["broken"],paired95=v["ci95"]) for k,v in d["arms"].items()},controls="Zero edit, frequency-independent uniform retention offset, magnitude-preserving shuffled cell assignments and negative gain (selection)",changed_keys=d["tensors"],result_path=p.name))
    for p in sorted(HERE.glob("uniform-retention-*-score.json")):
        d=load(p.name);out["trials"].append(dict(trial=p.stem,cohort=d["split"],rows=d["rows"],judgments=len(d["cases"]),arms={k:metrics(v) for k,v in d["arms"].items()},controls="Unedited donor and head-only for all, direction-shuffled and equal-norm gate-shuffled on separate validation",changed_keys=["emitter.lstm.bias_ih_l0[100:200]","emitter.lstm.bias_ih_l0_reverse[100:200]"],changed_float_entries=200,score_gate=d["gate_passed"],paired=d["paired"],result_path=p.name))
    if (HERE/"uniform-span-diagnostics.json").exists():out["posthoc_fixed_sentence_span_metrics"]={"result":"uniform-span-diagnostics.json","official_test":load("uniform-span-diagnostics.json")["test"]["arms"]}
    if (HERE/"uniform-ledger.json").exists():out["frozen_200_float_tensor_ledger"]="uniform-ledger.json"
    folder=HERE/"scale"
    if (folder/"plan.json").exists():
        plan=load("scale/plan.json")
        out["corpus_scale"]={"plan":"scale/plan.json","fixed_prefixes":plan["fitted_prefixes"]+[plan["final_refit"]],"validation_reserved":len(plan["validation"]),"train_side_calibration":len(plan["calibration"]),"primary_class_reweighting":plan["grid"]["class_power"],"full_inverse_frequency_secondary":"scale/equal-role-control-plan.json","official_test_scored_for_scale":False}
        shards=[tuple(map(int,p.stem.split("-"))) for p in (folder/"shards").glob("*.npz")]
        out["corpus_scale"]["unique_completed_fit_judgments"]=sum(hi-lo for lo,hi in shards)
        if (folder/"progress.json").exists():out["corpus_scale"]["progress"]=load("scale/progress.json")
        if (folder/"reverse-progress.json").exists():out["corpus_scale"]["reverse_worker"]=load("scale/reverse-progress.json")
        if (folder/"streamed-qc.json").exists():out["corpus_scale"]["streamed_algebra_and_decoder_QA"]="scale/streamed-qc.json"
        if (folder/"dense-words-benchmark.json").exists():out["corpus_scale"]["speed_probe"]={"result":"scale/dense-words-benchmark.json","used_in_main_job":False}
        if (folder/"lda-control-plan.json").exists():out["corpus_scale"]["secondary_discriminant_control"]="scale/lda-control-plan.json"
        if (folder/"quadratic/plan.json").exists():out["corpus_scale"]["exploratory_quadratic"]={"plan":"scale/quadratic/plan.json","reserved_separately":len(load("scale/quadratic/plan.json")["cases"])}
        if (folder/"quadratic/freeze.json").exists():
            v=load("scale/quadratic/freeze.json");out["corpus_scale"]["exploratory_quadratic"].update(chosen=v["chosen"],advanced_from_train=v["train_side_advance"],extra_parameters=v["extra_parameters"],freeze="scale/quadratic/freeze.json")
        if (folder/"quadratic/reserved48.json").exists():out["trials"].append(dict(trial="Covariance-based square-and-sum head",result_path="scale/quadratic/reserved48.json",**load("scale/quadratic/reserved48.json")))
        if (folder/"train-side-baselines.json").exists():out["corpus_scale"]["same_calibration_baselines"]={k:metrics(v) for k,v in load("scale/train-side-baselines.json")["scores"].items()}
        for p in sorted(folder.glob("train-side-*.json")):
            if not p.stem.split("-")[-1].isdigit():continue
            z=json.loads(p.read_text());out["trials"].append(dict(trial="Native closed-form readout at corpus scale",training_judgments=z["training_judgments"],training_rows=z["training_rows"],cohort="Held-out train-side calibration, donor pretrained by authors",rows=z["train_calibration_rows"],judgments=z["train_calibration_judgments"],arms={k:dict(**metrics(v["metrics"]),settings=v["settings"]) for k,v in z["arms"].items()},controls="Native donor states and fixed timescale-edited states, identical training case ordering",changed_keys=["emitter.hidden2tag.weight","emitter.hidden2tag.bias"],result_path="scale/"+p.name))
        if (folder/"reserved72.json").exists():
            z=load("scale/reserved72.json");out["trials"].append(dict(trial="Corpus-scale direct native readout",cohort="Separately reserved 72 validation judgments",rows=z["rows"],judgments=len(z["cases"]),arms={k:metrics(v) for k,v in z["models"].items()},selected=z["chosen"],gate=z["gate_passed"],matched_intervals=z["paired"],changed_keys=["emitter.hidden2tag.weight","emitter.hidden2tag.bias"],result_path="scale/reserved72.json"))
        if (folder/"reserved72-audit.json").exists():
            proof=load("scale/reserved72-audit.json")
            out["corpus_scale"]["physical_audit"]={"path":"scale/reserved72-audit.json","native_parity":proof["native_parity"],
                "changed_entries":sum(x["changed_values"] for x in proof["tensors"].values()),
                "exact_span_F1_with_supplied_cuts":proof["span_metrics"]["retention-4984"]["exact_labeled_span_f1"]}
        if (folder/"shared48-audit.json").exists():
            proof=load("scale/shared48-audit.json")
            out["corpus_scale"]["matched_incumbent"]={"path":"scale/shared48-audit.json","rows":proof["rows"],
                "judgments":proof["judgments"],"scores":{k:metrics(v) for k,v in proof["scores"].items()}}
    if (HERE/"chain-program/reserved96.json").exists():
        r=load("chain-program/reserved96.json")
        f=load("chain-program/freeze.json")
        out["trials"].append(dict(trial="Native emitter fit to exact full-document CRF marginals",
            cohort="Separately reserved 96 validation judgments", rows=r["rows"], judgments=len(r["cases"]),
            training_judgments=f["cases_fit"], training_rows=f["rows_fit"], selected=r["chosen"],
            arms={k:dict(**metrics(v), exact_span_f1=v["spans"]["exact_labeled_span_f1"]) for k,v in r["scores"].items()},
            changed_tensors=f["checkpoints"][r["chosen"]]["changes"], matched_intervals=r["paired"], native_parity=r["cpu_native_parity"],
            result_path="chain-program/reserved96.json", official_test_evaluated=False))
    if (HERE/"case-head-program/train-calibration-2400.json").exists():
        data=load("case-head-program/train-calibration-2400.json")
        out["trials"].append(dict(trial="Case-conditioned writes to both native emitter tensors",
            cohort="192 training-calibration judgments", rows=data["calibration_rows"], judgments=data["calibration_cases"],
            training_judgments=data["fit_cases"], training_rows=data["fit_rows"], baseline=metrics(data["baseline"]),
            arms={k:metrics(v) for k,v in data["settings"].items()},
            validation_opened=(HERE/"case-head-program/reserved96.json").exists(),
            result_path="case-head-program/train-calibration-2400.json", official_test_evaluated=False))
    if (HERE/"case-head-program/reserved96.json").exists():
        data=load("case-head-program/reserved96.json")
        out["trials"].append(dict(trial="Case-conditioned writes to both native emitter tensors",
            cohort="Separately reserved 96 validation judgments", rows=data["rows"], judgments=len(data["cases"]),
            arms={k:dict(**metrics(v), exact_span_f1=v["spans"]["exact_labeled_span_f1"]) for k,v in data["scores"].items()},
            matched_intervals=data["paired"], native_parity_rows=data["exact_native_parity_rows"],
            case_weight_file_sha256=data["case_weight_file_sha256"],
            result_path="case-head-program/reserved96.json", official_test_evaluated=False))
    if (HERE/"author-fold4-audit.json").exists():
        data=load("author-fold4-audit.json")
        out["archived_fold4"]={"path":"author-fold4-audit.json","rows":data["sentences"],"judgments":data["judgements"],
            "scores":metrics(data["scores"]),
            "gold_role_sequence_matches":{k:v["exact_role_sequence_matches"] for k,v in data["source_matches"].items()}}
    temp=HERE/"lane-summary.part";temp.write_text(json.dumps(out,indent=2)+"\n");temp.replace(HERE/"lane-summary.json")
if __name__=="__main__":write()
