from pathlib import Path
import json
import os
ROOT=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text())
def stat(s):return {"macro_f1_seven":s["macro_f1_7"],"accuracy":s["accuracy"],"support_by_role":s["true_counts"],"role_f1":s["role_f1"],"rows":s["rows"]}
def sumrun():
    model=read(ROOT/"model/revision-receipt.json")
    rec={"scope":"UltraBERT seven-role human annotation trials. All training/evaluation pilot figures below use case-disjoint splits of official train. Original source test untouched.",
         "model":{"repo":model["repo"],"revision":model["revision"],"model_sha256":next(t["sha256"] for t in model["files"] if t["file"]=="model.safetensors"),
                  "backbone_params":151032576,"encoder_fp32_parameter_bytes":604130304,"seven_class_linear_head_params_with_bias":5383,"rank6_factor_fp32_entries":10752,"rank6_factor_fp32_bytes":43008,"encoder_layers":24,"checkpoint_full_parameters":189960960,"license":model["license"]},
         "metric":"unweighted mean of seven pooled-sentence class F1s; absent class gets a zero. Accuracy is a separate pooled proportion.",
         "splits":{"pilot":"fit/edit use source-label-balanced anchor snippets from separate cases; selection and confirmation use label-blind whole cases limited to 20–65 nonblank sentences. The second sample nests the initial pilot.",
                   "shared_validation":"16 frozen official val judgments; every nonblank sentence. Independent train-split-only candidate selection."},
         "inputs":"Native FP32 encoder; contiguous text, per-row membership and newline separators, 384 text tokens per block; long rows use multiple blocks.",
         "role_order":["None","Facts","Issue","Arguments of Petitioner","Arguments of Respondent","Reasoning","Decision"],
         "incumbent_on_pilot":None,"incumbent_on_pilot_reason":"The pilot deliberately uses official training cases. The prior Attoflow release incorporates a published donor trained on this split; no prospective top-line comparison belongs here.",
         "runs":[],"common16":{"status":"feature extraction pending","cases":16,"rows":3558,"ultrabert_text_tokens_before_separators":120475,
         "incumbent_macro_f1_seven":0.6988453883,"incumbent_source":"../pooler/shared-cases/<ID>.json :: predictions.unchanged",
         "case_file":"../shared-confirmation.json","case_file_sha256":"86c94b34849ec0afa6c58fa3dac911b068465cd1dd855aa5c3ce953471797957"}}
    for stage in ["quick","screen","expanded"]:
        root=ROOT/f"write-{stage}-context-384";p=root/"report.json"
        if p.exists():
            r=read(p);s=r["results"]["confirm"];w=r["internal_edit"]
            rec["runs"].append({"stage":stage,"method":"last-MLP residual error correction","cases":s["cases"],"rows":s["direct_write"]["rows"],
            "fit_rows":r["results"]["fit"]["head_only"]["rows"],"edit_fit_rows":r["results"]["edit"]["head_only"]["rows"],
            "tensor":w["parameter"],"rank":w["rank_at_1e-6_relative"],"changed_fp32_entries":w["changed_entries"],"relative_frobenius":w["relative_frobenius"],
            "metrics":{k:stat(s[v]) for k,v in {"primary":"direct_write","paired_head_only":"head_only","same_norm_shuffle":"shuffled_control","native_head_only":"native_head_only"}.items()},
            "cases_for_internal_selection":r["results"]["select"]["cases"],"physical_forward":r.get("physical_native_pytorch_replay"),
            "receipt":str(p.relative_to(ROOT)),"checkpoint":str((root/"edit.safetensors").relative_to(ROOT))})
        root=ROOT/f"joint-{stage}-context-384";p=root/"report.json"
        if p.exists():
            r=read(p)
            for name,ex in r["experiments"].items():
                s=ex["results"]["confirm"];t=ex["tensor"]
                rec["runs"].append({"stage":stage,"method":"joint late-MLP write, "+name,"cases":r["subsets"]["confirm"]["cases"],
                  "rows":s["direct"]["rows"],"fit_rows":r["subsets"]["fit"]["rows"]+r["subsets"]["edit"]["rows"],
                  "tensor":t["name"],"rank":t["rank_above_relative_1e_6"],"changed_fp32_entries":t["changed_entries"],"relative_frobenius":t["relative_change"],
                  "metrics":{k:stat(s[v]) for k,v in {"primary":"direct","paired_head_only":"matched_zero","independently_reoptimized_head_only":"optimized_zero","same_norm_shuffle":"shuffled"}.items()},
                  "cases_for_internal_selection":r["subsets"]["select"]["cases"],"physical_forward":ex.get("fresh_forward"),
                  "receipt":str(p.relative_to(ROOT)),"checkpoint":str((root/f"{name}.safetensors").relative_to(ROOT))})
        root=ROOT/f"mlm-{stage}-context-384";p=root/"report.json"
        if p.exists():
            r=read(p);s=r["scores"]["confirm"];t=r["internal_projection_write"]
            rec["runs"].append({"stage":stage,"method":"native pretrained MLM head nonlinear projection with seven fixed vocabulary output coordinates",
                "rows":s["preceding_projection_write"]["rows"],"tensor":t["parameter"],"rank":t["rank_relative_1e-6"],
                "changed_fp32_entries":t["changed_entries"],"relative_frobenius":t["relative_frobenius"],
                "metrics":{k:stat(s[v]) for k,v in {"primary":"preceding_projection_write","paired_head_only":"calibrated_verbalizer","same_norm_shuffle":"shuffled_projection_write",
                    "independently_reoptimized_seven_cls_vocab_rows":"decoder_row_replacement"}.items()},"physical_forward":r["physical_runtime_check"],
                "receipt":str(p.relative_to(ROOT)),"checkpoint":str((root/"writes.safetensors").relative_to(ROOT))})
    v=ROOT/"vocab-screen"/"report.json"
    if v.exists():
        x=read(v)
        for name,m in x["experiments"].items():
            scores=m["results"]["confirm"]
            rec["runs"].append({"stage":"screen","method":"original input table legal token-count assignment, "+name+"; direct embedding pooling skips the contextual encoder",
                "rows":scores["primary"]["rows"],"tensor":m["tensor"],"metrics":{k:stat(scores[k]) for k in ["primary","matched_zero","optimized_zero","shuffle"]},
                "fit_counts":"vocab-counts.json","receipt":"vocab-screen/report.json","checkpoint":f"vocab-screen/{name}-patch.safetensors"})
    for tag in ["common4","common16","vocab-common16"]:
        p=ROOT/f"{tag}-report.json"
        if p.exists():
            x=read(p)
            rec[tag]={"status":"complete","cases":x["cases"],"sentences":x["rows"],"incumbent":stat(x["incumbent"]),
                      "primary_name":x.get("prospective_primary_name",x.get("primary_name")),"methods":{
                       k:{kk:(stat(vv) if isinstance(vv,dict) and "macro_f1_7" in vv else vv)
                          for kk,vv in v.items() if kk in ["primary","matched_zero","paired_zero","equal_norm_shuffle","independently_reoptimized_head_only","reoptimized_zero","paired_bootstrap_vs_incumbent","paired_bootstrap_vs_optimized_head_only"]} for k,v in x["methods"].items()},
                      "receipt":str(p.relative_to(ROOT)),"row_aligned_predictions":tag+"-predictions.npz"}
    hp=ROOT/"hybrid-vocab/report.json"
    if hp.exists():
        t=read(hp)
        rec["lexical_gate"]={"receipt":"hybrid-vocab/report.json","freeze":"hybrid-vocab/freeze.json","sets":{}}
        for name,row in t["splits"].items():
            rec["lexical_gate"]["sets"][name]={"rows":row["rows"],"cases":row["cases"],"metrics":{m:stat(row[m]) for m in ["incumbent","edited","matched_zero","reoptimized_zero","shuffle"]},
               "paired_delta_vs_incumbent":row["edited"]["paired_delta_vs_incumbent"],"changes_vs_incumbent":row["edited"]["relative_to_incumbent"]}
    for size in [2,8]:
        cp=ROOT/f"context-ablation-{size}"/"report.json"
        if cp.exists():
            ctx=read(cp)
            rec[f"context_ablation_{size}"]={"receipt":str(cp.relative_to(ROOT)),"ids":ctx["ids"],
              "regime":{name:{"rows":q["target_sentences"],"forward_blocks":q["number_of_forward_blocks"],"longest_actual_forward":q["longest_actual_forward"],
              "text_macro_f1":q["models"]["text"]["edited"]["macro_f1_7"],"document_macro_f1":q["models"]["position"]["edited"]["macro_f1_7"]} for name,q in ctx["conditions"].items()}}
    sp=ROOT/"vocab-common16-spans.json"
    if sp.exists():
        obj=read(sp);rec["lexical_common16_exact_spans"]={"receipt":sp.name,"methods":obj["methods"],"definition":obj["definition"]}
    for sub in ["vocab-screen","sentence-vocab"]:
        ep=ROOT/sub/"extended-report.json"
        if ep.exists():
            q=read(ep)
            rec[sub+"_validation_extension"]={"receipt":str(ep.relative_to(ROOT)),"primary":q["primary"],"sets":{}}
            for tag,bank in q["sets"].items():
                primary=bank["models"][q["primary"]]
                rec[sub+"_validation_extension"]["sets"][tag]={"cases":bank["cases"],"rows":bank["rows"],"incumbent":stat(bank["incumbent"]),"primary":stat(primary),
                  "paired_delta_vs_incumbent":primary["paired_interval_vs_incumbent"]}
    pq=ROOT/"vocab-screen/portable-replay.json"
    if pq.exists():rec["portable_lexical_runtime"]=dict(**read(pq),receipt=str(pq.relative_to(ROOT)))
    sr=ROOT/"sentence-vocab/report.json"
    if sr.exists():
        q=read(sr);fr=read(sr.parent/"freeze.json");mk=fr["primary_name"]
        rec["all_sentence_supervised_vocabulary_control"]={"receipt":str(sr.relative_to(ROOT)),"freeze":"sentence-vocab/freeze.json","fit_rows":fr["fit_rows"],
          "primary":mk,"tensor":q["tensors"][mk],"metrics":{}}
        for tag,ss in q["results"].items():
            m=ss[mk]
            rec["all_sentence_supervised_vocabulary_control"]["metrics"][tag]=dict(**{k:stat(m[k]) for k in ["edited","matched_zero","optimized_zero","equal_norm_shuffle"]},paired_delta_vs_optimized_zero=m["paired_vs_optimized_zero"])
    cv=ROOT/"conditional-vocab/report.json"
    if cv.exists():
        q=read(cv);rec["previous_role_weight_edits"]={"receipt":str(cv.relative_to(ROOT)),"primary_selection_prefers_zero":q["primary_selection_prefers_no_edit"],
        "tensor":"8 materializations of ultrabert.embed_tokens.weight, one per previous role or START","rank_each":[v["rank"] for v in q["source_float32_tensor_receipts"]],
        "metrics":{n:dict(**{k:stat(v[k]) for k in ["edited","matched_zero","reoptimized_zero","equal_norm_shuffle"]},paired_delta_vs_reoptimized_zero=v["paired_delta_vs_reoptimized_zero"]) for n,v in q["results"].items()}}
    vx=ROOT/"conditional-vocab/extended-report.json"
    if vx.exists():
        x=read(vx);rec["previous_role_extended"]={"receipt":str(vx.relative_to(ROOT)),"sets":{n:{"cases":v["cases"],"rows":v["rows"],"edited":stat(v["edited"]),"optimized_zero":stat(v["optimized_zero"]),"delta_vs_optimized_zero":v["delta_vs_optimized_zero"]} for n,v in x["sets"].items()}}
    native=ROOT/"native-before-after-flips.json"
    if native.exists():rec["real_sentence_tensor_flip_examples"]={"receipt":native.name,"methods":{k:{"case_id":v["id"],"flipped_rows":v["flipped_rows"]} for k,v in read(native)["methods"].items()}}
    scale=ROOT/"context-scale";fresh=scale/"manifest.json"
    if fresh.exists():
        x=read(fresh);rec["larger_contextual_experiment"]={"manifest":"context-scale/manifest.json","counts":x["counts"],"exclude_previously_used_cases":x["excluded_earlier_case_ids_count"],
           "separate_evaluation_cohort":"context-scale/validation/manifest.json","current_native_engine":"resumable FP32 CPU/CUDA","backend_qa":{}}
        item=rec["larger_contextual_experiment"]
        for key,pth in [("colab_package","package-receipt.json"),("restart","worker-smoke/result.json"),("cpu_physical","fit-worker-qa/run/physical-replay.json"),("ov_cpu","openvino/parity-CPU.json"),("ov_gpu","openvino/parity-GPU.json")]:
            f=scale/pth
            if f.exists():
                v=read(f)
                if key.startswith('ov'):v={"passed":v["passed"],"mixed_length_tokens_per_second":v["mixed_length_tokens_per_second"],"full384_tokens_per_second":v.get("full384_tokens_per_second")}
                item["backend_qa"][key]={"receipt":str(f.relative_to(ROOT)),"details":v}
        active=scale/"active-backend.json"
        if active.exists():
            item["active_backend"]=read(active)
            rr=scale/item["active_backend"]["features"]
            totals={}
            for group,rows in x["partitions"].items():
                done=[r for r in rows if (rr/group/(r["id"]+".npz")).exists()]
                totals[group]={"cases":len(done),"rows":sum(r["rows"] for r in done),"target_tokens":sum(r["ultrabert_tokens"] for r in done)}
            item["fully_cached"]=totals
        ca=scale/"openvino/case-parity.json"
        if ca.exists():
            v=read(ca);item["backend_qa"]["full_case_torch_cpu_vs_igpu"]={"cases":v["cases"],"rows":v["rows"],"tokens":v["tokens"],"all_class_argmax_agree":v["all_class_argmax_agree"],"receipt":str(ca.relative_to(ROOT))}
        for n in ["interim23","full24"]:
            for where in ["confirm","shared"]:
                f=scale/f"run-{n}"/(where+"-report.json")
                if f.exists():
                    v=read(f);item[n+"_"+where]={"receipt":str(f.relative_to(ROOT)),"metrics":{k:stat(co) for k,co in v["methods"].items()},"paired_delta_vs_matched_head":v["paired_delta_vs_matched_head"]}
    if fresh.exists():
        for n in ["interim23","full24"]:
            alt=scale/f"run-{n}"/"joint-factorization"/"shared-report.json"
            if alt.exists():
                v=read(alt);item["exact_joint_"+n]={"receipt":str(alt.relative_to(ROOT)),"metrics":{k:stat(co) for k,co in v["methods"].items()},"delta_vs_optimized_zero":v["delta_vs_reoptimized_zero"]}
            seq=scale/f"run-{n}"/"sequence-duration"/"shared-report.json"
            if seq.exists():
                v=read(seq);item["structured_"+n]={"receipt":str(seq.relative_to(ROOT)),"metrics":{k:stat(co) for k,co in v["methods"].items()},"delta_vs_optimized_head":v["versus_reoptimized_zero"]}
            same=scale/f"run-{n}"/"same-igpu-validation"/"shared-report.json"
            if same.exists():
                v=read(same);item["same_gpu_"+n]={"receipt":str(same.relative_to(ROOT)),"metrics":{k:stat(co) for k,co in v["methods"].items()}}
    out=ROOT/"lane-summary.json"
    tmp=out.with_suffix(".partial")
    tmp.write_text(json.dumps(rec,indent=2));tmp.replace(out)
if __name__=="__main__":sumrun()
