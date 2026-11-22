"""Paired case-level comparison with the strongest archived exact-span projection."""
import csv
import json
import numpy as np
from safetensors.numpy import load_file
from hier_full import CACHE,gold_cases
from hier_model import CRF,path_labels
from hier_trial import OUT
from discourse_span_edit import text_features
from program_trial import save_json
from seven_paths import ROOT
from span_eval import counts,document,summarize
from weight_edit import file_hash

def main():
    root=ROOT/"results"
    span_file=root/"archived-sentence-span-references.json"
    releases=json.loads(span_file.read_text())["scored_release_files"]
    top=max(releases,key=lambda k:releases[k]["scores"]["exact_labeled_spans"]["f1"])
    if top!="previous_predicted_role":raise ValueError("The top archive changed. Update the source mapping.")
    p=ROOT/"references/all-predictions/LegalSeg_InLegalBERT_Predictions/InLegalBert(i-1, label_p, i).csv"
    assert file_hash(p)==releases[top]["bytes_sha256"]
    with p.open(newline="",encoding="utf-8-sig") as f:
        data=np.array([(int(r["true"]),int(r["pred"])) for r in csv.DictReader(f)])
    profiles={"plain":OUT/"document.safetensors","exact":OUT/"role-specific-whole.safetensors","balanced":OUT/"role-specific-balanced.safetensors"}
    weights={k:load_file(str(v))[CRF] for k,v in profiles.items()}
    add={f"discourse_{name}":OUT/(f"whole-discourse-{name}.safetensors") for name in ("exact","balanced")}
    extra={k:load_file(str(v)) for k,v in add.items() if v.is_file()}
    a=[];b={k:[] for k in (*weights,*extra)}
    for case,rows in gold_cases().items():
        d=document(case,rows)
        use=[i for i,t,_ in rows if t.strip()]
        assert np.array_equal(data[use,0],d["reference_roles"])
        a.append(counts(d,d["reference_units"],data[use,1]))
        with np.load(CACHE/(case+".npz")) as f:
            xx=f["raw"]
        for k,trans in weights.items():
            b[k].append(counts(d,d["units"],path_labels(xx,trans)))
        if extra:
            cue,pos=text_features(d)
            for k,got in extra.items():
                output=xx+cue@got["cue_head"]+pos@got["position_head"]
                b[k].append(counts(d,d["units"],path_labels(output,got[CRF])))
    ref=np.stack(a);rng=np.random.default_rng(659113)
    counts_by_doc=rng.multinomial(len(ref),np.ones(len(ref))/len(ref),size=4000)
    def metric(vals):
        # Statistics are sufficient in a paired document resample.
        span=vals[:,49:52]
        exact=2*span[:,0]/np.maximum(1,span[:,1]+span[:,2])
        matrix=vals[:,:49].reshape(-1,7,7)
        acc=np.trace(matrix,axis1=1,axis2=2)/matrix.sum(axis=(1,2))
        return np.column_stack((exact,acc))
    draws=metric(counts_by_doc@ref)
    evaluations={}
    for k,v in b.items():
        now=np.stack(v)
        gap=metric(counts_by_doc@now)-draws
        evaluations[k]={"scores":summarize(now.sum(0)),"paired_gap_95pct":np.quantile(gap,[.025,.975],axis=0).T.tolist()}
    result={"cases":len(a),"baseline_release":top,"published_projection":summarize(ref.sum(0)),"raw_text_candidates":evaluations,
            "statistic_order":["exact_labeled_span_f1","token_accuracy"],"disclosure":"The public baseline receives original LegalSeg sentence cuts. The other profiles receive reconstructed raw text without those cuts. Scores are file-based comparisons on the official test and were analyzed after local weights were frozen."}
    save_json(root/"archive-span-paired-comparison.json",result)
    print([(k,t["paired_gap_95pct"]) for k,t in evaluations.items()],flush=True)
if __name__=="__main__":
    main()
