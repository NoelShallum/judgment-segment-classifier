"""Evaluate new physical patches with the previously recorded Gemini decisions."""
import hashlib
import json

import numpy as np
import torch

from discourse_span_edit import text_features
from duration_edit import decode as duration_decode
from hier_model import CRF, HierModel,path_labels
from hier_trial import DONOR,OUT
from program_trial import save_json
from safetensors.numpy import load_file
from seven_paths import LABELS, ROOT
from span_eval import counts,document,summarize

def main():
    torch.set_num_threads(2)
    root=ROOT/"seven-spans"
    source=json.loads((root/"cloud.json").read_text())
    gemini=json.loads((root/"gemini-cloud.json").read_text())
    variants={
      "contrastive_fragment":HierModel(DONOR,OUT/"contrastive-fragment.safetensors"),
      "role_specific_whole":HierModel(DONOR,OUT/"role-specific-whole.safetensors"),
      "role_specific_fragment":HierModel(DONOR,OUT/"role-specific-fragment.safetensors"),
    }
    dur=load_file(str(OUT/"duration-fragment.safetensors"))
    discourse={f"discourse_{kind}_{profile}":load_file(str(OUT/f"{kind}-discourse-{profile}.safetensors"))
               for kind in ("whole","fragment") for profile in ("balanced","exact")}
    parent=HierModel(DONOR)
    lines={k:[] for k in (*variants,*discourse,"duration_fragment","gemini")}
    labels={k:[] for k in lines}
    for block in source:
        case=block["doc"]
        d=document(case,[(r["row"],r["text"],LABELS.index(r["gold"])) for r in block["rows"]])
        assert hashlib.sha256(d["text"].encode()).hexdigest()==gemini[case]["raw_sha256"]
        offs=d["units"]
        assert [list(o) for o in offs]==gemini[case]["unit_offsets"]
        snippets=[d["text"][a:b] for a,b in offs]
        em=next(iter(variants.values()))
        features=em.contextualize(em.words(snippets))
        for k,model in variants.items():
            pred=path_labels(model.logits(features),model.transitions)
            lines[k].append(counts(d,offs,pred))
            labels[k].extend(pred)
        cues,positions=text_features(d)
        for k,weights in discourse.items():
            emissions=parent.logits(features)+cues@weights["cue_head"]+positions@weights["position_head"]
            pred=path_labels(emissions,weights[CRF])
            lines[k].append(counts(d,offs,pred))
            labels[k].extend(pred)
        p=duration_decode(variants["role_specific_fragment"].logits(features),dur[CRF],dur["length_bias"])
        lines["duration_fragment"].append(counts(d,offs,p));labels["duration_fragment"].extend(p)
        g=np.array([LABELS.index(k) for k in gemini[case]["unit_labels"]])
        lines["gemini"].append(counts(d,offs,g));labels["gemini"].extend(g)
    baseline=np.stack(lines["gemini"])
    result={"cases":len(source),"source_rows":sum(len(v["rows"]) for v in source),"frozen_gemini_span_reference":summarize(baseline.sum(0)),"candidates":{}}
    for k,z in lines.items():
        if k=="gemini":continue
        matrix=np.stack(z);rng=np.random.default_rng(7123);deltas=[]
        for ix in rng.integers(len(matrix),size=(3000,len(matrix))):
            a=summarize(matrix[ix].sum(0));b=summarize(baseline[ix].sum(0))
            deltas.append([a["exact_labeled_spans"]["f1"]-b["exact_labeled_spans"]["f1"],a["token_accuracy"]-b["token_accuracy"]])
        result["candidates"][k]={"scores":summarize(matrix.sum(0)),"span_f1_token_accuracy_gap_ci95":np.quantile(deltas,[.025,.975],axis=0).T.tolist(),
                 "agreement":int(sum(a==b for a,b in zip(labels[k],labels["gemini"])))}
    save_json(ROOT/"results"/"focused-span-gemini-replay.json",result)
    print({k:(v["scores"]["exact_labeled_spans"]["f1"],v["scores"]["token_accuracy"]) for k,v in result["candidates"].items()},flush=True)

if __name__=="__main__":
    main()
