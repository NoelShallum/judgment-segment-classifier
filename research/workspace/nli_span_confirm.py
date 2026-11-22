"""Replay the frozen entailment edit on the remaining clipped-passage cases."""
import json
import numpy as np
import openvino as ov
from safetensors.numpy import load_file
import torch

from discourse_span_edit import text_features
from fragment_trial import passages
from hier_model import CRF,HierModel,path_labels
from hier_trial import DONOR,OUT
from nli_span_edit import Entailment,vectors
from program_trial import save_json
from seven_paths import ROOT
from span_eval import counts,summarize
from transition_span_edit import features
from weight_edit import file_hash

def main():
    torch.set_num_threads(1)
    state=json.loads((ROOT/"results/nli-minilm-span-edit.json").read_text())["model_profiles"]["selected"]
    selected=OUT/"nli-fragment-selected.safetensors"
    assert file_hash(selected)==state["sha256"]
    program=load_file(str(selected))
    mode=bool(state["settings"][0])
    runner=Entailment()
    backend=ROOT/"runtime/nli-span-minilm"
    core=ov.Core()
    runner.runtime=core.compile_model(core.read_model(backend/"model.xml"),"GPU",{"INFERENCE_PRECISION_HINT":"f32","CACHE_DIR":str(backend/"compiled")})
    ds=passages()
    all_docs=ds["test"]+ds["cloud"]
    lex=vectors(runner,{d["id"]:d for d in all_docs},mode)
    parent=HierModel(DONOR);emissions=features(parent,all_docs)
    t=load_file(str(OUT/"fragment.safetensors"))["transition"]
    discourse=load_file(str(OUT/"fragment-discourse-exact.safetensors"))
    result={"selected":[],"base":[],"discourse":[]}
    for d,raw in zip(all_docs,emissions):
        x=lex[d["id"]]
        raw_nli=np.einsum("lrc,rc->lr",x,program["role_nli_weights"])+program["role_nli_bias"]
        clue,pos=text_features(d)
        inventories={
            "selected":(raw*program["hierarchy_mix"]+raw_nli,program[CRF]),
            "base":(raw,t),
            "discourse":(raw+clue@discourse["cue_head"]+pos@discourse["position_head"],discourse[CRF]),
        }
        for key,(matrix,trans) in inventories.items():
            result[key].append(counts(d,d["units"],path_labels(matrix,trans)))
    report={}
    for label,section in (("previous_54",slice(0,54)),("new_66",slice(54,120)),("all_120",slice(0,120)),("matched_12",slice(120,None))):
        arrays={k:np.stack(v[section]) for k,v in result.items()}
        q={k:summarize(x.sum(0)) for k,x in arrays.items()}
        rng=np.random.default_rng(845498)
        indices=rng.integers(len(arrays["base"]),size=(1800,len(arrays["base"])))
        diffs=[]
        for index in indices:
            z={k:summarize(v[index].sum(0))["exact_labeled_spans"]["f1"] for k,v in arrays.items()}
            diffs.append([z["selected"]-z["base"],z["selected"]-z["discourse"]])
        report[label]={"cases":len(arrays["base"]),"metrics":q,"paired_exact_span_gaps_95pct":np.quantile(diffs,[.025,.975],axis=0).T.tolist()}
        print(label,[(k,x["exact_labeled_spans"]["f1"]) for k,x in q.items()],flush=True)
    save_json(ROOT/"results/nli-minilm-full-confirmation.json",{"frozen_weights_sha256":state["sha256"],"gap_order":["vs original fragment","vs discourse clip"],
             "protocol":"The weights are those frozen on the earlier 30-case validation sample. The first 54 test cases are the originally reported engineering screen. The additional 66 cases are scored for the first time for this head. The 12 archived matched passages stay separate.",
             "cohorts":report})
if __name__=="__main__":
    main()
