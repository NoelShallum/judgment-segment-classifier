"""Write a targeted edit into ModernBERT's pretrained final LayerNorm."""
import argparse
import json
from pathlib import Path
import os
import numpy as np
from safetensors.torch import load_file as load_tensors,save_file as save_tensors
from safetensors.numpy import load_file,save_file
import torch

import encoder_span_edit as study
from hier_model import HierModel,CRF,path_labels
from hier_trial import DONOR,OUT
from program_trial import fixture_rows,save_json
from role_fixtures import ANCHORS,CHECKS
from seven_paths import ROOT
from span_eval import counts,summarize
from transition_span_edit import features
from weight_edit import file_hash


def role_scales(keys,labels,neutral):
    means=np.stack([keys[labels==j].mean(0) for j in range(7)])
    between=means.var(0)
    foreground=np.stack([keys[labels==j].var(0) for j in range(7)]).mean(0)
    nuisance=neutral.var(0)
    return np.log1p(between/np.maximum(foreground+nuisance,1e-7))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("encoder",choices=("granite","gte"));args=parser.parse_args()
    torch.set_num_threads(2)
    name=args.encoder
    parent_result=ROOT/"results"/("encoder-span-"+name+".json")
    if not parent_result.exists():parent_result=ROOT/"results"/("encoder-span-"+name+"-accelerated.json")
    parent_report=json.loads(parent_result.read_text())
    study.CACHE=study.CACHE/(name if parent_report.get("backend","torch")=="torch" else name+"-gpu")
    pretrained=study.Encoder(name)
    norm=pretrained.model.final_norm
    if norm.bias is not None and torch.count_nonzero(norm.bias).item()!=0:
        raise ValueError("This exact cached channel transformation requires a zero LayerNorm bias")
    groups,bank=study.make_documents()
    fixtures=fixture_rows(ANCHORS)+fixture_rows(CHECKS)
    a=pretrained.cache({"keys":[r[2] for r in fixtures],"unlabeled":study.neutral_texts()},"isolated")
    y=np.asarray([r[3] for r in fixtures]);keys=a["keys"];neutral=a["unlabeled"]
    scales=role_scales(keys,y,neutral)
    vectors={mode:pretrained.cache(bank,mode) for mode in study.MODES}
    base=parent_report
    hierarchy=HierModel(DONOR)
    out=ROOT/"checkpoints"/("encoder-span-"+name)
    old_norm=norm.weight.detach().clone()
    study_result={}
    for task,sets in groups.items():
        task_vectors={}
        meta=base["tasks"][task]["best"]
        mode=meta["mode"]
        with_profile=load_file(str(out/(task+"-selected.safetensors")))
        tr=with_profile["crf.transitions"]
        share=meta["new_encoder_fraction"]
        authored_w=with_profile["classifier.weight"]
        b=with_profile["classifier.bias"]
        for kind,docs in sets.items():
            ref=features(hierarchy,list(docs.values()))
            task_vectors[kind]=[(d,vectors[mode][f"{task}/{kind}/{id}"],r) for (id,d),r in zip(docs.items(),ref)]
        def score(kind, scale):
            agg=[counts(d,d["units"],path_labels(study.normalized(v*scale)@authored_w.T+b+(1-share)*r,tr))
                 for d,v,r in task_vectors[kind]]
            return summarize(np.sum(agg,axis=0))
        grid=[];best=None
        for gain in (0.,-.5,-.25,.25,.5,1.):
            factors=np.exp(np.clip(gain*(scales-np.median(scales)),-.5,.5)).astype("f4")
            m=score("select",factors)
            key=(m["exact_labeled_spans"]["f1"],m["token_accuracy"])
            grid.append([gain,*key])
            if best is None or key>best[0]:best=(key,gain,factors)
        _,gain,factors=best
        patch=out/(task+"-final-normalization.safetensors")
        new_norm=old_norm*torch.from_numpy(factors)
        save_tensors({"final_norm.weight":new_norm},str(patch))
        # Perform real forwards through the edited donor and compare to the cache replay.
        examples=next(iter(sets["select"].values()))
        ss=[examples["text"][l:r] for l,r in examples["units"]][:9]
        parent=pretrained.process([ss],mode)[0]
        with torch.no_grad():norm.weight.copy_(load_tensors(str(patch))["final_norm.weight"])
        edited=pretrained.process([ss],mode)[0]
        error=float(np.max(abs(edited-study.normalized(parent*factors))))
        with torch.no_grad():norm.weight.copy_(old_norm)
        if error>2e-5:
            raise ValueError(f"Analytic replay diverges from physical model: {error}")
        study_result[task]={"prototype_gain":gain,"changed_pretrained_parameters":torch.count_nonzero(new_norm-old_norm).item(),
                            "patch_sha256":file_hash(patch),"grid":grid,"physical_forward_equivalence_max_error":error,
                            "scores":{k:{"parent":score(k,np.ones_like(factors)),"edited":score(k,factors)} for k in sets}}
    save_json(ROOT/"results"/("encoder-norm-edit-"+name+".json"),
              {"method":"Use authored roles and unlabeled activations to estimate channel relevance. Directly multiply the pretrained encoder final normalization tensor. Forward inference verifies equivalence of cached search. No gradients.",
               "tasks":study_result})
if __name__=="__main__":
    main()
