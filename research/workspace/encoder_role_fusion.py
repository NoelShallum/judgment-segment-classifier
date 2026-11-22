"""Compose independently written encoder output rows by role, using frozen features."""
import hashlib
import itertools
import json

import numpy as np
from safetensors.numpy import load_file,save_file
import torch

import encoder_span_edit as shared
from discourse_span_edit import text_features
from hier_model import CRF,HierModel,path_labels
from hier_trial import DONOR,OUT
from program_trial import save_json
from seven_paths import ROOT
from span_eval import counts,summarize
from transition_span_edit import features
from weight_edit import file_hash


def center(x):
    return x-x.mean(1,keepdims=True)


def prepare(task,bank,keys):
    source=[]
    profile_cache=ROOT/"cache/encoder-span-edit-v2"
    for model,use_contextual in keys:
        tag=model+"-contextual-keys" if use_contextual else model
        default=ROOT/"results"/f"encoder-span-{model}.json"
        accelerated=ROOT/"results"/f"encoder-span-{model}-accelerated.json"
        data=json.loads((default if default.exists() else accelerated).read_text())
        cpu=data.get("backend","torch")=="torch"
        if use_contextual:
            data=json.loads((ROOT/"results"/f"encoder-contextual-keys-{model}.json").read_text())
        section=data["tasks"][task]["standalone_encoder"]
        cached=profile_cache/(model if cpu else model+"-gpu")/section["mode"]
        artifact=ROOT/"checkpoints"/("encoder-span-"+tag)/(task+"-pure.safetensors")
        profile=load_file(str(artifact))
        outputs={}
        for k,v in bank.items():
            sig=hashlib.sha256(repr(v).encode()).hexdigest()
            features=np.load(cached/(sig+".npy"),allow_pickle=False)
            outputs[k]=center(features@profile["classifier.weight"].T+profile["classifier.bias"])
        source.append({"name":tag,"weights":profile["classifier.weight"],"bias":profile["classifier.bias"],
                       "mode":section["mode"],"scores":outputs,"sha256":file_hash(artifact)})
    return source


def main():
    torch.set_num_threads(2)
    specs=[("minilm",False),("minilm",True),("granite",False),("gte",False)]
    gs,bank=shared.make_documents()
    hier=HierModel(DONOR)
    results={}
    for task,docs in gs.items():
        relevant={k:v for k,v in bank.items() if k.startswith(task+"/")}
        source=prepare(task,relevant,specs)
        def take(model):
            out={"name":model["name"],"mode":model["mode"],"sha256":model["sha256"]}
            return out
        profiles=("exact","balanced")
        heads={k:load_file(str(OUT/(task+"-discourse-"+k+".safetensors"))) for k in profiles}
        raw={group:features(hier,list(doc.values())) for group,doc in docs.items()}
        inputs={}
        for group,doc in docs.items():
            inputs[group]=[]
            for i,(case,d) in enumerate(doc.items()):
                p,z=text_features(d)
                body={name:raw[group][i]+p@h["cue_head"]+z@h["position_head"] for name,h in heads.items()}
                candidate=[v["scores"][f"{task}/{group}/{case}"] for v in source]
                inputs[group].append((d,body,np.stack(candidate)))
        groups=[
            ("all",list(range(7))),("document_structure",[0,2,6]),("speaker",[3,4]),
            ("factual_analysis",[1,5]),("reasoning",[5]),("issue_decision",[2,6])
        ]
        sets=[
            ("minilm",[0]),("minilm_context",[1]),("granite",[2]),("gte",[3]),
            ("three_encoders",[1,2,3]),("granite_gte",[2,3])
        ]
        def evaluate(f,base,amplitudes):
            metrics=[]
            for d,body,v in inputs[f]:
                adjustment=(v*amplitudes[:,None,:]).sum(0)
                labels=path_labels(body[base]+adjustment,heads[base][CRF])
                metrics.append(counts(d,d["units"],labels))
            return np.stack(metrics)
        metrics=[];best=None;balance=None
        clear=np.zeros((len(source),7),dtype=np.float32)
        a=summarize(evaluate("select","balanced",clear).sum(0))
        floor=a["token_accuracy"]-.01
        for family,streams in sets:
            for (target,roles),level,base in itertools.product(groups,(0.,.15,.35,.65),profiles):
                weights=clear.copy()
                weights[np.ix_(streams,roles)]=level/len(streams)
                q=summarize(evaluate("select",base,weights).sum(0))
                z=(q["exact_labeled_spans"]["f1"],q["token_accuracy"])
                metrics.append([family,target,level,base,*z])
                record=z,family,target,level,base,weights
                if best is None or z>best[0]:best=record
                if z[1]>=floor and (balance is None or z>balance[0]):balance=record
        grades={}
        for tag,run in (("span",best),("balanced",balance)):
            _,family,target,level,base,w=run
            output=OUT/f"{task}-encoder-fusion-{tag}.safetensors"
            # Each pre-existing head's common role component is removed before applying role-specific weights.
            # Update tensor layout is a block-structured linear mapping from concatenated encoder features.
            matrices=[m["weights"]-m["weights"].mean(axis=0,keepdims=True) for m in source]
            offsets=[m["bias"]-m["bias"].mean() for m in source]
            con=np.concatenate([weights[:,None]*a for weights,a in zip(w,matrices)],axis=1)
            delta=sum(weights*b for weights,b in zip(w,offsets))
            save_file({**heads[base],"auxiliary.weight":con.astype("f4"),"auxiliary.bias":delta.astype("f4")},str(output))
            scores={}
            for group in docs:
                obs=evaluate(group,base,w);prev=evaluate(group,base,clear)
                rng=np.random.default_rng(8821);gaps=[]
                for ii in rng.integers(len(obs),size=(1400,len(obs))):
                    a=summarize(obs[ii].sum(0));b=summarize(prev[ii].sum(0))
                    gaps.append([a["exact_labeled_spans"]["f1"]-b["exact_labeled_spans"]["f1"],a["token_accuracy"]-b["token_accuracy"]])
                scores[group]={"cases":len(obs),"candidate":summarize(obs.sum(0)),"incumbent":summarize(prev.sum(0)),
                               "span_f1_token_accuracy_gap_ci95":np.quantile(gaps,[.025,.975],axis=0).T.tolist()}
            grades[tag]={"spec":[family,target,level,base],"weights_sha256":file_hash(output),"scores":scores}
            print(task,tag,grades[tag]["spec"],scores["test"]["candidate"]["exact_labeled_spans"]["f1"],flush=True)
        results[task]={"source":[take(s) for s in source],"profiles":grades,"grid":metrics}
        save_json(ROOT/"results"/"role-specialized-encoder-fusion.json",{"method":"Physical block write combining separately authored MiniLM, Granite, and GTE role rows with the same incumbent legal hierarchy. The coefficient table allows specific roles to change while keeping others unaffected. The zero coefficient is eligible and the development-frozen choices are measured on case-separate confirmation and test. No gradients.", "tasks":results})
if __name__=="__main__":
    main()
