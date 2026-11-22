"""Validate a physical midpoint of two published fine-tuned encoders using only the current excerpt."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np

from encoder import components, features, fingerprint, write_config
from program_trial import anchored_edit, get_features, metric, official_sample, save_json, paired_document_interval, fixture_rows
from role_fixtures import ANCHORS
from weight_edit import LABELS, ORIGINAL, ROOT, file_hash, offsets, patched_checkpoint, prior_bias, tensor


CONTEXT = ROOT/"references/context-parent/InLegalBERT(i-2, i-1, i)model.safetensors"
CONTEXT_HASH = "a81f509d250f98b44872e89e10186d5dcc17d37905b69aa1d9a077fb841a62ac"
MIDPOINT = ROOT/"checkpoints/midpoint-parent/model.safetensors"


def write_midpoint():
    if file_hash(CONTEXT)!=CONTEXT_HASH:
        raise ValueError("Unexpected contextual parent")
    _, a_header=offsets(ORIGINAL);_, b_header=offsets(CONTEXT)
    a_spec={k:(v["dtype"],v["shape"]) for k,v in a_header.items() if k!="__metadata__"}
    b_spec={k:(v["dtype"],v["shape"]) for k,v in b_header.items() if k!="__metadata__"}
    if a_spec!=b_spec or any(dtype!="F32" for dtype,shape in a_spec.values()):
        raise ValueError("Parent tensors are not compatible")
    info=MIDPOINT.parent/"merge.json"
    if MIDPOINT.is_file() and info.is_file() and file_hash(MIDPOINT)==json.loads(info.read_text())["sha256"]:
        return
    changes={}
    for name in a_spec:
        a,b=tensor(ORIGINAL,name),tensor(CONTEXT,name)
        changes[name]=a+0.5*(b-a)
    details=patched_checkpoint(ORIGINAL,MIDPOINT,changes)
    save_json(info,{"sha256":file_hash(MIDPOINT),"formula":"A + 0.5 * (B - A), float32 per tensor",
                    "source_a":file_hash(ORIGINAL),"source_b":file_hash(CONTEXT),
                    "edited_scalars":sum(d["changed"] for d in details.values()),"tensor_count":len(a_spec)})
    write_config(MIDPOINT.parent)


def get_vectors(path,groups,backend):
    if path==ORIGINAL:
        model,tok,device=components()
        res={}
        for name,rows in groups.items():
            res[name]=get_features(model,tok,device,rows,name)[0]
        return res
    digest=file_hash(path)
    target=ROOT/"cache"
    result={};engine=None
    for key,rows in groups.items():
        texts=[r[2] for r in rows]
        cache=target/f"mixed-{digest[:15]}-{fingerprint(texts)[:20]}-{backend}.npy"
        if cache.is_file():
            result[key]=np.load(cache,allow_pickle=False)
            continue
        if backend=="intel":
            from openvino_worker import get_engine,embed
            if engine is None:
                engine,tok=get_engine("GPU",path,ROOT/"runtime"/("encoder-"+digest[:15]))
            x=embed(engine,tok,texts)
        else:
            if engine is None:
                engine,tok,device=components(checkpoint=path)
            x=features(engine,tok,device,texts)
        with cache.with_suffix(".partial").open("wb") as stream:np.save(stream,x,allow_pickle=False)
        cache.with_suffix(".partial").replace(cache)
        result[key]=x
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--backend",choices=["torch","intel"],default="torch");args=ap.parse_args()
    write_midpoint()
    select,weights,st=official_sample("select",160)
    confirm,cweights,cstats=official_sample("confirm",160)
    groups={"authored-anchors":fixture_rows(ANCHORS),"select-160":select}
    y=np.array([r[3] for r in select]);ay=np.array([r[3] for r in groups["authored-anchors"]])
    best=None;records=[]
    for name,frac,path in (("original",0,ORIGINAL),("midpoint",0.5,MIDPOINT),("context_parent",1,CONTEXT)):
        print("Scoring",name,"on isolated validation excerpts",flush=True)
        vectors=get_vectors(path,groups,args.backend)
        dw,db=anchored_edit(vectors["authored-anchors"],ay)
        w0,b0=tensor(path,"classifier.weight"),tensor(path,"classifier.bias")
        for prior,margin in ((0,0),(.65,4)):
            w=w0+margin*dw;b=b0+prior_bias(prior)+margin*db
            preds=(vectors["select-160"]@w.T+b).argmax(1);scores=metric(y,preds,weights)
            setting={"encoder":name,"context_parent_fraction":frac,"prior":prior,"margin":margin}
            records.append({"setting":setting,"selection":scores})
            if best is None or scores["macro_f1"]>best[0]+1e-12:
                best=(scores["macro_f1"],path,setting,w,b)
        del vectors;gc.collect()
    value,path,setting,w,b=best
    target=ROOT/"checkpoints/selected-mixed"
    changes=patched_checkpoint(path,target/"model.safetensors",{"classifier.weight":w,"classifier.bias":b})
    write_config(target)
    freeze={"method":"Weight interpolation of a standalone and a context-trained BERT followed by optional authored head edit",
            "selected":setting,"original_sha256":file_hash(ORIGINAL),"context_parent_sha256":CONTEXT_HASH,
            "backbone_source_sha256":file_hash(path),"target_sha256":file_hash(target/"model.safetensors"),
            "final_head_changes":changes,"labels":LABELS}
    save_json(target/"edit.json",freeze)
    cx=get_vectors(path,{"confirm-160":confirm},args.backend)["confirm-160"]
    regular=get_vectors(ORIGINAL,{"confirm-160":confirm},args.backend)["confirm-160"]
    direct=ROOT/"checkpoints/selected-direct/model.safetensors"
    priorpred=(regular@tensor(direct,"classifier.weight").T+tensor(direct,"classifier.bias")).argmax(1)
    pred=(cx@w.T+b).argmax(1);cy=np.array([r[3] for r in confirm])
    report={"protocol":"Six prespecified candidates, two compatible public parents trained on related tasks. Current sentence only for every candidate. No previous text, predicted label, gold label or document position at inference. Select on the fixed validation development sample, then confirm without altering weights.",
            "selection_stats":st,"confirmation_stats":cstats,"candidates":records,"selected":freeze,
            "selected_development_macro_f1":value,"confirmation":metric(cy,pred,cweights),
            "confirmation_direct":metric(cy,priorpred,cweights),"paired_interval_vs_direct":paired_document_interval(confirm,cy,priorpred,pred,cweights),
            "official_test_used":False}
    save_json(ROOT/"results/mixed-encoder-validation.json",report)
    print("Selected",setting,"selection",value,"confirmation",report["confirmation"]["macro_f1"],flush=True)


if __name__=="__main__":
    main()
