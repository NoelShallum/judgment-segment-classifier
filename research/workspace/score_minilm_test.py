"""Full official single-excerpt test for the validation-selected compact encoder."""
import argparse
from collections import Counter
import json
import math
import os
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from encoder import fingerprint
from program_trial import metric, save_json
from score_official_test import official, compare
from train_minilm import MAX_LENGTH, logits, tokenize
from weight_edit import ROOT, file_hash


def export_intel(model, target, expected):
    import openvino as ov
    folder=ROOT/"runtime"/("minilm-"+expected[:16]);folder.mkdir(exist_ok=True,parents=True)
    xml=folder/"model.xml"
    if not xml.is_file() or not xml.with_suffix(".bin").is_file():
        class Scores(torch.nn.Module):
            def __init__(self, underlying):
                super().__init__();self.model=underlying
            def forward(self, input_ids, attention_mask):
                return self.model(input_ids=input_ids,attention_mask=attention_mask,return_dict=False)[0]
        example=(torch.ones(2,16,dtype=torch.long),torch.ones(2,16,dtype=torch.long))
        with torch.inference_mode():
            representation=ov.convert_model(Scores(model).eval(),example_input=example)
        for port,name in zip(representation.inputs,("input_ids","attention_mask")):
            port.get_tensor().set_names({name})
        representation.reshape({n:ov.PartialShape([-1,-1]) for n in ("input_ids","attention_mask")})
        ov.save_model(representation,xml,compress_to_fp16=False)
    return ov.Core().compile_model(xml,"GPU",{"INFERENCE_PRECISION_HINT":"f32","PERFORMANCE_HINT":"LATENCY",
                                  "NUM_STREAMS":"1","CACHE_DIR":str(folder/"compiled")})


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--backend",choices=["torch","intel"],default="torch")
    ap.add_argument("--per-role",type=int,default=2048);ap.add_argument("--epochs",type=int,default=2)
    args=ap.parse_args()
    tag=f"minilm-{args.per_role}-e{args.epochs}"
    validation=json.loads((ROOT/"results"/(tag+"-validation.json")).read_text())
    target=ROOT/"checkpoints"/tag
    h=file_hash(target/"model.safetensors")
    if h!=validation["target_sha256"]:
        raise ValueError("Validation-selected weights changed")
    tok=AutoTokenizer.from_pretrained(target,local_files_only=True);tok.truncation_side="left"
    torch.set_num_threads(int(os.getenv("ROLE_CPU_THREADS","2")))
    model=AutoModelForSequenceClassification.from_pretrained(target,local_files_only=True).eval()
    rows,truth,release=official()
    cache=ROOT/"cache"/("test-logits-"+h[:16]+"-"+args.backend);cache.mkdir(exist_ok=True)
    devices=["cpu"]
    verification={}
    if args.backend=="intel":
        from openvino_worker import embed
        engine=export_intel(model,target,h)
        from program_trial import official_sample
        probe=[r[2] for r in official_sample("select",16)[0]]
        original=logits(model,tok,tokenize(tok,probe),"cpu")
        proposed=embed(engine,tok,probe,7,MAX_LENGTH)
        err=float(np.abs(original-proposed).max())
        differences=int(np.count_nonzero(original.argmax(1)!=proposed.argmax(1)))
        if err>.002 or differences:
            raise ValueError(f"Compact model GPU parity failed: {err}, {differences} differences")
        verification={"validation_examples":len(probe),"max_absolute_logit_difference":err,"argmax_differences":differences}
        devices=["openvino-gpu-fp32","pytorch-cpu-parity"]
        def evaluate(texts):
            values=embed(engine,tok,texts,7,MAX_LENGTH)
            highest=np.partition(values,-2,axis=1)[:,-2:]
            near=np.where(highest[:,1]-highest[:,0]<max(.0005,10*err))[0]
            if len(near):
                values[near]=logits(model,tok,tokenize(tok,[texts[i] for i in near]),"cpu")
            return values
    else:
        device="cuda" if torch.cuda.is_available() else "cpu"
        model.to(device);devices=["pytorch-"+device]
        def evaluate(texts):
            return logits(model,tok,tokenize(tok,texts),device)
    all_logits=[]
    for i in range(0,len(rows),512):
        texts=[r[2] for r in rows[i:i+512]]
        key=f"{i:06d}-"+fingerprint(texts)[:16]
        filename=cache/(key+".npy")
        if filename.is_file():
            batch=np.load(filename,allow_pickle=False)
        else:
            batch=evaluate(texts)
            temp=filename.with_name(filename.name+".partial")
            with temp.open("wb") as stream:np.save(stream,batch,allow_pickle=False)
            temp.replace(filename)
        assert batch.shape==(len(texts),7) and np.isfinite(batch).all()
        all_logits.append(batch)
        if (i//512)%15==0:
            print("Scored",i+len(texts),"/",len(rows),flush=True)
    pred=np.concatenate(all_logits).argmax(1)
    report={"test_rows":len(rows),"judgments":len({r[1] for r in rows}),"target_sha256":h,
            "devices":devices,"backend_verification":verification,"frozen_validation":validation["selected"],
            "model":metric(truth,pred),"published_sentence_only":metric(truth,release),
            "vs_author_released":compare(rows,truth,pred,release)}
    save_json(ROOT/"results"/(tag+"-official-test.json"),report)
    print("Full-test macro-F1",report["model"]["macro_f1"],flush=True)


if __name__=="__main__":
    main()
