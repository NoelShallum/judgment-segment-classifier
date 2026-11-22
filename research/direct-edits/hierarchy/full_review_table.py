"""Substitute a native word table in the full frozen, rerouted 20% reviewer and run FP32 CPU inference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from embedding_write import HERE,ROOT,BASE,plan,digest,file_hash,dump,confusion,metric
from native_memory import data,confusion_docs,native
from hier_model import LABELS
from encoder import TOK
# The workspace OpenVINO install is module-only; prioritize the earlier NumPy and Torch already loaded.
sys.path.append(str(ROOT/"runtime/ov311"))
import openvino as ov
from transformers import BertTokenizerFast
import openvino_worker as seq_bert
import predict_wide_bert_review as runtime
import sentence_phrase_sequence as seq

def get_engine(device):
    spec=json.loads((seq_bert.ART/"source.json").read_text())
    xml=seq_bert.ART/"encoder.xml"; assert file_hash(xml)==spec["xml_sha256"] and file_hash(seq_bert.ART/"encoder.bin")==spec["encoder_bin_sha256"]
    settings={"INFERENCE_PRECISION_HINT":"f32","PERFORMANCE_HINT":"LATENCY","NUM_STREAMS":"1","CACHE_DIR":str(HERE/("compiled-bert" if device=="CPU" else "compiled-bert-gpu"))}
    if device=="CPU":settings["INFERENCE_NUM_THREADS"]="1"
    engine=ov.Core().compile_model(str(xml),device,settings)
    tok=BertTokenizerFast.from_pretrained(str(TOK),local_files_only=True);tok.truncation_side="left"
    return engine,tok

def run(limit,device,retention=False,split="shared",allow_reference_features=False):
    model,*bundle=seq.phrase.load_predictor("target phrases");word=model.word.clone()
    gate_keys=("emitter.lstm.bias_ih_l0","emitter.lstm.bias_ih_l0_reverse"); params=native(model); raw_bias={k:params[k].numpy().copy() for k in gate_keys}
    persistent=HERE/("exact-windows-"+device.lower());persistent.mkdir(exist_ok=True)
    seq.phrase.load_predictor=lambda _: (model,*bundle)
    thread_set=torch.set_num_threads;torch.set_num_threads=lambda _:thread_set(1)
    raw_embed=seq_bert.embed_ids;cache={};engine_data=None
    def compile_once(device):
        nonlocal engine_data
        if engine_data is None:engine_data=get_engine(device)
        return engine_data
    def vector(engine,rows,max_length):
        key=[tuple(s) for s in rows]
        names={k:persistent/(hashlib.sha256(np.asarray(k,dtype="<i8").tobytes()).hexdigest()+".npy") for k in key}
        for k in dict.fromkeys(key):
            if k not in cache and names[k].exists():cache[k]=np.load(names[k],allow_pickle=False)
            elif k not in cache and allow_reference_features and (HERE/"exact-windows-gpu-cached"/names[k].name).exists():cache[k]=np.load(HERE/"exact-windows-gpu-cached"/names[k].name,allow_pickle=False)
        missing=list(dict.fromkeys(k for k in key if k not in cache))
        if missing:
            value=raw_embed(engine,[list(k) for k in missing],max_length=max_length)
            for k,v in zip(missing,value):
                cache[k]=v;tmp=names[k].with_suffix(".part")
                with tmp.open("wb") as fp:np.save(fp,v,allow_pickle=False)
                tmp.replace(names[k])
        return np.asarray([cache[k] for k in key])
    seq_bert.get_engine=compile_once;seq_bert.embed_ids=vector
    predictor=runtime.predictor(device,.20)
    idx={s:i for i,s in enumerate(LABELS)};docs=data(split,limit)
    if split=="shared":reference={k:json.loads((HERE.parent/"pooler/shared-cases"/(k+".json")).read_text())["predictions"]["unchanged"] for k,*_ in docs}
    else:
        db=np.load(HERE/("strong-"+split+".npz"));reference={k:db["pred"][a:b] for k,a,b in zip(db["keys"],db["ends"],db["ends"][1:])}
    saved={"method":"Shipped full 20% reviewer, dynamic routing recomputed per candidate, entire judgment and frozen official boundaries",
           "parent_cpu_reference":"Unchanged from cached unedited full FP32 reviewer; additionally replayed on two judgments for label parity.",
           "gradient_steps":0,"tensor":"sent_encoder.emb.weight","rows":sum(len(g) for _,_,g,_ in docs),"keys":[d[0] for d in docs],"arms":{}}
    exp="retention" if retention else "word-table"
    tar=HERE/f"full-review-{exp}-{split}-{limit}.json"
    saved["cohort"]=split;saved["reused_legacy_fp32_vectors_allowed"]=allow_reference_features
    ys=np.concatenate([d[2] for d in docs]);ends=np.r_[0,np.cumsum([len(d[2]) for d in docs])]
    base=np.concatenate([reference[d[0]] for d in docs]);sc=confusion_docs(ys,base,ends)
    saved["incumbent"]=metric(sc.sum(0))
    saved["device"]=device
    folder=HERE/("full-review-cases-"+exp+"-"+split+"-"+device.lower());folder.mkdir(exist_ok=True)
    if retention:
        saved["tensor"]=list(gate_keys)
        proof=json.loads((HERE/"uniform-retention-gate.json").read_text());saved["freeze_sha256"]=file_hash(HERE/"uniform-retention-gate.json")
    for arm in (("base","targeted","gate-shuffled") if retention else ("base","probe-res","probe-res-shuf")):
        model.word=word.clone()
        for k in gate_keys:params[k].numpy()[:]=raw_bias[k]
        if arm!="base" and retention:
            from safetensors.numpy import load_file
            patch=HERE/(proof["target_file"] if arm=="targeted" else proof["controls"][arm]["file"]);expected=proof["target_sha256"] if arm=="targeted" else proof["controls"][arm]["sha256"];assert file_hash(patch)==expected
            for k,v in load_file(str(patch)).items():params[k].numpy()[:]=v
        elif arm!="base":
            indices,delta=plan(model,arm);model.word[indices]+=torch.from_numpy(delta)*.28
        found=[];start=time.monotonic();referred=0;positions=[]
        active=docs[:2] if arm=="base" else docs
        for i,(k,txt,g,_) in enumerate(active):
            receipt=folder/f"{arm}-{k}.json"
            if receipt.exists():
                result=json.loads(receipt.read_text());pred=result["pred"]
            else:
                result=predictor(txt);pred=[idx[s] for s in result["labels"]]
                dump(receipt,dict(pred=pred,reviewed=result["reviewed"],referrals=result["referrals"]))
            if arm=="base":np.testing.assert_array_equal(reference[k],pred)
            found.extend(pred);referred+=result["reviewed"];positions.append(result["referrals"])
            if (i+1)%4==0:print(arm,i+1,"of",len(active),round(time.monotonic()-start),flush=True)
        if arm=="base":
            saved["cpu_baseline_replay_keys"]=[d[0] for d in active];saved["baseline_cpu_parity_disagreements"]=0;dump(tar,saved);continue
        h=np.asarray(found);cm=confusion_docs(ys,h,ends)
        rng=np.random.default_rng(9122);dd=[]
        for j in rng.integers(len(active),size=(2000,len(active))):
            a,b=metric(cm[j].sum(0)),metric(sc[j].sum(0));dd.append([a["macro_f1"]-b["macro_f1"],a["accuracy"]-b["accuracy"]])
        saved["arms"][arm]=dict(**metric(cm.sum(0)),corrected=int(((h!=base)&(h==ys)).sum()),harmed=int(((h!=base)&(base==ys)).sum()),changed=int((h!=base).sum()),paired95=np.quantile(dd,[.025,.975],axis=0).T.tolist(),weight_table_sha256=digest(model.word.numpy()),gate_sha256={k:digest(params[k].numpy()) for k in gate_keys},reviewed=referred,unique_BERT_windows_cached=len(cache))
        tmp=HERE/(tar.stem+"-"+arm+".part");f=tmp.with_suffix(".npz")
        with tmp.open("wb") as fp:np.savez(fp,keys=saved["keys"],gold=ys,pred=h,base=base,ends=ends)
        tmp.replace(f);dump(tar,saved)
        print(arm,saved["arms"][arm]["macro_f1"],saved["arms"][arm]["accuracy"],len(h),flush=True)
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--limit",type=int,default=16);parser.add_argument("--device",choices=("CPU","GPU"),default="CPU");parser.add_argument("--retention",action="store_true");parser.add_argument("--split",choices=("shared","select","confirm"),default="shared");parser.add_argument("--allow-reference-features",action="store_true");args=parser.parse_args();run(args.limit,args.device,args.retention,args.split,args.allow_reference_features)
