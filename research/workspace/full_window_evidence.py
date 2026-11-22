"""Evaluate full-excerpt coverage while preserving the published tail-only reference."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from transformers import BertTokenizerFast

from cache_full import read_text_only
from encoder import TOK,components,features,fingerprint
from paragraph_transfer import read,compare,INDEX
from program_trial import save_json
from weight_edit import LABELS,ORIGINAL,ROOT,tensor
from windowing import token_windows


def coverage(tok,texts):
    long={};lengths=[]
    for start in range(0,len(texts),1024):
        encoded=tok(texts[start:start+1024],add_special_tokens=False,truncation=False,return_attention_mask=False,
                    return_token_type_ids=False,verbose=False)["input_ids"]
        for j,seq in enumerate(encoded):
            lengths.append(len(seq))
            if len(seq)>510:long[start+j]=seq
    return long,lengths


def embed_rows(ids,backend):
    if backend=="intel":
        from openvino_worker import get_engine,embed_ids
        record=json.loads((ROOT/"results/openvino-gpu-fp32-parity.json").read_text())
        if not record["comparable"]:raise ValueError("Unverified GPU backend")
        engine,tok=get_engine("GPU")
        return embed_ids(engine,ids)
    import torch
    model,tok,device=components()
    out=[]
    with torch.inference_mode():
        for start in range(0,len(ids),4):
            batch=tok.pad([{"input_ids":i} for i in ids[start:start+4]],padding=True,return_tensors="pt").to(device)
            out.extend(model.bert(**batch).pooler_output.detach().cpu().numpy())
    return np.asarray(out,dtype=np.float32)


def paragraph_report(rows,old,full,long,stats):
    ref=ROOT/"checkpoints/selected-direct/model.safetensors"
    heads={"original":ORIGINAL,"direct":ref}
    associative=ROOT/"checkpoints/associative-direct/model.safetensors"
    if associative.is_file():heads["associative"]=associative
    before={};after={}
    for name,p in heads.items():
        w,b=tensor(p,"classifier.weight"),tensor(p,"classifier.bias")
        before[name]=INDEX[(old@w.T+b).argmax(1)]
        after[name]=INDEX[(full@w.T+b).argmax(1)]
    overlap=json.loads((ROOT/"results/prr-paragraph-transfer.json").read_text())["cases_with_matching_spans"]
    keep=np.array([r[0] not in overlap for r in rows])
    report={"protocol":stats,"projection":"Fact, Argument (both parties), Analysis and Verdict. Other predictions count as incorrect.",
            "tail":compare(rows,before),"full_excerpt":compare(rows,after),
            "changed_paragraphs":{k:int(np.count_nonzero(before[k]!=after[k])) for k in heads},
            "long_paragraphs_only":{"tail":compare([rows[i] for i in long],{k:v[long] for k,v in before.items()}),
                                     "full_excerpt":compare([rows[i] for i in long],{k:v[long] for k,v in after.items()})}}
    report["no_detected_training_match"]={"tail":compare([r for r,a in zip(rows,keep) if a],{k:v[keep] for k,v in before.items()}),
                   "full_excerpt":compare([r for r,a in zip(rows,keep) if a],{k:v[keep] for k,v in after.items()})}
    save_json(ROOT/"results/full-window-paragraphs.json",report)
    print("Paragraph direct tail/full",report["tail"]["models"]["direct"]["four_role_macro_f1"],
          report["full_excerpt"]["models"]["direct"]["four_role_macro_f1"],flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("split",choices=["paragraph","test"]);ap.add_argument("--backend",choices=["torch","intel"],default="torch");args=ap.parse_args()
    rows=read("PRR.zip") if args.split=="paragraph" else None
    texts=[r[2] for r in rows] if rows else read_text_only("test")
    tok=BertTokenizerFast.from_pretrained(TOK,local_files_only=True)
    ids,lengths=coverage(tok,texts);long=sorted(ids);file=ROOT/"cache"/("full-window-"+args.split+"-"+fingerprint(texts)[:16]+".npz")
    print("Full passages",len(texts),"over 510 tokens",len(long),"maximum content tokens",max(lengths),flush=True)
    if file.is_file():
        saved=np.load(file,allow_pickle=False);assert saved["indices"].tolist()==long
        values=saved["values"]
    else:
        all_windows=[];owners=[];weights=[]
        for j,i in enumerate(long):
            for tokens,weight in token_windows(tok,ids[i],512):
                all_windows.append(tokens);owners.append(j);weights.append(weight)
        values=np.zeros((len(long),768),np.float64)
        if all_windows:
            print("Encoding",len(all_windows),"overlapping windows",flush=True)
            vectors=embed_rows(all_windows,args.backend)
            np.add.at(values,owners,vectors*np.asarray(weights,dtype=np.float64)[:,None])
            values/=np.bincount(owners,weights=weights)[:,None]
        values=values.astype(np.float32)
        temp=file.with_suffix(".partial")
        with temp.open("wb") as stream:np.savez(stream,indices=np.asarray(long,dtype=np.int32),values=values)
        temp.replace(file)
    stats={"split":args.split,"rows":len(texts),"over_capacity":len(long),"maximum_content_tokens":max(lengths),
           "feature_file":file.name,"whole_population_fingerprint":fingerprint(texts),
           "pooling":"Cover all content with 510-token windows and nominal overlap 128. Average window logits by nonduplicated coverage length, using midpoint boundaries. Equivalent to averaging pooled hidden vectors before the unchanged linear classifier.",
           "uses_sentence_or_paragraph_flag":False}
    save_json(ROOT/"results"/("window-coverage-"+args.split+".json"),stats)
    if rows:
        path=ROOT/"cache"/("external-prr-fp32-"+fingerprint(texts)[:24]+".npy")
        if path.is_file():old=np.load(path,allow_pickle=False)
        else:
            model,tok,device=components();old=features(model,tok,device,texts,"external-prr-torch-fp32")
        full=old.copy();full[long]=values
        paragraph_report(rows,old,full,long,stats)


if __name__=="__main__":main()
