"""Run the authors' original float32 encoder on standalone excerpts."""
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
from weight_edit import LABELS, ORIGINAL, ROOT

TOK=ROOT/"references/tokenizer"

def components(checkpoint=ORIGINAL,device=None):
    torch.set_num_threads(int(os.environ.get("ROLE_CPU_THREADS","4")))
    device=device or ("cuda" if torch.cuda.is_available() else "cpu")
    config=BertConfig.from_json_file(str(TOK/"config.json"))
    config.num_labels=len(LABELS)
    config.id2label=dict(enumerate(LABELS));config.label2id={v:k for k,v in enumerate(LABELS)}
    config.architectures=["BertForSequenceClassification"]
    model=BertForSequenceClassification(config)
    state=load_file(str(checkpoint),device="cpu")
    result=model.load_state_dict(state,strict=False)
    if result.missing_keys or set(result.unexpected_keys)-{"bert.embeddings.position_ids","bert.embeddings.token_type_ids"}:
        raise ValueError(str(result))
    del state
    model.eval().to(device)
    tokenizer=BertTokenizerFast.from_pretrained(str(TOK),local_files_only=True)
    tokenizer.truncation_side="left"  # The authors retain the last 510 content tokens.
    return model,tokenizer,device

def fingerprint(texts):
    h=hashlib.sha256()
    for t in texts:
        b=t.encode();h.update(len(b).to_bytes(8,"big"));h.update(b)
    return h.hexdigest()

@torch.inference_mode()
def features(model,tokenizer,device,texts,cache_tag=None):
    if not texts:raise ValueError("No texts to encode")
    cache=ROOT/"cache";cache.mkdir(exist_ok=True)
    source=fingerprint(texts)
    fn=cache/(cache_tag+"-"+source[:16]+".npy") if cache_tag else None
    if fn and fn.exists():
        arr=np.load(fn,allow_pickle=False)
        if arr.shape!=(len(texts),model.config.hidden_size):raise ValueError("Invalid feature cache")
        return arr
    enc=tokenizer(texts,padding=False,truncation=True,max_length=512,return_attention_mask=False,return_token_type_ids=False)["input_ids"]
    order=sorted(range(len(enc)),key=lambda i:len(enc[i]))
    out=np.zeros((len(texts),model.config.hidden_size),dtype="float32")
    begin=time.monotonic();n=0;last=begin
    while n<len(order):
        length=len(enc[order[n]])
        batch=min(32 if device=="cuda" else 12,max(1,(4096 if device=="cuda" else 900)//max(length,1)))
        ids=order[n:n+batch];pack=tokenizer.pad([{"input_ids":enc[i]} for i in ids],padding=True,return_tensors="pt").to(device)
        vectors=model.bert(**pack,return_dict=True).pooler_output
        out[ids]=vectors.detach().float().cpu().numpy();n+=len(ids)
        if time.monotonic()-last>30:
            print(f"{cache_tag}: encoded {n}/{len(texts)}, {time.monotonic()-begin:.0f}s",flush=True);last=time.monotonic()
    if fn:
        temp=fn.with_name(fn.name+".partial")
        with temp.open("wb") as f:np.save(f,out,allow_pickle=False)
        os.replace(temp,fn)
    return out

def write_config(target):
    config=BertConfig.from_json_file(str(TOK/"config.json"));config.num_labels=7
    config.id2label=dict(enumerate(LABELS));config.label2id={v:k for k,v in enumerate(LABELS)}
    config.architectures=["BertForSequenceClassification"]
    config.save_pretrained(str(target))
    tok=BertTokenizerFast.from_pretrained(str(TOK),local_files_only=True)
    tok.truncation_side="left";tok.init_kwargs.update(truncation_side="left",model_max_length=512)
    tok.save_pretrained(str(target))

if __name__=="__main__":
    import gc
    from weight_edit import prior_bias
    samples=["Notice was served on 3 May. We therefore allow the appeal.","Counsel for the respondent contends that the appeal is barred by limitation.","Whether the proviso applies to these facts is the issue for determination."]
    scores=[]
    for p in (ORIGINAL,ROOT/"checkpoints/direct-prior/model.safetensors"):
        model,tok,device=components(p)
        with torch.inference_mode():
            ids=tok(samples,padding=True,return_tensors="pt").to(device)
            scores.append(model(**ids).logits.cpu().numpy())
        del model;gc.collect()
    error=float(np.max(np.abs(scores[1]-scores[0]-prior_bias()[None,:])))
    assert error<1e-4,error
    write_config(ROOT/"checkpoints/direct-prior")
    print(json.dumps({"inputs":len(samples),"max_direct_forward_delta_error":error,"base_predictions":[LABELS[i] for i in scores[0].argmax(1)],"edited_predictions":[LABELS[i] for i in scores[1].argmax(1)]}))
