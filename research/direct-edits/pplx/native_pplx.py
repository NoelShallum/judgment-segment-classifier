"""Capture the verified Qwen3.5-based contextual model before its native projection."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import quote

import numpy as np
import torch
from transformers import AutoModel

HERE=Path(__file__).resolve().parent


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def put(path,value):
    stage=path.with_suffix(".tmp")
    stage.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    stage.replace(path)


def t4():
    rows=subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],text=True).strip().splitlines()
    assert len(rows)==1 and rows[0].split(",")[0].strip() in {"Tesla T4","NVIDIA T4","T4"},rows
    assert torch.cuda.is_available() and "T4" in torch.cuda.get_device_name(0)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    return {"devices":rows,"torch":str(torch.__version__),"precision":"FP32"}


def fetch(model, where):
    where.mkdir(exist_ok=True)
    for file,digest in model["sha256"].items():
        path=where/file
        if path.exists() and sha(path)==digest:
            continue
        url=f"https://huggingface.co/{model['model_id']}/resolve/{model['revision']}/{quote(file)}"
        part=path.with_suffix(".partial")
        subprocess.run(["curl","-fLsS","--retry","3",url,"-o",str(part)],check=True)
        assert sha(part)==digest,file
        part.replace(path)


@torch.inference_mode()
def capture(model, chunks):
    fea={k:v.to(model.device) for k,v in model.prepare_inputs(chunks,task="document").items()}
    hid=model(input_ids=fea["input_ids"],attention_mask=fea["attention_mask"],use_cache=False,return_dict=True).last_hidden_state.float()
    prefix=torch.cat((hid.new_zeros(hid.shape[0],1,hid.shape[-1]),hid.cumsum(1)),1)
    a,b=fea["chunk_token_spans"].unbind(-1)
    r=torch.arange(len(chunks),device=hid.device)[:,None]
    work=(prefix[r,b]-prefix[r,a])/(b-a).clamp(min=1)[...,None]
    return work,fea


def fragments(model, texts, budget):
    tok=model.tokenizer
    opts=dict(add_special_tokens=False,split_special_tokens=True,return_offsets_mapping=True,verbose=False)
    lengths=tok(texts,**opts)["offset_mapping"]
    unit=[]
    for i,(text,spans) in enumerate(zip(texts,lengths)):
        if not spans:
            raise ValueError(f"Empty tokenizer result: {i}")
        cap=budget-10
        for j in range(0,len(spans),cap):
            if j==0 and len(spans)<=cap:
                piece=text
            else:
                a=spans[j][0]
                b=spans[min(j+cap,len(spans))-1][1]
                piece=text[a:b]
            unit.append((i,piece,min(cap,len(spans)-j)))
    return unit


def corpus_blocks(units,budget):
    group,size=[],1
    for n,text,tokens in units:
        cost=tokens+5
        if group and size+cost>budget:
            yield group
            group,size=[],1
        group.append((n,text,tokens))
        size+=cost
    if group:
        yield group


def batches(units,budget):
    ordered=sorted(units,key=lambda x:x[-1])
    start=0
    while start<len(ordered):
        step=min(16,max(1,4096//(ordered[start][-1]+20)))
        yield ordered[start:start+step]
        start+=step


def encode_case(model, records, budget):
    texts=[r[1] for r in records]
    units=fragments(model,texts,budget)
    annotations={}
    for mode in ("contextual","isolated"):
        accum=np.zeros((len(records),1024),dtype=np.float32)
        count=np.zeros(len(records),dtype=np.float32)
        tokens=0
        blocks=corpus_blocks(units,budget) if mode=="contextual" else batches(units,budget)
        for group in blocks:
            chunks=[[item[1] for item in group]] if mode=="contextual" else [[item[1]] for item in group]
            vec,fea=capture(model,chunks)
            tokens+=int(fea["attention_mask"].sum().item())
            take=vec[0] if mode=="contextual" else vec[:,0]
            for (index,_,n), feat in zip(group,take.cpu().numpy(),strict=True):
                accum[index]+=n*feat
                count[index]+=n
        assert (count>0).all()
        annotations[mode]=accum/count[:,None]
        annotations[mode+"_tokens"]=tokens
    annotations["gold"]=np.asarray([r[2] for r in records],np.int8)
    annotations["row"]=np.asarray([r[0] for r in records],np.int64)
    return annotations


def smoke(model, weights, records):
    chunks=[[r[1] for r in records[:3]]]
    x,fea=capture(model,chunks)
    with torch.inference_mode():
        upstream=model._pool(fea,"document")
        unedited=torch.nn.functional.linear(x,weights)
        source_error=float((unedited-upstream).abs().max())
        assert source_error<1e-6
        actual=model.contextual_projection.weight
        probe=torch.randn((7,1024),generator=torch.Generator().manual_seed(1223)).to(actual.device)*.005
        actual[:7].copy_(probe)
        direct=model._pool(fea,"document")
        algebraic=torch.nn.functional.linear(x,probe)
        err=float((direct[..., :7]-algebraic).abs().max())
        actual.copy_(weights)
        assert err<1e-5
    return {"native_projection_max_difference":source_error,"native_rewrite_max_difference":err,
            "tokens":int(fea["input_ids"].numel()),"role_probe_shape":[7,1024]}


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage",choices=("smoke","full"))
    args=cli.parse_args()
    device=t4()
    protocol=json.loads((HERE/"protocol.json").read_text())
    texts=json.loads((HERE/"cases.json").read_text())
    source=HERE/"pinned-model"
    fetch(protocol["model"],source)
    model=AutoModel.from_pretrained(str(source),trust_remote_code=True,local_files_only=True,
                                   dtype=torch.float32,attn_implementation="sdpa").eval().requires_grad_(False).to("cuda")
    weight=model.contextual_projection.weight.detach().clone()
    assert weight.shape == (1024,1024)
    evidence=smoke(model,weight,texts[protocol["shared"][0]])
    np.save(HERE/"projection.npy",weight.cpu().numpy(),allow_pickle=False)
    put(HERE/"native-parity.json",{**device,**evidence,"checkpoint_revision":protocol["model"]["revision"]})
    print("PPLX source replay verified",evidence,flush=True)
    out=HERE/"features"
    out.mkdir(exist_ok=True)
    chosen=protocol["fit"]+protocol["select"]+protocol["shared"]
    if args.stage=="smoke":
        chosen=protocol["fit"][:1]+protocol["shared"][:1]
    for cid in chosen:
        feat=out/(cid+".npz")
        receipt=out/(cid+".json")
        if feat.exists() and receipt.exists() and json.loads(receipt.read_text())["sha256"]==sha(feat):
            continue
        start=time.monotonic()
        roles=encode_case(model,texts[cid],protocol["context_token_budget"])
        temp=feat.with_suffix(".tmp")
        with temp.open("wb") as f:
            np.savez(f,**roles)
        temp.replace(feat)
        ans={"case":cid,"rows":len(roles["gold"]),"sha256":sha(feat),"seconds":time.monotonic()-start,
             "tokens_contextual":roles["contextual_tokens"],"tokens_isolated":roles["isolated_tokens"],"backend":device}
        put(receipt,ans)
        print("PPLX complete",ans,flush=True)


if __name__=="__main__":
    main()
