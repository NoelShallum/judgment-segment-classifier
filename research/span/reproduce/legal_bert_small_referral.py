"""Matched referral pilot with target-token pooling from a 35M legal encoder."""
import argparse
from itertools import product
import json
from time import perf_counter

import numpy as np
from safetensors.numpy import save_file
from safetensors.torch import load_file as load_torch
import torch
from transformers import BertConfig,BertModel,BertTokenizerFast

import sentence_bert_referral_write as bert
from sentence_document_graph_write import ROOT,OUT,cm,file_hash,paired,row_summary,save_json

TAG="legal-bert-small-referral"
MODEL=ROOT/"references/legal-bert-small"
CACHE=ROOT/"cache"/TAG
IR=CACHE/"target-context.xml"
SPLITS=("fit","select","confirm")


class Pooled(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base=base
    def forward(self, input_ids, attention_mask, target):
        values=self.base(input_ids=input_ids,attention_mask=attention_mask,return_dict=False)[0]
        mask=target.unsqueeze(-1)
        return (values*mask).sum(1)/mask.sum(1).clamp_min(1)


def engine(device):
    import openvino as ov
    prov=json.loads((MODEL/"provenance.json").read_text())
    enc=MODEL/"encoder.safetensors"
    fingerprint=file_hash(enc)
    assert prov["encoder_sha256"]==fingerprint
    if not IR.is_file():
        base=BertModel(BertConfig.from_json_file(str(MODEL/"config.json"))).eval()
        base.load_state_dict(load_torch(enc),strict=True)
        graph=ov.convert_model(Pooled(base),example_input=tuple(torch.ones((2,24),dtype=torch.long) for _ in range(3)))
        for port,name in zip(graph.inputs,("input_ids","attention_mask","target")):
            port.get_tensor().set_names({name})
        graph.reshape({name:ov.PartialShape([-1,-1]) for name in ("input_ids","attention_mask","target")})
        ov.save_model(graph,IR,compress_to_fp16=False)
    ovcore=ov.Core()
    config={"INFERENCE_PRECISION_HINT":"f32","PERFORMANCE_HINT":"LATENCY","NUM_STREAMS":"1",
            "CACHE_DIR":str(CACHE/("compiled-"+device.lower()))}
    if device=="CPU":config["INFERENCE_NUM_THREADS"]="2"
    return ovcore.compile_model(str(IR),device,config),fingerprint


def infer(compiled,tokenizer,windows,width_out=512):
    result=np.empty((len(windows),width_out),np.float32)
    groups={k:[] for k in (16,32,64,96)}
    for i,w in enumerate(windows):
        groups[next(k for k in groups if len(w)<=k)].append(i)
    for width,rows in groups.items():
        size=min(32,1536//width)
        for lo in range(0,len(rows),size):
            batch=rows[lo:lo+size]
            fixed=batch+[batch[-1]]*(size-len(batch))
            x=np.full((size,width),tokenizer.pad_token_id,np.int64)
            m=np.zeros_like(x)
            t=np.zeros_like(x)
            for j,i in enumerate(fixed):
                tokens=windows[i]
                sep=tokenizer.sep_token_id
                start=max((k+1 for k,c in enumerate(tokens[:-1]) if c==sep),default=1)
                x[j,:len(tokens)]=tokens
                m[j,:len(tokens)]=1
                t[j,min(start,len(tokens)-2):len(tokens)-1]=1
            result[batch]=compiled({"input_ids":x,"attention_mask":m,"target":t})[0][:len(batch)]
    assert np.isfinite(result).all()
    return result


def run(device):
    CACHE.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    tokenizer=BertTokenizerFast.from_pretrained(MODEL,local_files_only=True)
    enc,source=engine(device)
    result={}
    for split in SPLITS:
        rows,_=bert.load_views(split)
        path=CACHE/(split+"-"+device+".npy")
        windows=bert.pack_ids(tokenizer,rows,bert.MODES[1])
        if not path.exists():
            start=perf_counter()
            vec=[]
            for lo in range(0,len(windows),256):
                vec.append(infer(enc,tokenizer,windows[lo:lo+256]))
                print(split,min(lo+256,len(windows)),"/",len(windows),round(perf_counter()-start,1),flush=True)
            v=np.concatenate(vec)
            with path.with_suffix(".part").open("wb") as out:
                np.save(out,v,allow_pickle=False)
            path.with_suffix(".part").replace(path)
        result[split]=(rows,np.load(path,allow_pickle=False))
    benchmark(result,device,source)


def benchmark(data,device,source):
    size_record=json.loads((MODEL/"provenance.json").read_text())
    train,validation,confirm=(data[k] for k in SPLITS)
    tags=np.concatenate([r[1][r[3]] for r in train[0]])
    original=bert.current_metrics(validation[0])[1]
    roles=("document scores","target-pooled Legal-BERT Small")
    folder=ROOT/"results"
    plan={"backbone_sha256":source,"parameters":size_record["encoder_parameters"],"rows_fit":len(tags),
          "layers":size_record["layers"],"window_cap":96,"backend":device,"arms":{}}
    estimates={}
    def design(bank,name):
        old=np.concatenate([r[4] for r in bank[0]])
        return np.column_stack((old,bank[1])) if name==roles[1] else old
    for i,name in enumerate(roles):
        x=design(train,name)
        v=design(validation,name)
        peak=(-1.,)
        keep=None
        for rare in (0.,.28,.55):
            for damp,w in bert.fit(x,tags,rare):
                z=bert.response(v,w)
                for gain,margin in product((.45,.65,.85,1.),(.02,.08,.15)):
                    y=bert.apply(validation[0],z,gain,margin)
                    s=row_summary(bert.aggregate(validation[0],y).sum(0))
                    key=(s["macro_f1"],s["accuracy"],rare,damp,gain,margin)
                    if key>peak and s["accuracy"]>=original["accuracy"]-.001:
                        peak=key
                        keep=w
        assert keep is not None
        path=OUT/(TAG+f"-{i}.safetensors")
        save_file(keep,str(path))
        plan["arms"][name]={"validation_macro_f1":peak[0],"validation_accuracy":peak[1],
                             "settings":peak[2:],"weights_sha256":file_hash(path)}
        estimates[name]=(keep,peak[-2:])
    freeze=folder/(TAG+"-frozen.json")
    save_json(freeze,plan)
    out={"freeze_sha256":file_hash(freeze),"splits":{}}
    for label,bank in (("select",validation),("confirm",confirm)):
        rows=bank[0]
        base,b=bert.current_metrics(rows)
        q=out["splits"][label]={"cases":len(rows),"parent":b,"models":{}}
        for name,(w,(g,m)) in estimates.items():
            scores=bert.response(design(bank,name),w)
            mat=bert.aggregate(rows,bert.apply(rows,scores,g,m))
            q["models"][name]={"scores":row_summary(mat.sum(0)),"paired_parent":paired(mat,base)}
        print(label,{k:v["scores"]["macro_f1"] for k,v in q["models"].items()},flush=True)
    save_json(folder/(TAG+".json"),out)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--device",default="GPU",choices=("GPU","CPU"))
    run(parser.parse_args().device)
