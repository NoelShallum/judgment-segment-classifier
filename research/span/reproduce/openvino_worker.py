"""Benchmark a local Intel FP32 backend against existing PyTorch features before scoring."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import openvino as ov
from transformers import BertTokenizerFast

from cache_full import BASE_HASH, BLOCK, block_dir, read_text_only, verify_block
from encoder import TOK, fingerprint
from program_trial import save_json
from weight_edit import ORIGINAL, ROOT, file_hash, tensor


ART = ROOT / "runtime/openvino-bert-fp32"


def export(source=ORIGINAL, directory=ART):
    import torch
    from encoder import components
    directory.mkdir(exist_ok=True, parents=True)
    xml = directory / "encoder.xml"
    spec = directory / "source.json"
    expected_hash=BASE_HASH if source==ORIGINAL else file_hash(source)
    if xml.is_file() and spec.is_file() and (directory/"encoder.bin").is_file():
        if json.loads(spec.read_text())["source"] != expected_hash:
            raise ValueError("Wrong exported source")
        return xml
    if file_hash(source) != expected_hash:
        raise ValueError("Wrong source checkpoint")
    model, tok, _ = components(checkpoint=source,device="cpu")
    class Encoder(torch.nn.Module):
        def __init__(self, bert):
            super().__init__()
            self.bert = bert
        def forward(self, input_ids, attention_mask):
            return self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)[1]
    wrapper = Encoder(model.bert).eval()
    example=(torch.ones(2,16,dtype=torch.long),torch.ones(2,16,dtype=torch.long))
    with torch.inference_mode():
        representation=ov.convert_model(wrapper, example_input=example)
    for port, name in zip(representation.inputs, ("input_ids","attention_mask")):
        port.get_tensor().set_names({name})
    representation.reshape({name: ov.PartialShape([-1,-1]) for name in ("input_ids","attention_mask")})
    ov.save_model(representation, xml, compress_to_fp16=False)
    save_json(spec,{"source":expected_hash,"openvino":ov.__version__,"encoder_bin_sha256":file_hash(directory/"encoder.bin"),
                    "xml_sha256":file_hash(xml),"compress_to_fp16":False})
    return xml


def get_engine(device,source=ORIGINAL,directory=ART):
    path=export(source,directory)
    core=ov.Core()
    settings={"INFERENCE_PRECISION_HINT":"f32","PERFORMANCE_HINT":"LATENCY",
              "CACHE_DIR":str(directory/"compiled"),"NUM_STREAMS":"1"}
    if device == "CPU":
        settings["INFERENCE_NUM_THREADS"] = "3"
    print("Compiling FP32 model on",core.get_property(device,"FULL_DEVICE_NAME"),flush=True)
    engine=core.compile_model(path,device,settings)
    tok=BertTokenizerFast.from_pretrained(TOK,local_files_only=True);tok.truncation_side="left"
    return engine,tok


def embed(engine,tok,texts,output_width=768,max_length=512):
    ids=tok(texts,truncation=True,max_length=max_length,return_attention_mask=False,return_token_type_ids=False)["input_ids"]
    return embed_ids(engine,ids,output_width,max_length)


def embed_ids(engine,ids,output_width=768,max_length=512):
    out=np.empty((len(ids),output_width),dtype=np.float32)
    widths=sorted({min(max_length,width) for width in (16,32,64,128,256,512)} | {max_length})
    buckets={width:[] for width in widths}
    for i,tokens in enumerate(ids):
        if not 0<len(tokens)<=max_length:
            raise ValueError("Tokenized model input exceeds the encoder window or is empty")
        width=next(limit for limit in widths if len(tokens)<=limit)
        buckets[width].append(i)
    for width,indexes in buckets.items():
        batch_size=min(24,max(1,1536//width))
        for start in range(0,len(indexes),batch_size):
            group=indexes[start:start+batch_size]
            # Repeat the last real row only to give the GPU kernel a fixed shape.
            # BERT has no cross-example attention; its padded predictions are discarded.
            fixed=group+[group[-1]]*(batch_size-len(group))
            x=np.zeros((batch_size,width),dtype=np.int64)
            mask=np.zeros_like(x)
            for row,index in enumerate(fixed):
                n=len(ids[index])
                x[row,:n]=ids[index]
                mask[row,:n]=1
            values=engine({"input_ids":x,"attention_mask":mask})[engine.output(0)]
            out[group]=values[:len(group)]
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--device",choices=["GPU","CPU"],default="GPU")
    ap.add_argument("--advance",action="store_true")
    args=ap.parse_args()
    engine,tok=get_engine(args.device)
    directory=block_dir("test");texts=read_text_only("test")
    actual_device="openvino-"+args.device.lower()+"-fp32"
    # Compare with two completed PyTorch blocks that span unrelated judgments.
    head_paths=(ORIGINAL,ROOT/"checkpoints/selected-direct/model.safetensors")
    heads=[(tensor(p,"classifier.weight"),tensor(p,"classifier.bias")) for p in head_paths]
    parity=[]
    for block in (0,1):
        group=texts[block*BLOCK:(block+1)*BLOCK]
        assert verify_block(directory,block,fingerprint(group),len(group))
        base=np.load(directory/f"{block:04d}.npy",allow_pickle=False)
        start=time.monotonic();proposed=embed(engine,tok,group);elapsed=time.monotonic()-start
        delta=np.abs(proposed-base)
        changed=[int(np.count_nonzero((proposed@w.T+b).argmax(1)!=(base@w.T+b).argmax(1))) for w,b in heads]
        row={"block":block,"elapsed_s":round(elapsed,2),"mean_absolute_feature_difference":float(delta.mean()),
             "max_absolute_feature_difference":float(delta.max()),"argmax_differences":changed}
        parity.append(row);print("Parity",row,flush=True)
    comparable=all(r["mean_absolute_feature_difference"]<.00005 and r["max_absolute_feature_difference"]<.0005
                   and r["argmax_differences"]==[0,0] for r in parity)
    save_json(ROOT/"results"/(actual_device+"-parity.json"),{"device":actual_device,"openvino":ov.__version__,
              "source":BASE_HASH,"rows":sum(1 for _ in range(2*BLOCK)),"comparable":comparable,"blocks":parity})
    if not comparable:
        raise RuntimeError("Backend did not pass numerical equivalence check")
    if not args.advance:
        return
    # The CPU and GPU commit into independent directories. They can finish the
    # same block concurrently without overwriting either data/manifest pair.
    output_directory=directory/actual_device
    output_directory.mkdir(exist_ok=True)
    for i in reversed(range((len(texts)+BLOCK-1)//BLOCK)):
        group=texts[i*BLOCK:(i+1)*BLOCK];sig=fingerprint(group)
        if verify_block(directory,i,sig,len(group)) or verify_block(output_directory,i,sig,len(group)):
            continue
        start=time.monotonic();x=embed(engine,tok,group)
        path=output_directory/f"{i:04d}.npy";tmp=path.with_name(path.name+f".{os.getpid()}.partial")
        with tmp.open("wb") as stream:np.save(stream,x,allow_pickle=False)
        tmp.replace(path)
        elapsed=round(time.monotonic()-start,2)
        save_json(output_directory/f"{i:04d}.json",{"block":i,"first_row":i*BLOCK,"rows":len(group),"split":"test",
                  "input_fingerprint":sig,"base_sha256":BASE_HASH,"feature_sha256":file_hash(path),
                  "device":actual_device,"elapsed_s":elapsed})
        print("Completed",actual_device,i,"in",elapsed,flush=True)


if __name__ == "__main__":
    main()
