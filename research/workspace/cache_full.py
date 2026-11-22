"""Resumable, label-blind cache of the original pooled encoder on an official split."""
import argparse
import csv
import json
import os
from pathlib import Path
import time
import numpy as np
from encoder import components,features,fingerprint
from weight_edit import ORIGINAL,ROOT,file_hash
from import_official import FILES
from program_trial import save_json

BLOCK=512
BASE_HASH="bbe8af20d1524c7535b1cbe68c6d4c3c844c6cc9fccef46c03c9ff56e7c18cea"

def read_text_only(split):
    source=ROOT/"references/official"/(split+".csv")
    assert file_hash(source)==FILES[source.name][1],"Unexpected official split"
    with source.open(encoding="utf-8-sig",newline="") as f:
        return [row["Text"] or "" for row in csv.DictReader(f)]

def block_dir(split):
    return ROOT/"cache"/("full-"+split+"-fp32-v1")

def verify_block(directory,index,expected,row_count):
    data=directory/f"{index:04d}.npy";meta=directory/f"{index:04d}.json"
    if not data.exists() or not meta.exists():return False
    j=json.loads(meta.read_text())
    if j["input_fingerprint"]!=expected or j["base_sha256"]!=BASE_HASH or j["rows"]!=row_count or j["feature_sha256"]!=file_hash(data):raise ValueError("Mismatched block "+str(index))
    a=np.load(data,mmap_mode="r",allow_pickle=False)
    if a.dtype!=np.float32 or a.shape!=(row_count,768):raise ValueError("Shape mismatch")
    return True

def main():
    p=argparse.ArgumentParser();p.add_argument("--split",choices=["val","test"],default="test");p.add_argument("--start",type=int,default=0);p.add_argument("--stop",type=int);p.add_argument("--step",type=int,default=1);p.add_argument("--device",default=None);args=p.parse_args()
    texts=read_text_only(args.split);directory=block_dir(args.split);directory.mkdir(parents=True,exist_ok=True)
    assert file_hash(ORIGINAL)==BASE_HASH,"Wrong checkpoint revision"
    model,tok,device=components(device=args.device)
    if device.startswith("cuda"):
        import torch
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    total=(len(texts)+BLOCK-1)//BLOCK
    print(f"Encoding {len(texts)} {args.split} rows into {total} blocks on {device}; labels are unused.",flush=True)
    for i in range(args.start,min(total,args.stop if args.stop is not None else total),args.step):
        group=texts[i*BLOCK:(i+1)*BLOCK];sig=fingerprint(group)
        if verify_block(directory,i,sig,len(group)):continue
        begin=time.monotonic()
        x=features(model,tok,device,group)
        dest=directory/f"{i:04d}.npy";tmp=dest.with_name(dest.name+f".{os.getpid()}.partial")
        with tmp.open("wb") as h:np.save(h,x,allow_pickle=False)
        os.replace(tmp,dest)
        save_json(directory/f"{i:04d}.json",{"block":i,"first_row":i*BLOCK,"rows":len(group),"split":args.split,"input_fingerprint":sig,"base_sha256":BASE_HASH,"feature_sha256":file_hash(dest),"device":device,"elapsed_s":round(time.monotonic()-begin,2)})
        print("Completed",args.split,i,"of",total,"in",round(time.monotonic()-begin,1),"s",flush=True)
if __name__=="__main__":main()
