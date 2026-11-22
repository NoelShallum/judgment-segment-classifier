"""Apply prespecified frozen local checkpoints to exactly the approved Gemini inputs, using the product windowing policy."""
import argparse
import gc
import hashlib
import json
import numpy as np
import torch
from encoder import components,fingerprint
from mixed_encoder_trial import CONTEXT
from seven_paths import FOLDERS,folder_for
from program_trial import save_json
from weight_edit import LABELS,ORIGINAL,ROOT,file_hash,tensor
from windowing import token_windows

@torch.inference_mode()
def vectors(source,texts):
    cache=ROOT/"cache"/("matched-windows-"+file_hash(source)[:16]+"-"+fingerprint(texts)[:20]+".npy")
    if cache.is_file():return np.load(cache,allow_pickle=False)
    model,tok,device=components(checkpoint=source)
    content=tok(texts,add_special_tokens=False,return_attention_mask=False,return_token_type_ids=False,verbose=False)["input_ids"]
    windows=[(i,ids,weight) for i,seq in enumerate(content) for ids,weight in token_windows(tok,seq,512)]
    windows.sort(key=lambda a:len(a[1]))
    out=np.zeros((len(texts),768),dtype=np.float64)
    mass=np.zeros(len(texts),dtype=np.float64)
    start=0
    while start<len(windows):
        group=windows[start:start+min(12,max(1,768//len(windows[start][1])))]
        inputs=tok.pad([{"input_ids":ids} for _,ids,_ in group],padding=True,return_tensors="pt").to(device)
        pooled=model.bert(**inputs).pooler_output.detach().cpu().numpy()
        for j,(i,_,weight) in enumerate(group):
            out[i]+=pooled[j].astype(np.float64)*weight
            mass[i]+=weight
        start+=len(group)
    result=(out/mass[:,None]).astype(np.float32)
    temp=cache.with_suffix(".partial")
    with temp.open("wb") as f:np.save(f,result,allow_pickle=False)
    temp.replace(cache)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--corpus",choices=tuple(FOLDERS),default="pilot");args=ap.parse_args()
    folder=folder_for(args.corpus)
    rows=json.loads((folder/"approved.json").read_text())
    groups=[(ORIGINAL,{"original":ORIGINAL,"direct":ROOT/"checkpoints/selected-direct/model.safetensors",
        "associative_direct":ROOT/"checkpoints/associative-direct/model.safetensors"}),
        (CONTEXT,{"context_parent":CONTEXT,"associative_context":ROOT/"checkpoints/associative-context/model.safetensors"})]
    out={}
    for source,heads in groups:
        x=vectors(source,[r["text"] for r in rows])
        for k,p in heads.items():
            pred=(x@tensor(p,"classifier.weight").T+tensor(p,"classifier.bias")).argmax(1)
            out[k]={"sha256":file_hash(p),"predictions":{r["id"]:LABELS[j] for r,j in zip(rows,pred)}}
        gc.collect()
    save_json(folder/"local.json",out)
    print("Completed internal inference for",len(rows),"inputs and",len(out),"checkpoints.",flush=True)
if __name__=="__main__":main()
