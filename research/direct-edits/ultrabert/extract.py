"""Extract exact sufficient statistics for sentence pooling and a final-MLP weight write."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import AutoModel

from prepare import ROOT, LABELS, docs, save

ARCH = "hf-native-5.21.0-fp32-v1"


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def groups(stage):
    pool = json.loads((ROOT/"pools.json").read_text())["pools"]
    profiles = {"quick":(3,2,3,4), "screen":(9,5,6,8), "expanded":(24,14,12,16),
                "deep":(48,28,20,24)}
    if stage=="common":
        rc = json.loads((ROOT.parent/"shared-confirmation.json").read_text())
        ids = set(rc["small_validation_judgment_ids"])
        val = [{"id":cid,"ncase":len(rows),"rows":rows} for cid,rows in docs("val.csv") if cid in ids]
        assert len(val)==len(ids)
        return {"common":sorted(val,key=lambda r:rc["small_validation_judgment_ids"].index(r["id"]))}
    nfit,nedit,nsel,nconf=profiles[stage]
    ans = {s:[item for r in range(7) for item in pool[s][str(r)][:n]]
           for s,n in [("fit",nfit),("edit",nedit)]}
    ans["select"]=pool["select"]["short"][:nsel]
    ans["confirm"]=pool["confirm"]["short"][:nconf]
    if stage!="quick":
        ans["natural_select"]=pool["select"]["mixed"][:2 if stage=="screen" else 6]
    ids = {s:{i["id"] for i in items} for s,items in ans.items()}
    assert not any(ids[a]&ids[b] for a in ["fit","edit"] for b in ["select","confirm"])
    assert not ids["fit"]&ids["edit"] and not ids["select"]&ids["confirm"]
    return ans


def identifier(item):
    return sha(item["id"]+"\0"+"\1".join(str(r["row"])+"\0"+r["text"] for r in item["rows"]))


def blocks(item, tokenizer, cap, mode):
    """Tokenize sentence text in full. Newline separators have no sentence owner."""
    join = tokenizer.encode("\n",add_special_tokens=False).ids
    ids=[]; owners=[]
    for i,row in enumerate(item["rows"]):
        tokens = tokenizer.encode(row["text"],add_special_tokens=False).ids
        assert tokens
        if ids and (mode=="alone" or len(ids)+len(join)+len(tokens)>cap):
            yield ids,owners
            ids=[]; owners=[]
        if ids:
            ids.extend(join); owners.extend([-1]*len(join))
        while len(tokens)>cap:
            yield tokens[:cap],[i]*cap
            tokens=tokens[cap:]
        ids+=tokens; owners += [i]*len(tokens)
    if ids: yield ids,owners


class Encoder:
    def __init__(self,device="cpu"):
        folder=str(ROOT/"model")
        self.device=torch.device(device)
        self.model=AutoModel.from_pretrained(folder,trust_remote_code=False,local_files_only=True,
                                             dtype=torch.float32,attn_implementation="sdpa").eval().to(self.device)
        self.tokenizer=Tokenizer.from_file(folder+"/tokenizer.json")
        self.saved={}
        self.model.norm.register_forward_pre_hook(self.capture_pre("e"))
        self.model.layers[-1].mlp.down_proj.register_forward_pre_hook(self.capture_pre("k"))
        self.model.layers[11].register_forward_hook(self.capture_output("m"))

    def capture_pre(self,name):
        def save_input(module, inputs): self.saved[name]=inputs[0][0].detach()
        return save_input

    def capture_output(self,name):
        def save_output(module, inputs, output): self.saved[name]=output[0].detach()
        return save_output

    @torch.inference_mode()
    def run(self,item,cap,mode):
        count=len(item["rows"])
        accum={name:np.zeros((count,dim),dtype=np.float64) for name,dim in
               [("e",768),("k",1024),("m",768),("native",768)]}
        weights=np.zeros(count,dtype=np.int64)
        sizes=[]
        for tok,owners in blocks(item,self.tokenizer,cap,mode):
            ids=torch.tensor([tok],device=self.device)
            out=self.model(input_ids=ids,attention_mask=torch.ones_like(ids))
            self.saved["native"]=out.last_hidden_state[0]
            own=np.asarray(owners)
            sizes.append(len(tok))
            for owner in np.unique(own):
                if owner<0:continue
                token_mask=own==owner
                ix=np.flatnonzero(token_mask)
                weights[owner]+=len(ix)
                lo=int(ix[0]); hi=int(ix[-1])+1
                assert hi-lo==len(ix)
                for name,matrix in accum.items():
                    matrix[owner]+=self.saved[name][lo:hi].sum(0).float().cpu().numpy()
        assert weights.min()>0
        return {**{n:(m/weights[:,None]).astype(np.float32) for n,m in accum.items()},
                "row":np.array([r["row"] for r in item["rows"]]),"ix":np.array([r["ix"] for r in item["rows"]]),
                "start":np.array([r["start"] for r in item["rows"]]),"end":np.array([r["end"] for r in item["rows"]]),
                "y":np.array([r["y"] for r in item["rows"]]),"tok":weights,"block_lengths":np.array(sizes)}


def path_for(item,cap,mode):
    return ROOT/"features"/f"{ARCH}-{mode}-{cap}"/(item["id"]+"-"+identifier(item)[:16]+".npz")


def read(item,cap,mode):
    p=path_for(item,cap,mode)
    with np.load(p,allow_pickle=False) as f: return {n:f[n] for n in f}


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--stage",choices=["quick","screen","expanded","deep","common"],default="quick")
    a.add_argument("--mode",choices=["context","alone"],default="context")
    a.add_argument("--cap",type=int,default=384)
    a.add_argument("--device",default="cpu")
    a.add_argument("--only",default="")
    z=a.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","3"))); torch.set_num_interop_threads(1)
    bank=groups(z.stage)
    if z.only:bank={n:bank[n] for n in z.only.split(",")}
    enc=None
    t=time.monotonic()
    for name,items in bank.items():
        new=all_tokens=all_rows=0
        print("start",name,len(items),flush=True)
        for i,item in enumerate(items):
            p=path_for(item,z.cap,z.mode)
            if p.exists():
                v=read(item,z.cap,z.mode)
            else:
                if enc is None:enc=Encoder(z.device)
                v=enc.run(item,z.cap,z.mode)
                p.parent.mkdir(parents=True,exist_ok=True)
                temp=p.with_suffix(".partial")
                with temp.open("wb") as f:np.savez(f,**v)
                temp.replace(p); new+=1
            all_tokens+=int(v["tok"].sum());all_rows+=len(v["row"])
            if (i+1)%8==0 or i+1==len(items):
                print(name,i+1,"/",len(items),"rows",all_rows,"tokens",all_tokens,"new",new,"wall",round(time.monotonic()-t,1),flush=True)
    grouped={s:[{"id":v["id"],"hash":identifier(v),"first":v["rows"][0]["row"],
                 "last":v["rows"][-1]["row"],"rows":len(v["rows"])} for v in items] for s,items in bank.items()}
    save(ROOT/f"views-{z.stage}-{z.cap}-{z.mode}-{z.only or 'all'}.json",
         {"standard":ARCH,"split":grouped,"model_sha256":"cade6bc9f5a0e7430a323fa2a0065a7d851b1751fa55b3df5a56212b6e084f44",
          "mode":z.mode,"block_token_cap":z.cap,"roles":LABELS,
          "pretrained_parameter":"ultrabert.layers.23.mlp.down_proj.weight",
          "offsets":"newline-joined unmodified texts of nonblank source rows, within each case; end exclusive"})


if __name__=="__main__":main()
