"""Reuse byte-verified existing FP32 BERT hidden features for the exact same token sequence."""
import argparse
import hashlib
import json
import sys
import types
import numpy as np
from embedding_write import HERE,ROOT
import sentence_wide_bert_benchmark as wide
import sentence_referral_actual_replay as replay
import sentence_referral_utility_write as utility
import sentence_bert_referral_write as B

def main(split):
    worker=types.ModuleType("openvino_worker")
    def offline(*args,**kwargs):raise RuntimeError("The exact offline features are incomplete")
    worker.embed_ids=worker.get_engine=offline;sys.modules["openvino_worker"]=worker
    bank,p,risks,sig=replay.risk(split)
    union=np.union1d(*(utility.route(bank[1],rank,wide.RATE) for rank in risks.values()))
    rows,tok=B.load_views(split);new=[]
    for r,a,b in zip(rows,bank[1],bank[1][1:]):
        new.append((*r[:3],union[(union>=a)&(union<b)]-a,*r[4:]))
    vec,meta=wide.encode(new,split,"GPU",sig)
    legacy=json.loads((ROOT/"results/sentence-wide-bert-benchmark.json").read_text())["splits"][split]["embedding_provenance"]
    assert meta==legacy
    ids=B.pack_ids(tok,new,B.MODES[1]);assert len(ids)==len(vec)
    folder=HERE/"exact-windows-gpu-cached";folder.mkdir(exist_ok=True)
    for idx,value in zip(ids,vec):
        p=folder/(hashlib.sha256(np.asarray(idx,dtype="<i8").tobytes()).hexdigest()+".npy")
        if not p.exists():
            tmp=p.with_suffix(".part")
            with tmp.open("wb") as fp:np.save(fp,value,allow_pickle=False)
            tmp.replace(p)
    print("exact cached released FP32 windows",split,len(ids),meta["device"],flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("split",choices=("select","confirm"));main(ap.parse_args().split)
