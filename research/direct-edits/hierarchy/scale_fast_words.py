"""Benchmark an equivalent dense bidirectional evaluation on the identical donor weights."""
import argparse
import json
import os
import sqlite3
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from native_memory import HERE,DONOR,HierModel,dump,file_hash
from scale_program import DB,FOLDER,compiler

class DenseWords:
    def __init__(self,model):
        self.model=model;self.recurrents=[]
        for suffix in ("","_reverse"):
            cell=nn.LSTM(model.word.shape[1],model.sent.hidden_size,batch_first=True)
            cell.load_state_dict({name:getattr(model.sent,name+suffix) for name in cell.state_dict()})
            self.recurrents.append(cell.eval())

    @torch.inference_mode()
    def __call__(self,texts):
        toks=[torch.tensor(self.model.tokenize(s),dtype=torch.int64) for s in texts]
        lengths=[len(t) for t in toks];order=sorted(range(len(toks)),key=lengths.__getitem__)
        ret=torch.empty(len(toks),200)
        for i in range(0,len(toks),64):
            ix=order[i:i+64];seq=[toks[j] for j in ix];ends=torch.tensor([len(t)-1 for t in seq],dtype=torch.long)
            ar=torch.arange(len(ix));both=[]
            for j,cell in enumerate(self.recurrents):
                words=seq if j==0 else [v.flip((0,)) for v in seq]
                x=self.model.word[pad_sequence(words,batch_first=True)]
                h,_=cell(x);both.append(h[ar,ends])
            ret[ix]=torch.cat(both,dim=1)
        return ret

def bench():
    plan=json.loads((FOLDER/"plan.json").read_text());db=sqlite3.connect(f"file:{DB}?mode=ro",uri=True)
    model=HierModel(DONOR);encoder=DenseWords(model);viterbi=compiler();texts=[];ends=[0];keys=[]
    for k in plan["fit"][:24]:
        t=[r[0] for r in db.execute("SELECT text FROM sentence WHERE doc=? ORDER BY seq",(k,))]
        texts+=t;keys.append(k);ends.append(len(texts))
        if len(texts)>2600:break
    times={}
    t=time.monotonic();old=model.words(texts);times["normal"]=time.monotonic()-t
    t=time.monotonic();new=encoder(texts);times["dense"]=time.monotonic()-t
    e=(new-old).numpy();pred={}
    for k,x in (("normal",old),("dense",new)):
        h=np.concatenate([model.contextualize(x[a:b]) for a,b in zip(ends,ends[1:])])
        pred[k]=viterbi(model.logits(h),ends,model.transitions)
    out=dict(rows=len(texts),cases=keys,seconds=times,maximum_absolute_activation_error=float(np.abs(e).max()),mean_absolute_activation_error=float(np.abs(e).mean()),different_decoded_labels=int(np.sum(pred["normal"]!=pred["dense"])),speed_ratio=times["normal"]/times["dense"])
    dump(FOLDER/"dense-words-benchmark.json",out);print(out,flush=True)
if __name__=="__main__":bench()
