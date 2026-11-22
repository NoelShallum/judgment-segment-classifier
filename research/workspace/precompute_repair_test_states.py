"""Warm the frozen encoder cache without scoring or using test labels."""
from concurrent.futures import ProcessPoolExecutor
import os
import numpy as np
import torch

from hier_full import DONOR,ROOT,gold_cases
from hier_model import HierModel
from span_eval import document

def encode(chunk):
    torch.set_num_threads(1)
    model=HierModel(DONOR)
    dest=ROOT/"cache/known-sentence-pair-edit"
    written=0
    for case,pieces in chunk:
        path=dest/(case+".npy")
        if path.exists():
            continue
        states=model.contextualize(model.words(pieces))
        tmp=path.with_suffix(f".{os.getpid()}.part")
        with tmp.open("wb") as file:
            np.save(file,states,allow_pickle=False)
        tmp.replace(path)
        written+=1
    return written

if __name__=="__main__":
    docs=[]
    for case,rows in gold_cases().items():
        d=document(case,rows)
        docs.append((case,[d["text"][a:b] for a,b in d["reference_units"]]))
    batches=[[] for _ in range(5)]
    costs=[0]*len(batches)
    for case,pieces in sorted(docs,key=lambda x:-sum(map(len,x[1]))):
        i=min(range(len(batches)),key=costs.__getitem__)
        costs[i]+=sum(map(len,pieces))
        batches[i].append((case,pieces))
    with ProcessPoolExecutor(len(batches)) as pool:
        print("Cache entries written",list(pool.map(encode,batches)),flush=True)
