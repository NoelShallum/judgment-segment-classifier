"""Cache raw-boundary hierarchy states using only each judgment's text."""
from concurrent.futures import ProcessPoolExecutor
import os

import numpy as np

from hier_full import CACHE,DONOR,ROOT,gold_cases,torch
from hier_model import HierModel
from span_eval import document

DEST=ROOT/"cache/whole-joint-states"


def encode(cases):
    torch.set_num_threads(1)
    model=HierModel(DONOR)
    seen=0
    for key,texts in cases:
        p=DEST/(key+".npy")
        if p.is_file():
            continue
        h=model.contextualize(model.words(texts))
        with np.load(CACHE/(key+".npz")) as source:
            np.testing.assert_allclose(model.logits(h),source["raw"],atol=1.e-5,rtol=1.e-5)
        temp=p.with_suffix(f".{os.getpid()}.part")
        with temp.open("wb") as s:
            np.save(s,h,allow_pickle=False)
        temp.replace(p)
        seen+=1
    return seen


def main():
    DEST.mkdir(exist_ok=True)
    bundle=[]
    for k,r in gold_cases().items():
        d=document(k,r)
        pieces=[d["text"][a:b].strip() for a,b in d["units"]]
        bundle.append((k,pieces))
    groups=[[] for _ in range(5)]
    load=[0]*len(groups)
    for k,pieces in sorted(bundle,key=lambda row:-sum(map(len,row[1]))):
        i=min(range(len(load)),key=load.__getitem__)
        groups[i].append((k,pieces))
        load[i]+=sum(map(len,pieces))
    with ProcessPoolExecutor(len(groups)) as pool:
        print("Raw judgment states cached",list(pool.map(encode,groups)),flush=True)
    assert len(list(DEST.glob("*.npy")))==len(bundle)


if __name__=="__main__":
    main()
