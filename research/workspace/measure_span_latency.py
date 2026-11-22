"""Sample CPU latency for the packaged whole-judgment inference profiles."""
from collections import defaultdict
import csv
import platform
import time

import numpy as np
from safetensors.numpy import load_file
import torch

from hier_model import HierModel
from hier_trial import DONOR,OUT
from predict_discourse_spans import infer
from program_trial import TO_ID,save_json
from seven_paths import ROOT
from span_eval import document

def main():
    torch.set_num_threads(2)
    path=ROOT/"references/official/val.csv"
    sizes=defaultdict(int)
    with path.open(newline="",encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):sizes[r["Index"]]+=len(r["Text"].split())
    cases=[min(sizes,key=lambda k:(abs(sizes[k]-n),k)) for n in (1800,6500,16000)]
    rows=defaultdict(list)
    with path.open(newline="",encoding="utf-8-sig") as f:
        for i,r in enumerate(csv.DictReader(f)):
            k=r["Index"]
            if k in cases:rows[k].append((i,r["Text"],TO_ID[r["Label"]]))
    model=HierModel(DONOR);patch=load_file(str(OUT/"whole-discourse-balanced.safetensors"))
    infer(model,patch,"The case was listed for hearing. The appeal is dismissed.")
    results=[]
    for k in cases:
        d=document(k,rows[k]);mins=[];count=None
        for _ in range(3):
            t=time.monotonic();p=infer(model,patch,d["text"]);ends=time.monotonic()-t
            mins.append(ends);c=len(p["spans"])
            if count is not None:assert count==c
            count=c
        results.append({"whitespace_tokens":sizes[k],"model_units":len(d["units"]),"predicted_spans":count,
                        "median_warm_s":float(np.median(mins)),"warm_s":mins})
        print(sizes[k],round(float(np.median(mins)),3),flush=True)
    save_json(ROOT/"results/whole-span-cpu-latency.json",{"machine":"Intel Core i5-8250U CPU","platform":platform.system(),
       "torch_threads":2,"configuration":"Profile already loaded. Includes rule splitting, semantic inference, and offset output. Two other research processes were active. Three representative validation samples near stated sizes; timings are illustrative.",
       "results":results})
if __name__=="__main__":
    main()
