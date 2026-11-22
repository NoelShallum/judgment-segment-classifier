"""Program a seven-role readout for Granite from fixed official training-span keys."""
from collections import defaultdict
import csv
import hashlib
import json
import numpy as np
from scipy.linalg import solve
import torch

import encoder_span_edit as study
from program_trial import official_sample,save_json
from seven_paths import ROOT

KEYS_PER_ROLE=128

def source_keys():
    rows,_,_=official_sample("train",KEYS_PER_ROLE)
    wanted={r[0] for r in rows}
    docset={r[1] for r in rows}
    kept={}
    texts=defaultdict(list)
    with (ROOT/"references/official/train.csv").open(newline="",encoding="utf-8-sig") as f:
        for i,r in enumerate(csv.DictReader(f)):
            key=r["Index"]
            if key not in docset:continue
            if i in wanted:kept[i]=(key,len(texts[key]))
            texts[key].append(r["Text"])
    examples={}
    target={}
    for ix,case,text,label in rows:
        key,center=kept[ix]
        assert key==case and text==texts[key][center]
        a=max(0,center-3);b=min(len(texts[key]),center+4)
        examples[str(ix)]=texts[key][a:b]
        target[str(ix)]=(center-a,label)
    return examples,target

def prototypes(x,y):
    labels=[x[y==i] for i in range(7)]
    centers=np.stack([a.mean(0) for a in labels])
    yield "gold prototype",study.normalized(centers),np.zeros(7)
    centered=[a-centers[i] for i,a in enumerate(labels)]
    p=sum(a.T@a/len(a) for a in centered)/7
    scale=p.trace()/len(p)
    for damper,prior in ((.1,0.),(.1,.25),(.5,0.),(.5,.25),(1.,.25)):
        w=solve(p+damper*scale*np.eye(len(p)),centers.T,assume_a="pos").T
        b=-.5*(w*centers).sum(1)+prior*np.log(np.array([603059,169653,12791,64987,50097,202346,19574])/1122507)
        # Match the scale of the original cosine-based head's score.
        m=np.mean(np.linalg.norm(w,axis=1))
        yield f"source LDA {damper} prior {prior}",w/m,b/m

def main():
    torch.set_num_threads(1)
    src="granite"
    runner=study.Encoder(src,"openvino-source")
    study.CACHE=study.CACHE/(src+"-gpu")
    task_groups,bank=study.make_documents()
    keys,targets=source_keys()
    projection={};encoded={}
    for mode in study.MODES:
        content=runner.cache(keys,mode)
        xx=np.stack([content[k][targets[k][0]] for k in keys])
        y=np.asarray([targets[k][1] for k in keys])
        projection[mode]=list(prototypes(xx,y))
        encoded[mode]=runner.cache(bank,mode)
    result=study.evaluate(src+"-source-keys",task_groups,encoded,projection)
    save_json(ROOT/"results/encoder-source-keys-granite.json",{"keys_per_role":KEYS_PER_ROLE,"keys":len(keys),
      "parameters":runner.params,"method":"The base Granite model is frozen. Source keys are 128 annotated training rows per role. For a contextual key, include nearby source rows without their labels. Write the class prototypes and regularized linear discriminants as an explicit output tensor. Inference query units come from raw input. The use of gold training labels is a closed-form supervised readout fit, with zero gradient steps. Validation chooses candidates.",
      "tasks":result})
    for k,v in result.items():
        x=v["best"]["scores"]["test"]
        print(k,v["best"]["edit"],v["best"]["mode"],x["candidate"]["exact_labeled_spans"]["f1"],x["hierarchy"]["exact_labeled_spans"]["f1"],flush=True)
if __name__=="__main__":
    main()
