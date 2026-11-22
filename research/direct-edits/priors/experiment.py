"""Write a document-conditioned native hierarchical emission bias from corpus statistics."""
import argparse
import csv
import json
import sys
from itertools import groupby
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file
from scipy.linalg import cho_factor, cho_solve
from prior_model import features, targets, adapted_bias

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(ROOT))
from weight_edit import LABELS, file_hash
from hier_model import HierModel, path_labels

NAMES=("None","Facts","Issue","Arguments of Petitioner","Arguments of Respondent","Reasoning","Decision")
IDS={name:i for i,name in enumerate(NAMES)}
DONOR=ROOT/"references/hier"
MANIFEST=HERE.parent/"shared-confirmation.json"
def write(path,payload):
    tmp=path.with_suffix(".part")
    tmp.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
    tmp.replace(path)


def cases(split,limit_ids=None):
    with (ROOT/"references/official"/(split+".csv")).open(encoding="utf-8-sig",newline="") as f:
        seen=set()
        for key,items in groupby(csv.DictReader(f),lambda r:r["Index"]):
            if limit_ids is not None and key not in limit_ids:
                continue
            assert key not in seen
            seen.add(key)
            rows=[(r["Text"].strip(),IDS[r["Label"]]) for r in items if r["Text"].strip()]
            if rows:
                texts,gold=zip(*rows)
                yield key,list(texts),np.asarray(gold)


def fit():
    ids,x,y=[],[],[]
    for k,sentences,g in cases("train"):
        ids.append(k)
        x.append(features(sentences))
        y.append(np.bincount(g,minlength=7))
    assert len(ids)==4984 and sum(v.sum() for v in y)==1122376
    x=np.stack(x).astype(np.float64)
    y=np.stack(y)
    center=x.mean(0)
    scale=1/np.maximum(x.std(0),.075)
    z=np.clip((x-center)*scale,-5,5)
    global_fraction=y.sum(0)/y.sum()
    logp=np.log((y+14*global_fraction)/(y.sum(1,keepdims=True)+14))
    logp-=logp.mean(1,keepdims=True)
    avg=logp.mean(0)
    gram=z.T@z
    ridge=.45*np.mean(gram.diagonal())
    gram.flat[::len(gram)+1]+=ridge
    solve=cho_factor(gram)
    parameters={"center":center,"scale":scale,"offset":avg,"global_fraction":global_fraction,
                "role_prior.weight":cho_solve(solve,z.T@(logp-avg))}
    rng=np.random.default_rng(120116)
    parameters["shuffled.weight"]=cho_solve(solve,z.T@(logp[rng.permutation(len(logp))]-avg))
    save_file({k:np.ascontiguousarray(v) for k,v in parameters.items()},str(HERE/"priors.safetensors"))
    write(HERE/"fit.json",{"train_csv_sha256":file_hash(ROOT/"references/official/train.csv"),
        "model_sha256":file_hash(HERE/"priors.safetensors"),"train_judgments":len(ids),"train_rows":int(y.sum()),
        "native_donor_sha256":file_hash(DONOR/"hier.safetensors"),"global_fraction":global_fraction.tolist(),
        "statistic":"One case = one training example. Closed-form ridge from the three-part bag of words and formatting to the centered seven-role log proportion.",
        "native_target":"emitter.hidden2tag.bias","gradient_steps":0})
    print("fit",len(y),x.shape,flush=True)


def cm(g,p):
    return np.bincount(7*g+p,minlength=49).reshape(7,7)


def stat(arr):
    z=np.sum(arr,axis=0)
    f1=2*z.diagonal()/np.maximum(z.sum(0)+z.sum(1),1)
    return dict(macro_f1=float(f1.mean()),accuracy=float(np.trace(z)/z.sum()),per_role_f1=dict(zip(LABELS,f1.tolist())))


def make_cache(name,ids,weights,dataset="val"):
    import torch
    torch.set_num_threads(1)
    donor=HierModel(DONOR)
    where=HERE/"cases"
    where.mkdir(exist_ok=True)
    bank={}
    fit=file_hash(HERE/"priors.safetensors")
    for key,text,g in cases(dataset,set(ids)):
        target=where/(key+".npz")
        if not target.exists():
            cached=ROOT/"cache/known-sentence-pair-edit"/(key+".npy")
            if cached.is_file():
                h=np.load(cached,allow_pickle=False)
                assert h.shape==(len(text),donor.weight.shape[1])
            else:
                h=donor.contextualize(donor.words(text))
            z=donor.logits(h)
            native=path_labels(z,donor.transitions)
            p=targets(text,weights)
            with target.with_suffix(".part").open("wb") as f:
                np.savez(f,signature=fit,g=g,emissions=z,prediction=native,**p)
            target.with_suffix(".part").replace(target)
        with np.load(target,allow_pickle=False) as f:
            assert f["signature"]==fit
            np.testing.assert_array_equal(f["g"],g)
            bank[key]={k:f[k] for k in f.files if k!="signature"}
    assert set(bank)==set(ids)
    return [bank[i] for i in ids],donor


def scores(data,donor,strength,arm):
    cms,preds,edits=[],[],[]
    for r in data:
        bias=adapted_bias(r,arm,strength,donor)
        pred=path_labels(r["emissions"]+(bias[3:]-donor.bias[3:]),donor.transitions)
        if strength==0: np.testing.assert_array_equal(pred,r["prediction"])
        cms.append(cm(r["g"],pred))
        preds.append(pred)
        edits.append(bias)
    return np.stack(cms),np.concatenate(preds),np.stack(edits)


def run(mode):
    paper=ROOT/"results/sentence-bert-referral-frozen.json"
    existing=json.loads(paper.read_text())
    shared=json.loads(MANIFEST.read_text())
    if mode=="select":
        ids=[str(i) for i in existing["selection_cases"] if str(i) not in shared["extended_validation_judgment_ids"]]
    else:
        ids=shared["small_validation_judgment_ids"] if mode=="shared" else shared["extended_validation_judgment_ids"]
    weights=load_file(str(HERE/"priors.safetensors"))
    data,donor=make_cache(mode,ids,weights)
    if mode=="select":
        trials=[]
        for g in (0.,.125,.25,.5,1.,2.):
            c,_,_=scores(data,donor,g,"targeted")
            score=stat(c)
            trials.append([g,score["macro_f1"],score["accuracy"]])
        gain=max(trials[1:],key=lambda row:(row[1],row[2],-row[0]))[0]
        write(HERE/"freeze.json",{"fit_receipt_sha256":file_hash(HERE/"fit.json"),"ids":ids,"grid":trials,
                                  "gain":gain,"decision":"Score the highest ranked nonzero edit plus identity and two control gates on the shared cases."})
        print("select",trials,flush=True)
        return
    freeze=json.loads((HERE/"freeze.json").read_text())
    assert not set(freeze["ids"])&set(ids)
    out={"cases":len(ids),"rows":sum(len(r["g"]) for r in data),"ids":ids,"arms":{},
         "freeze_sha256":file_hash(HERE/"freeze.json"),"manifest_sha256":file_hash(MANIFEST)}
    g=np.concatenate([r["g"] for r in data])
    old,orig,_=scores(data,donor,0,"targeted")
    out["arms"]["donor"]=stat(old)
    sys.path.insert(0,str(HERE.parent/"pooler"))
    from experiment import interval
    for arm in ("targeted","shuffled","constant"):
        counts,p,edits=scores(data,donor,freeze["gain"],arm)
        changed=p!=orig
        out["arms"][arm]={**stat(counts),"delta_ci95":interval(old,counts),
            "changed":int(changed.sum()),"corrected":int(np.sum(changed&(p==g))),"broken":int(np.sum(changed&(orig==g))),
            "median_l2_bias_delta":float(np.median(np.linalg.norm(edits-donor.bias[None,:],axis=1)))}
        if arm=="targeted":
            # Write and read back the complete native bias for every observed case.
            target=HERE/(mode+"-case-biases.safetensors")
            save_file({k+".emitter.hidden2tag.bias":v for k,v in zip(ids,edits)},str(target))
            src=load_file(str(target))
            for k,v in zip(ids,edits): np.testing.assert_array_equal(src[k+".emitter.hidden2tag.bias"],v)
            # Whole-document re-inference on the first case with the new FP32 tensor.
            # This exercises the original model.logits path directly, prior to CRF.
            k,text,_=next(cases("val",{ids[0]}))
            donor.bias=src[k+".emitter.hidden2tag.bias"].copy()
            got=path_labels(donor.logits(donor.contextualize(donor.words(text))),donor.transitions)
            np.testing.assert_array_equal(got,p[:len(text)])
            donor=HierModel(DONOR)
            out["native_parity"]={"case":k,"sentences":len(got),"disagreements":0,"case_tensor_sha256":file_hash(target),
                                  "tensor_shape":list(donor.bias.shape),"modified_role_entries":7}
    write(HERE/(mode+".json"),out)
    print(mode,out["rows"],{k:v["macro_f1"] for k,v in out["arms"].items()},flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("mode",choices=("fit","select","shared","extended"))
    mode=cli.parse_args().mode
    fit() if mode=="fit" else run(mode)
