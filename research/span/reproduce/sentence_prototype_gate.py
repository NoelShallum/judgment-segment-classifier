"""Filter prototype-driven edits using calibration judgments outside prototype fitting."""
import csv
from concurrent.futures import ProcessPoolExecutor
from itertools import groupby, product
import json
import random

import numpy as np
from safetensors.numpy import load_file,save_file

from precompute_repair_test_states import encode
from program_trial import TO_ID
from sentence_prototype_edit import (
    DONOR,EDGE,OUT,ROOT,HierModel,KnownSentences,additional,aligned_embeddings,
    cm_case,document,file_hash,gold_cases,independent_validation,load_decoder,measure,
    packed_decode,paired,prepare_score,revised,row_summary,sample_whole,
    save_json,torch,training_documents,transition_scores,
)


def calibration_documents():
    used={int(d["id"]) for d in training_documents()}
    chosen=set(random.Random(794262).sample(sorted(set(range(4984))-used),192))
    cases=[]
    with (ROOT/"references/official/train.csv").open(newline="",encoding="utf-8-sig") as stream:
        for case,rows in groupby(csv.DictReader(stream),key=lambda r:r["Index"]):
            if int(case) in chosen:
                y=[(i,r["Text"],TO_ID[r["Label"]]) for i,r in enumerate(rows)]
                cases.append(document(case,y))
    assert len(cases)==192 and not ({int(d["id"]) for d in cases}&used)
    return cases


def prefill(cases):
    pieces=[(d["id"],[d["text"][a:b] for a,b in d["reference_units"]]) for d in cases]
    workers=[[] for _ in range(4)]
    sizes=[0]*len(workers)
    for case,text in sorted(pieces,key=lambda r:-sum(map(len,r[1]))):
        pos=min(range(len(sizes)),key=sizes.__getitem__)
        workers[pos].append((case,text))
        sizes[pos]+=sum(map(len,text))
    with ProcessPoolExecutor(len(workers)) as process:
        print("Calibration cache entries written",sum(process.map(encode,workers)),flush=True)


def options(pack,features,base,proto,mode):
    before=packed_decode(pack,base,pack.logits)
    extra=additional(features,pack.logits,proto,mode)
    after=revised(pack,base,extra,mode,float(proto["gain"]))
    i=np.arange(pack.units)
    margin=extra[i,after]-extra[i,before]
    after_support=np.zeros(pack.units,dtype=np.int8)
    old_support=np.zeros(pack.units,dtype=np.int8)
    for offset in (-1,1):
        j=i+offset
        allowed=(j>=0)&(j<len(i))
        if offset>0:
            allowed[:-1]&=~pack.start_mask[1:]
        else:
            allowed&=~pack.start_mask
        at=i[allowed]
        incoming=before[j[allowed]]
        after_support[at]+=after[at]==incoming
        old_support[at]+=before[at]==incoming
    return before,after,margin,after_support-old_support


def fit(pack,before,after,evidence,context):
    changed=before!=after
    target=pack.gold
    if not changed.any():
        raise ValueError("Prototype branch changed no labels on calibration judgments")
    cuts=np.unique(np.quantile(evidence[changed],[.25,.50,.75]).astype("f4"))
    level=np.digitize(evidence,cuts)
    density=len(cuts)+1
    pair=7*before.astype(np.int64)+after
    ctx=(context+2).astype(np.int64)
    mass=np.bincount(target,minlength=7)/len(target)
    for selector in ("margin","neighbors"):
        bucket=level if selector=="margin" else ctx
        n=density if selector=="margin" else 5
        key=n*pair+bucket
        for role_balance in (0.,.5):
            priority=mass[target]**(-role_balance)
            priority/=priority.mean()
            gain=((after==target).astype(np.float64)-(before==target))*priority
            total=np.bincount(key[changed],minlength=49*n).reshape(49,n)
            net=np.bincount(key[changed],weights=gain[changed],minlength=49*n).reshape(49,n)
            rollup=(net.sum(1)+10*gain[changed].mean())/(total.sum(1)+10)
            policy=(net+rollup[:,None]*16)/(total+16)
            yield (selector,role_balance),{"cuts":cuts,"change_utility":policy.astype(np.float32)}


def apply(options,adjustment,selector,threshold):
    before,after,evidence,context=options
    pair=7*before.astype(np.int64)+after
    level=np.digitize(evidence,adjustment["cuts"]) if selector=="margin" else context+2
    return np.where(adjustment["change_utility"][pair,level]>threshold,after,before)


def main():
    cases=calibration_documents()
    prefill(cases)
    torch.set_num_threads(2)
    frozen=ROOT/"results/sentence-prototype-frozen.json"
    source=OUT/"known-sentence-prototypes.safetensors"
    receipt=json.loads(frozen.read_text())
    assert file_hash(source)==receipt["weights_sha256"]
    both=load_file(str(source))
    prefix="prototype."
    proto={k[len(prefix):]:v for k,v in both.items() if k.startswith(prefix)}
    base={k:v for k,v in both.items() if not k.startswith(prefix)}
    mode=receipt["choice"][2]
    used=set(receipt["training_cases"])
    assert not ({d["id"] for d in cases}&used)
    encoder=HierModel(DONOR)
    vocab=json.loads((EDGE/"vocab.json").read_text())
    decoder=load_decoder()

    def prepare(docs,name):
        h=list(aligned_embeddings(encoder,docs,name))
        supplied=[{**d,"units":d["reference_units"]} for d in docs]
        bank=[(prepare_score(d,encoder.logits(x),{"exact":base}),transition_scores(d["text"],d["units"],vocab,base))
                for d,x in zip(supplied,h)]
        p=KnownSentences(bank,base,decoder).bind(supplied)
        return p,options(p,np.concatenate(h),base,proto,mode)

    data,opt=prepare(cases,"prototype-calibration")
    changed=int((opt[0]!=opt[1]).sum())
    helpful=int(((opt[1]==data.gold)&(opt[0]!=opt[1])).sum())
    harmful=int(((opt[0]==data.gold)&(opt[0]!=opt[1])).sum())
    study=dict(fit(data,*opt))
    print("Prototype calibration",len(cases),len(data.gold),changed,helpful,harmful,flush=True)
    groups=sample_whole()
    val,view=prepare(groups["select"],"select")
    old=measure(cm_case(val,view[0]).sum(0))
    candidate=measure(cm_case(val,view[1]).sum(0))
    grid=[];eligible=[];fallback=[]
    for (selector,balance),p in study.items():
        for threshold in (-.15,-.04,.0,.035,.10,.18):
            y=apply(view,p,selector,threshold)
            q=measure(cm_case(val,y).sum(0)).tolist()
            n=int((view[0]!=y).sum())
            spec=(selector,balance,threshold)
            grid.append([*spec,*q,n])
            if n:
                entry=(q[0],q[1],spec)
                fallback.append(entry)
                if q[1]>=old[1]-.004:
                    eligible.append(entry)
    best=max(eligible or fallback)
    selector,balance,thr=best[2]
    changed_weights={**study[selector,balance],"selector":np.asarray(0 if selector=="margin" else 1,dtype=np.int8),
                     "threshold":np.asarray(thr)}
    out=OUT/"known-sentence-prototype-gated.safetensors"
    save_file({**both,**{"prototype_gate."+k:v for k,v in changed_weights.items()}},str(out))
    loaded=load_file(str(out))
    w={k.removeprefix("prototype_gate."):v for k,v in loaded.items() if k.startswith("prototype_gate.")}
    assert all(np.array_equal(v,w[k]) for k,v in changed_weights.items())
    target=ROOT/"results/sentence-prototype-gate-frozen.json"
    save_json(target,{"choice":best[2],"grid":grid,"predecessor_sha256":file_hash(source),
                      "weights_sha256":file_hash(out),"prototype_freeze_sha256":file_hash(frozen),
                      "calibration_cases":[d["id"] for d in cases],"calibration_rows":len(data.gold),
                      "calibration_changes":{"altered":changed,"helpful":helpful,"harmful":harmful},
                      "selection_cases":receipt["selection_cases"],"preceding":old.tolist(),"prototype":candidate.tolist()})
    print("Gate choice",best,old.tolist(),candidate.tolist(),flush=True)
    seen={d["id"] for v in groups.values() for d in v}
    omit=set(json.loads((EDGE/"overlap.json").read_text())["independent_exclusions_for_text_disjoint_check"])
    groups["independent"]=[d for d in independent_validation(seen) if d["id"] not in omit]
    report={"choice":best[2],"freeze_sha256":file_hash(target),"splits":{},
       "protocol":"Fit a shrunk lookup table for the net correctness of prototype-proposed role changes on 192 calibration judgments excluded from the prototype fit. Compare cosine-score margin bins and surrounding incumbent labels. Select threshold on development, then replay a frozen physical checkpoint."}
    for name in ("select","confirm","independent","test"):
        if name=="test":
            groups[name]=[document(k,r) for k,r in gold_cases().items()]
        p,view=(val,view) if name=="select" else prepare(groups[name],name)
        before,after,*_=view
        y=apply(view,w,"margin" if int(w["selector"])==0 else "neighbors",float(w["threshold"]))
        a,b,c=cm_case(p,y),cm_case(p,before),cm_case(p,after)
        if name=="select":
            np.testing.assert_allclose(measure(a.sum(0))[:2],best[:2],atol=1e-12,rtol=0.)
        if name=="test":
            original=json.loads((ROOT/"results/sentence-label-change-qa.json").read_text())["scores"]
            assert row_summary(b.sum(0))==original
            stage=ROOT/"results/sentence-prototype-edit.json"
            if stage.is_file() and "test" in (direct:=json.loads(stage.read_text()))["splits"]:
                assert row_summary(c.sum(0))==direct["splits"]["test"]["candidate"]
        report["splits"][name]={"cases":len(a),"candidate":row_summary(a.sum(0)),"preceding":row_summary(b.sum(0)),
            "prototype_ungated":row_summary(c.sum(0)),"paired_to_preceding":paired(a,b),
            "paired_to_ungated":paired(a,c),"modified":int((y!=before).sum())}
        save_json(ROOT/"results/sentence-prototype-gate-edit.json",report)
        print(name,measure(a.sum(0)).tolist(),measure(b.sum(0)).tolist(),flush=True)


if __name__=="__main__":
    main()
