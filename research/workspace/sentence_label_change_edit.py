"""Selectively keep calibrated sentence decisions with a train-written table."""
import json
from dataclasses import dataclass
from itertools import product

import numpy as np
from safetensors.numpy import load_file,save_file

from sentence_emission_calibration import (
    DONOR,EDGE,ONE,OUT,ROOT,ZERO,HierModel,KnownSentences,aligned_embeddings,
    document,edit_documents,embeddings,file_hash,gold_cases,independent_validation,
    load_decoder,prepare_score,row_summary,sample_whole,save_json,torch,transition_scores
)
from sentence_role_retention import measure,paired
from hier_full import CACHE,cm

CUTS=np.asarray([-.7,-.2,.2,.7])
NBINS=len(CUTS)+1
CANDIDATES=[("unchanged",0.),*(("pair-"+family,t) for family in ("accuracy","rarity")
       for t in (0.,.02,.07,.16)),*(("margin-"+family,t) for family in ("accuracy","rarity") for t in (0.,.02,.07,.16))]


@dataclass
class Predictions:
    bank: KnownSentences
    before: np.ndarray
    after: np.ndarray
    sides: np.ndarray

    @classmethod
    def run(cls,bank,w,b):
        old=bank.decode(ZERO,ONE).astype(np.int64)
        emissions=bank.logits
        changed=np.ascontiguousarray(emissions@w+b,dtype=np.float64)
        try:
            bank.logits=changed
            new=bank.decode(ZERO,ONE).astype(np.int64)
        finally:
            bank.logits=emissions
        i=np.arange(len(old))
        bins=np.digitize(changed[i,new]-changed[i,old],CUTS)
        return cls(bank,old,new,bins)


def fit(p,weighted):
    old,new=p.before,p.after
    valid=old!=new
    gold=p.bank.gold
    favors=(gold==new).astype(np.float64)-(gold==old)
    if weighted:
        frequencies=np.bincount(gold,minlength=7)/len(gold)
        favors*=float(np.sqrt(frequencies).mean())/np.sqrt(frequencies[gold])
    keys=NBINS*(7*old+new)+p.sides
    totals=np.bincount(keys[valid],minlength=49*NBINS).reshape(7,7,NBINS)
    sumv=np.bincount(keys[valid],weights=favors[valid],minlength=49*NBINS).reshape(7,7,NBINS)
    base=sumv.sum(2)/(totals.sum(2)+12.)
    detailed=(sumv+20*base[:,:,None])/(totals+20.)
    return base,detailed,totals


def keep(p,weights,profile,per_case=False):
    variant,floor=profile
    if variant=="unchanged":
        picks=p.before
    else:
        t=weights[variant]
        prob=t[p.before,p.after] if t.ndim==2 else t[p.before,p.after,p.sides]
        picks=np.where(prob>floor,p.after,p.before)
    if per_case:
        return np.stack([cm(p.bank.gold[a:z],picks[a:z]) for a,z in zip(p.bank.sizes,p.bank.sizes[1:])])
    return measure(cm(p.bank.gold,picks))


def main():
    torch.set_num_threads(2)
    parent_file=OUT/"known-sentence-role-pair.safetensors"
    edit_file=OUT/"known-sentence-emission-calibration-retention.safetensors"
    parent=load_file(str(parent_file))
    patch=load_file(str(edit_file))
    w,b=patch["emission.weight"],patch["emission.bias"]
    m=HierModel(DONOR)
    engine=load_decoder()
    lex=json.loads((EDGE/"vocab.json").read_text())

    def pack(docs,raw):
        copies=[{**d,"units":d["reference_units"]} for d in docs]
        packed=[(prepare_score(d,x,{"exact":parent}),transition_scores(d["text"],d["units"],lex,parent)) for d,x in zip(copies,raw)]
        return Predictions.run(KnownSentences(packed,parent,engine).bind(copies),w,b)

    train=edit_documents()
    source=pack(train,[m.logits(x) for x in embeddings(m,train,"aligned-source-train")])
    weights={}
    counts=None
    for name,weighted in (("accuracy",False),("rarity",True)):
        simple,granular,counts=fit(source,weighted)
        weights["pair-"+name]=simple
        weights["margin-"+name]=granular
    groups=sample_whole()
    def evaluate(split):
        if split!="test":
            docs=groups[split]
            return pack(docs,[m.logits(x) for x in aligned_embeddings(m,docs,split)])
        docs=[]; raw=[]
        for k,rr in gold_cases().items():
            docs.append(document(k,rr))
            sentences=[line for _,line,_ in rr if line.strip()]
            if len(sentences)==len(rr):
                with np.load(CACHE/(k+".npz")) as fp:raw.append(fp["context"])
            else:
                raw.append(m.logits(m.contextualize(m.words(sentences))))
        return pack(docs,raw)
    selection=evaluate("select")
    chart={k:keep(selection,weights,k).tolist() for k in CANDIDATES}
    original=("unchanged",0.)
    base=chart[original]
    valid=[k for k,s in chart.items() if s[1]>=base[1]-.002]
    chosen=max(valid,key=lambda k:(chart[k][0],chart[k][1],k==original))
    output=OUT/"known-sentence-label-change.safetensors"
    if chosen!=original:
        save_file({**patch,"label_change.utility":weights[chosen[0]].astype("f4"),
                   "label_change.threshold":np.asarray(chosen[1],np.float32),
                   "label_change.margin_bins":CUTS.astype("f4")},str(output))
    frozen=ROOT/"results/sentence-label-change-frozen.json"
    save_json(frozen,{"choice":chosen,"options":[[*k,*v] for k,v in chart.items()],
        "fit_cases":[d["id"] for d in train],"selection_cases":[d["id"] for d in groups["select"]],
        "train_change_counts":counts.tolist(),"weight_sha256":file_hash(output) if chosen!=original else None,
        "parent_sha256":file_hash(parent_file),"calibration_sha256":file_hash(edit_file)})
    if chosen!=original:
        actual=load_file(str(output))["label_change.utility"]
        weights[chosen[0]]=actual
    print("Selected change rule",chosen,chart[chosen],"preceding",base,flush=True)
    used={d["id"] for v in groups.values() for d in v}
    excluded=set(json.loads((EDGE/"overlap.json").read_text())["independent_exclusions_for_text_disjoint_check"])
    if chosen!=original:
        groups["independent"]=[d for d in independent_validation(used) if d["id"] not in excluded]
    report={"choice":chosen,"freeze_sha256":file_hash(frozen),"splits":{},
        "protocol":"Estimate the real value of retaining each calibrated old-to-new label change from official training sentence labels. Compare expected accuracy and inverse-frequency utilities, with optional calibrated score-margin bins shrunk to a pair mean. The selection floor allows 0.2 points of accuracy loss. No gradients."}
    for s in ("select","confirm","independent","test") if chosen!=original else ("select",):
        pred=selection if s=="select" else evaluate(s)
        a=keep(pred,weights,chosen,True)
        z=keep(pred,weights,original,True)
        report["splits"][s]={"candidate":row_summary(a.sum(0)),"preceding":row_summary(z.sum(0)),
                             "cases":len(z),"paired_to_preceding":paired(a,z)}
        save_json(ROOT/"results/sentence-label-change.json",report)
        print(s,measure(a.sum(0)).tolist(),measure(z.sum(0)).tolist(),flush=True)


if __name__=="__main__":
    main()
