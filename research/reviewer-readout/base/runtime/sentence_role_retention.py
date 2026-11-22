"""Select a gradient-free sentence decoder edit with per-role retention objectives."""
from dataclasses import dataclass
import json
from itertools import product

import numpy as np
from safetensors.numpy import load_file,save_file

from known_sentence_pair_edit import (
    EDGE,CRF,DONOR,ROOT,OUT,HierModel,KnownSentences,ONE,ZERO,aligned_embeddings,
    cm,document,edit_documents,embeddings,file_hash,gold_cases,independent_validation,
    load_decoder,prepare_score,row_summary,sample_whole,save_json,torch,transition_scores
)
from hier_full import CACHE

REASONING=5
DECISION=6
TARGETS=(REASONING,DECISION)
# Macro-F1 and accuracy are followed by the two protected per-role F1 scores.
NAMES=("macro_f1","accuracy","reasoning","decision")


@dataclass
class Coefficients:
    bias: np.ndarray
    gain: np.ndarray
    unary: np.ndarray

    def copy(self):
        return Coefficients(self.bias.copy(),self.gain.copy(),self.unary.copy())

    def merge(self, old, fraction):
        return Coefficients(*(b+fraction*(a-b) for a,b in zip(
            (self.bias,self.gain,self.unary),(old.bias,old.gain,old.unary))))

    def as_json(self):
        return {k:getattr(self,k).tolist() for k in ("bias","gain","unary")}


def neutral():
    return Coefficients(ZERO.copy(),ONE.copy(),np.zeros(7,np.float32))


def from_frozen(path):
    current=json.loads(path.read_text())
    return Coefficients(np.asarray(current["bias"],np.float32),np.asarray(current["gain"],np.float32),
                        np.zeros(7,np.float32))


def measure(cm):
    tp=cm.diagonal()
    count=cm.sum(0)+cm.sum(1)
    f1=np.divide(2*tp,count,out=np.zeros(7),where=count>0)
    return np.array([f1.mean(),tp.sum()/cm.sum(),f1[REASONING],f1[DECISION]])


class Bench(KnownSentences):
    def labels(self,edit):
        source=self.logits
        try:
            self.logits=np.ascontiguousarray(source+edit.unary,dtype=np.float64)
            return self.decode(edit.bias,edit.gain)
        finally:
            self.logits=source

    def stats(self,edit,per_case=False):
        y=self.labels(edit)
        if per_case:
            return np.stack([cm(self.gold[a:b],y[a:b]) for a,b in zip(self.sizes,self.sizes[1:])])
        return measure(cm(self.gold,y))


def candidate_search(bench,initial,goals,allowed):
    edit=initial.copy()
    best=None
    history=[]
    def fitness():
        x=bench.stats(edit)
        short=np.maximum(goals-x[2:],0).sum()
        penalty=.003*np.square(edit.bias).mean()+.002*np.square(edit.gain-1).mean()+.007*np.square(edit.unary).mean()
        return (x[0]+.10*x[1]-allowed*short-penalty,*x)
    best=fitness()
    history.append([float(v) for v in best])
    edges=list(product(range(7),repeat=2))
    edges.sort(key=lambda q:0 if set(q)&set(TARGETS) else 1)
    for radius in (.30,.16,.08):
        old=best
        # Start with emissions and grouped protected transitions, then refine individual pairs.
        groups=[(edit.unary,(role,),.9*radius,-.7,.7) for role in (5,6,0,1,2,3,4)]
        for target in TARGETS:
            groups.extend((edit.bias,tuple(np.asarray(index).T),radius,-.95,.95) for index in (
                [(r,target) for r in range(7) if r!=target],
                [(target,r) for r in range(7) if r!=target]))
        groups.extend((edit.bias,idx,radius,-.95,.95) for idx in edges)
        groups.extend((edit.gain,idx,1.2*radius,.25,1.85) for idx in edges)
        for weights,idx,step,lo,hi in groups:
            start=weights[idx].copy()
            selected=start
            for change in (step,-step):
                weights[idx]=np.clip(start+change,lo,hi)
                after=fitness()
                if after>best:
                    best=after
                    selected=weights[idx].copy()
            weights[idx]=selected
        history.append([float(v) for v in best])
        print("Training penalty",allowed,"step",radius,"scores",history[-1],flush=True)
        if old==best:
            break
    return edit,history


def apply(parent,e):
    q=parent[CRF].copy()
    q[3:,3:]+=e.bias
    return {**parent,CRF:q,"edge.log_ratio":(parent["edge.log_ratio"]*e.gain).astype(np.float32),
            "position_head":(parent["position_head"]+e.unary).astype(np.float32)}


def paired(a,b):
    rng=np.random.default_rng(7721)
    res=[]
    for i in rng.integers(len(a),size=(1800,len(a))):
        res.append(measure(a[i].sum(0))-measure(b[i].sum(0)))
    delta=np.quantile(res,[.025,.975],axis=0).T
    return {k:v.tolist() for k,v in zip(NAMES,delta)}


def main():
    torch.set_num_threads(2)
    original=neutral()
    latest=from_frozen(ROOT/"results/known-sentence-pair-edit-frozen.json")
    base=load_file(str(EDGE/"role-edge-balanced.safetensors"))
    engine=load_decoder()
    vocab=json.loads((EDGE/"vocab.json").read_text())
    model=HierModel(DONOR)

    def pack(docs,raw,parent):
        refs=[{**d,"units":d["reference_units"]} for d in docs]
        rows=[(prepare_score(d,x,{"exact":parent}),transition_scores(d["text"],d["units"],vocab,parent))
              for d,x in zip(refs,raw)]
        return Bench(rows,parent,engine).bind(refs)

    train=edit_documents()
    bench=pack(train,[model.logits(x) for x in embeddings(model,train,"aligned-source-train")],base)
    floors=bench.stats(original)[2:]
    options={"preceding":latest,"original":original}
    for part in (.25,.50,.75):
        options[f"interpolation-{part:g}"]=latest.merge(original,part)
    mask=np.zeros((7,7),bool)
    mask[TARGETS,:]=True
    mask[:,TARGETS]=True
    rollback=latest.copy()
    rollback.bias[mask]=0.
    rollback.gain[mask]=1.
    options["restore-protected-pairs"]=rollback
    traces={}
    for penalty in (0.,.30,.75):
        for seed,init in (("preceding",latest),("restored",rollback)) if penalty else (("preceding",latest),):
            key=f"search-{seed}-{penalty:g}"
            candidate,history=candidate_search(bench,init,floors,penalty)
            traces[key]=history
            options[key]=candidate
            options[key+"-half"]=candidate.merge(init,.5)
    settings=sample_whole()
    saved={}
    dev=pack(settings["select"],[model.logits(x) for x in aligned_embeddings(model,settings["select"],"select")],base)
    measured={name:dev.stats(edit).tolist() for name,edit in options.items()}
    select_floor=np.asarray(measured["original"])[2:]-.005
    eligible=[name for name,v in measured.items() if np.all(np.asarray(v)[2:]>=select_floor)]
    choices={"macro":max(options,key=lambda n:(measured[n][0],measured[n][1],n=="preceding")),
             "retention":max(eligible,key=lambda n:(measured[n][0],measured[n][1],n=="original"))}
    for goal,name in choices.items():
        path=OUT/f"known-sentence-role-retention-{goal}.safetensors"
        save_file(apply(base,options[name]),str(path))
        saved[goal]=file_hash(path)
        print("Validation",goal,name,measured[name],flush=True)
    freeze=ROOT/"results/sentence-role-retention-frozen.json"
    save_json(freeze,{"frozen_profiles":choices,"sha256":saved,"search_trace":traces,
        "coefficients":{k:options[n].as_json() for k,n in choices.items()},"f1_floors":select_floor.tolist(),
        "scores":["macro_f1","accuracy","reasoning","decision"],"grid":measured,
        "train_case_ids":[x["id"] for x in train],"selection_case_ids":[x["id"] for x in settings["select"]]})
    used={d["id"] for rows in settings.values() for d in rows}
    omitted=set(json.loads((EDGE/"overlap.json").read_text())["independent_exclusions_for_text_disjoint_check"])
    settings["independent"]=[d for d in independent_validation(used) if d["id"] not in omitted]
    report={"choices":choices,"frozen_sha256":file_hash(freeze),"splits":{},
        "protocol":"All fits use 144 official training judgments, and candidate selection uses 72 separate validation judgments. The retention choice must keep reasoning and decision F1 within 0.5 points of the original profile on selection. The supplied-sentence test removes blank rows from model inputs before scoring. Existing neural weights remain fixed."}
    for split in ("select","confirm","independent","test"):
        if split=="test":
            docs=[]; raws=[]
            for k,rr in gold_cases().items():
                d=document(k,rr)
                docs.append(d)
                keep=[i for i,(_,line,_) in enumerate(rr) if line.strip()]
                if len(keep)!=len(rr):
                    parts=[line for _,line,_ in rr if line.strip()]
                    raws.append(model.logits(model.contextualize(model.words(parts))))
                else:
                    with np.load(CACHE/(k+".npz")) as f:raws.append(f["context"])
        else:
            docs=settings[split]
            raws=[model.logits(x) for x in aligned_embeddings(model,docs,split)]
        original_pack=pack(docs,raws,base)
        legacy=original_pack.stats(original,True)
        released=original_pack.stats(latest,True)
        outcomes={}
        for name in choices:
            pred=pack(docs,raws,load_file(str(OUT/f"known-sentence-role-retention-{name}.safetensors"))).stats(original,True)
            outcomes[name]={"cases":len(pred),"candidate":row_summary(pred.sum(0)),
                "original":row_summary(legacy.sum(0)),"preceding":row_summary(released.sum(0)),
                "paired_to_preceding":paired(pred,released),"paired_to_original":paired(pred,legacy)}
            print(split,name,[round(x,4) for x in measure(pred.sum(0))],
                  [round(x,4) for x in measure(released.sum(0))],flush=True)
        report["splits"][split]=outcomes
        save_json(ROOT/"results/sentence-role-retention.json",report)


if __name__=="__main__":
    main()
