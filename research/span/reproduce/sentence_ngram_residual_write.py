"""Write a CPU lexical memory and a guarded sentence-role residual."""
from concurrent.futures import ProcessPoolExecutor
import csv
from itertools import groupby,product
import json

import numpy as np
from scipy.sparse import csr_matrix
from safetensors.numpy import save_file
from sklearn.feature_extraction.text import HashingVectorizer

from audit_joint_repair_routing import load_views,ordered
from embeddinggemma_routed_write import (
    EDGE,OUT,PARENT,ROOT,cm,document,edit_documents,file_hash,gold_cases,independent_validation,
    load_file,measure,paired,row_summary,sample_whole,save_json,torch,
)
from program_trial import TO_ID

STEM="sentence-ngram-residual"
BINS=2**15
SOURCE=ROOT/"references/official/train.csv"
ROUTE="low assigned probability"
ARM=("unchanged","hierarchy","unigram-bigram","with-neighbors")
VECTORIZER=HashingVectorizer(n_features=BINS,ngram_range=(1,2),norm=None,binary=True,
                            alternate_sign=False,token_pattern=r"(?u)\b\w+\b",dtype=np.float32)


def neighbor_labels(y):
    return (y,np.r_[y[1:],-1],np.r_[-1,y[:-1]])


def frequencies(excluded):
    """Accumulate [self, previous, next] word features conditioned on the target label."""
    train=np.arange(4984)
    choices=np.random.default_rng(72085).choice([i for i in train if str(i) not in excluded],
                                               size=2000,replace=False)
    sampled=set(map(str,choices))
    counts=np.zeros((3,BINS,7),np.float64)
    dst=[];ys=[];keys=[]
    def flush():
        x=VECTORIZER.transform(dst)
        y=np.asarray(ys)
        for axis in range(3):
            ids=np.flatnonzero(y[:,axis]>=0)
            labels=csr_matrix((np.ones(len(ids)),(ids,y[ids,axis])),shape=(len(y),7))
            counts[axis]+=(x.T@labels).toarray()
        return x.shape[0]
    seen=0
    with SOURCE.open(newline="",encoding="utf-8-sig") as stream:
        data=csv.DictReader(stream)
        for key,items in groupby(data,key=lambda r:r["Index"]):
            if key not in sampled:
                continue
            a=[(r["Text"],TO_ID[r["Label"]]) for r in items if r["Text"].strip()]
            if not a:
                continue
            phrases,y=zip(*a)
            dst.extend(phrases)
            ys.extend(zip(*neighbor_labels(np.asarray(y))))
            keys.append(key)
            if len(dst)>=14000:
                seen+=flush()
                dst.clear()
                ys.clear()
    if dst:
        seen+=flush()
    assert set(keys).isdisjoint(excluded) and len(keys)==len(set(keys)) and all(counts.sum(1).ravel()>0)
    pooled=(counts[:, :, :].sum(2,keepdims=True)+.5)
    pooled/=pooled.sum(1,keepdims=True)
    # Zero-count hashed bins fall back to the shared lexical distribution.
    smooth=counts+120.*pooled
    distribution=smooth/smooth.sum(1,keepdims=True)
    weight=np.clip(np.log(distribution/pooled),-3.0,3.0)
    support=counts.sum(2)
    weight*=np.minimum(1,support[:,:,None]/24)
    return {"lexical":weight.astype("f4",order="C")},keys,seen


def lexical(c,memory):
    x=VECTORIZER.transform(c.texts)
    norms=np.sqrt(np.maximum(np.diff(x.indptr),1))[:,None]
    self_score=(x@memory[0])/norms
    left=(x@memory[1])/norms
    right=(x@memory[2])/norms
    zero=np.zeros((1,7))
    return np.column_stack((self_score,np.r_[zero,left[:-1]],np.r_[right[1:],zero]))


def features(case,lex,mode):
    if mode=="hierarchy":
        return case.structure
    columns=7 if mode=="unigram-bigram" else 21
    values=lex[:,:columns]
    order=case.structure[:,:7]
    return np.column_stack((case.structure,values,values[:,:7]*order))


def prepare(docs,group,memory):
    c=load_views((docs,group))
    return [(x,lexical(x,memory)) for x in c]


def fit(cases,mode):
    select=[ordered(c,ROUTE)[:max(1,int(np.ceil(.32*len(c.current))))] for c,_ in cases]
    z=np.concatenate([features(c,l,mode)[q] for (c,l),q in zip(cases,select)])
    gold=np.concatenate([c.gold[q] for (c,_),q in zip(cases,select)])
    prior=np.concatenate([c.prior[q] for (c,_),q in zip(cases,select)])
    mean=z.mean(0)
    scale=1./np.maximum(z.std(0),.04)
    x=np.column_stack(((z-mean)*scale,np.ones(len(z)))).astype("f8")
    d=np.r_[np.ones(z.shape[1]),.025]
    prob=np.maximum(np.bincount(gold,minlength=7)/len(gold),.012)
    for rarity in (0.,.32):
        w=prob[gold]**-rarity
        gram=x.T@(w[:,None]*x)
        target=np.eye(7)[gold]-prior
        rhs=x.T@(w[:,None]*target)
        for ridge in (.08,.35):
            weight=np.linalg.solve(gram+np.diag(ridge*gram.trace()/len(gram)*d),rhs)
            yield rarity,ridge,{"center":mean,"scale":scale,"matrix":np.ascontiguousarray(weight[:-1]),
                                "bias":np.ascontiguousarray(weight[-1])}


def evidence(cases,weights,arm):
    return [
        np.clip(((features(c,l,arm)-weights["center"])*weights["scale"])@weights["matrix"]+weights["bias"],-.65,.65)
        for c,l in cases
    ]


def route_data(cases):
    return [ordered(c,ROUTE) for c,l in cases]


def predict(cases,routes,delta,rate,gain,cut):
    result=[]
    for (c,_),r,z in zip(cases,routes,delta):
        y=c.current.copy()
        q=r[:3] if rate==0. else r[:max(1,int(np.ceil(rate*len(y))))]
        total=c.prior[q]+gain*z[q]
        proposed=total.argmax(1)
        k=np.arange(len(q))
        accept=total[k,proposed]-total[k,y[q]]>=cut
        y[q[accept]]=proposed[accept]
        result.append(y)
    return result


def stats(cases,labels):
    return np.stack([cm(c.gold,y) for (c,_),y in zip(cases,labels)])


def main():
    torch.set_num_threads(1)
    train=edit_documents()
    share,source,rows=frequencies({d["id"] for d in train})
    memfile=OUT/(STEM+"-memory.safetensors")
    save_file(share,str(memfile))
    mem=load_file(str(memfile))["lexical"]
    np.testing.assert_array_equal(mem,share["lexical"])
    print("Wrote lexical memory",len(source),rows,flush=True)
    source_rows=prepare(train,"train",mem)
    weights={(mode,r,lam):w for mode in ARM[1:] for r,lam,w in fit(source_rows,mode)}
    del source_rows
    groups=sample_whole()
    dev=prepare(groups["select"],"select",mem)
    base=stats(dev,[c.current for c,_ in dev])
    value=measure(base.sum(0))
    routes=route_data(dev)
    winner=(value[0],value[1],True,("unchanged",))
    grid=[]
    best_by_arm={}
    for key,w in weights.items():
        delta=evidence(dev,w,key[0])
        for rate,gain,cut in product((0.,.05,.12,.25),(.28,.56,.90),(.045,.10,.19)):
            r=predict(dev,routes,delta,rate,gain,cut)
            data=measure(stats(dev,r).sum(0))
            choice=(*key,rate,gain,cut)
            changed=sum(int((c.current!=y).sum()) for (c,_),y in zip(dev,r))
            grid.append([*choice,*data.tolist(),changed])
            rank=(data[0],data[1],False,choice)
            if data[1]>=value[1]-.002:
                winner=max(winner,rank)
                if key[0] not in best_by_arm or rank>best_by_arm[key[0]]:
                    best_by_arm[key[0]]=rank
    choice=winner[-1]
    physical=None
    hash_head=None
    if choice[0]!="unchanged":
        file=OUT/(STEM+".safetensors")
        tensors={**weights[choice[:3]],"settings":np.asarray(choice[3:])}
        save_file(tensors,str(file))
        physical=load_file(str(file))
        for key,t in tensors.items():
            np.testing.assert_array_equal(t,physical[key])
        hash_head=file_hash(file)
    receipt=ROOT/"results"/(STEM+"-frozen.json")
    save_json(receipt,{"choice":choice,"score_grid":grid,"select_parent":value.tolist(),
              "select_candidate":[float(winner[0]),float(winner[1])],
              "best_by_arm":{k:z[-1] for k,z in best_by_arm.items()},
              "memory_training_cases":source,"memory_training_rows":rows,"memory_sha256":file_hash(memfile),
              "readout_training_cases":[d["id"] for d in train],"selection_cases":[c.key for c,_ in dev],
              "head_sha256":hash_head,"source_sha256":file_hash(SOURCE),"parent_sha256":file_hash(PARENT)})
    print("Frozen residual",choice,winner[:2],value.tolist(),flush=True)
    seen={d["id"] for g in groups.values() for d in g}
    excluded=set(json.loads((EDGE/"overlap.json").read_text())["independent_exclusions_for_text_disjoint_check"])
    groups["independent"]=[d for d in independent_validation(seen) if d["id"] not in excluded]
    archived=json.loads((ROOT/"results/sentence-joint-edit.json").read_text())["splits"]
    result={"freeze_sha256":file_hash(receipt),"splits":{}}
    for group in ("select","confirm","independent","test"):
        if group=="test":
            if physical is None:
                break
            docs=[document(k,r) for k,r in gold_cases().items()]
            with ProcessPoolExecutor(3) as pool:
                # Restore official order after striped workers.
                cache={c.key:(c,l) for batch in pool.map(prepare,[docs[i::3] for i in range(3)],
                                                          [group]*3,[mem]*3) for c,l in batch}
            cases=[cache[d["id"]] for d in docs]
        else:
            cases=dev if group=="select" else prepare(groups[group],group,mem)
        old=[c.current for c,_ in cases]
        r=old if physical is None else predict(cases,route_data(cases),evidence(cases,physical,choice[0]),*physical["settings"])
        a,b=stats(cases,r),stats(cases,old)
        report={"cases":len(cases),"baseline":row_summary(b.sum(0)),"candidate":row_summary(a.sum(0)),
                "changed":sum(int((one!=two).sum()) for one,two in zip(old,r)),"paired":paired(a,b)}
        assert report["baseline"]==archived[group]["candidate"]
        if group=="select":
            np.testing.assert_allclose(measure(a.sum(0))[:2],winner[:2],atol=1.e-12,rtol=0)
        result["splits"][group]=report
        save_json(ROOT/"results"/(STEM+".json"),result)
        print(group,report["candidate"]["macro_f1"],report["baseline"]["macro_f1"],report["changed"],flush=True)


if __name__=="__main__":
    main()
