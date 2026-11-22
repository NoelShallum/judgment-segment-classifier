"""Write a frozen sentence-graph readout using long-range edges within each judgment."""
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import json

import numpy as np
from safetensors.numpy import save_file

from embeddinggemma_routed_write import (
    DONOR,EDGE,OUT,PARENT,ROOT,HierModel,aligned_embeddings,case_inputs,cm,document,gold_cases,
    independent_validation,load_file,measure,paired,row_summary,sample_whole,save_json,torch,
)
from sentence_hidden_edit import training_documents
from audit_joint_repair_routing import ordered
from sentence_joint_runtime import neighbors
from sentence_graph_runtime import graph_context,remote_indices,scaled_mean
from weight_edit import file_hash

STEM="sentence-document-graph"
ARMS=("local","semantic","two-hop")
def graph_features(case):
    structure=case.structure
    q=structure[:,:7]
    left,right=neighbors(q,np.arange(len(q))==0)
    local=np.column_stack((structure,left,right,q*left,q*right))
    return local,*graph_context(structure,case.current,local)


def prepare(args):
    docs,group=args
    torch.set_num_threads(1)
    model=HierModel(DONOR)
    parent=load_file(str(PARENT))
    vocab=json.loads((EDGE/"vocab.json").read_text())
    out=[]
    for d,h in zip(docs,aligned_embeddings(model,docs,group)):
        c=case_inputs(d,h,model,parent,vocab)
        out.append((c,graph_features(c)))
    return out


def fits(cases,arm):
    features=np.concatenate([h[arm] for _,h in cases]).astype(np.float64)
    mean=features.mean(0)
    scale=1/np.maximum(features.std(0),.035)
    x=np.column_stack(((features-mean)*scale,np.ones(len(features))))
    labels=np.concatenate([c.gold for c,_ in cases])
    old=np.concatenate([c.prior for c,_ in cases])
    y=np.eye(7)[labels]
    freq=np.maximum(y.mean(0),.012)
    d=np.r_[np.ones(x.shape[1]-1),.05]
    for rarity in (0.,.28):
        w=freq[labels]**-rarity
        gram=x.T@(w[:,None]*x)
        hit=x.T@(w[:,None]*y)
        prev=x.T@(w[:,None]*old)
        for lam in (.08,.32):
            target=np.column_stack((hit,hit-prev))
            coefs=np.linalg.solve(gram+np.diag(lam*gram.trace()/len(gram)*d),target)
            yield rarity,lam,{"center":mean,"scale":scale,"matrix":np.ascontiguousarray(coefs[:-1]),
                              "bias":np.ascontiguousarray(coefs[-1])}


def responses(cases,arm,w):
    return [((parts[arm]-w["center"])*w["scale"])@w["matrix"]+w["bias"] for _,parts in cases]


def route_cases(cases,budget):
    return [ordered(c,"low assigned probability")[:max(1,int(np.ceil(budget*len(c.gold))))] if budget<1
            else np.arange(len(c.gold)) for c,_ in cases]


def score(cases,raw,routes,kind,gain,guard,threshold):
    prediction=[]
    for (c,l),z,q in zip(cases,raw,routes):
        p=c.current.copy()
        if kind=="residual":
            corrected=c.prior[q]+gain*np.clip(z[q,7:],-.65,.65)
        else:
            evidence=np.maximum(z[q,:7],0.)
            evidence/=np.maximum(evidence.sum(1,keepdims=True),1.e-8)
            corrected=(1-gain)*c.prior[q]+gain*evidence
        dest=corrected.argmax(1)
        i=np.arange(len(q))
        accept=corrected[i,dest]-corrected[i,p[q]]>=threshold
        if guard:
            accept &= p[q]!=5
        p[q[accept]]=dest[accept]
        prediction.append(p)
    return prediction


def confusion(cases,ys):
    return np.stack([cm(c.gold,y) for (c,_),y in zip(cases,ys)])


def test_chunk(args):
    docs,choice=args
    cases=prepare((docs,"test"))
    old=[c.current for c,_ in cases]
    w=load_file(str(OUT/(STEM+".safetensors")))
    arm,_,_,kind,budget,gain,guard,threshold=choice
    actual=score(cases,responses(cases,ARMS.index(arm),w),route_cases(cases,budget),kind,gain,guard,threshold)
    return [(c.key,x,y,int((a!=b).sum())) for (c,h),x,y,a,b in
            zip(cases,confusion(cases,actual),confusion(cases,old),actual,old)]


def main():
    torch.set_num_threads(1)
    root=ROOT/"results"
    train=training_documents()
    print("Preparing graph features",len(train),flush=True)
    with ProcessPoolExecutor(3) as pool:
        held=list(pool.map(prepare,[(train[i::3],"graph-fit") for i in range(3)]))
    cached={c.key:(c,h) for group in held for c,h in group}
    train_cases=[cached[d["id"]] for d in train]
    learned={(a,b,l):w for a in ARMS for b,l,w in fits(train_cases,ARMS.index(a))}
    print("Graph readouts fitted",sum(len(c.gold) for c,_ in train_cases),flush=True)
    del cached,train_cases,held
    cohorts=sample_whole()
    dev=prepare((cohorts["select"],"select"))
    original=measure(confusion(dev,[c.current for c,_ in dev]).sum(0))
    routes={v:route_cases(dev,v) for v in (.15,.35,1.)}
    best=(original[0],original[1],True,("unchanged",))
    other={}
    grid=[]
    for (arm,rarity,lam),w in learned.items():
        raw=responses(dev,ARMS.index(arm),w)
        for kind,budget,gain,guard,cut in product(("residual","posterior"),routes,(.35,.60,.9),(False,True),(.03,.08,.16)):
            y=score(dev,raw,routes[budget],kind,gain,guard,cut)
            metrics=measure(confusion(dev,y).sum(0)).tolist()
            opt=(arm,rarity,lam,kind,budget,gain,guard,cut)
            grid.append((opt,metrics))
            fit=(metrics[0],metrics[1],False,opt)
            if metrics[1]>=original[1]-.002 and metrics[2]>=original[2]-.007:
                best=max(best,fit)
                other[arm]=max(other.get(arm,fit),fit)
    choice=best[-1]
    fixed=None
    weights_sha=None
    if choice[0]!="unchanged":
        patch=OUT/(STEM+".safetensors")
        weights=learned[choice[:3]]
        save_file(weights,str(patch))
        fixed=load_file(str(patch))
        for k,v in weights.items():
            np.testing.assert_array_equal(v,fixed[k])
        weights_sha=file_hash(patch)
    freeze=root/(STEM+"-frozen.json")
    save_json(freeze,{"choice":choice,"choices_by_arm":{k:v[-1] for k,v in other.items()},"grid":grid,
                      "train_cases":[d["id"] for d in train],"selection_cases":[c.key for c,_ in dev],
                      "selection_baseline":original.tolist(),"parent_sha256":file_hash(PARENT),
                      "weights_sha256":weights_sha})
    print("Frozen graph",choice,best[:2],original.tolist(),flush=True)
    used={d["id"] for batch in cohorts.values() for d in batch}
    exclude=set(json.loads((EDGE/"overlap.json").read_text())["independent_exclusions_for_text_disjoint_check"])
    cohorts["independent"]=[d for d in independent_validation(used) if d["id"] not in exclude]
    original=json.loads((root/"sentence-joint-edit.json").read_text())["splits"]
    out={"freeze_sha256":file_hash(freeze),"splits":{}}
    for group in ("select","confirm","independent","test"):
        if group=="test":
            if fixed is None:
                break
            docs=[document(k,r) for k,r in gold_cases().items()]
            with ProcessPoolExecutor(3) as pool:
                rows={k:(a,b,c) for chunk in pool.map(test_chunk,[(docs[i::3],choice) for i in range(3)])
                      for k,a,b,c in chunk}
            ordered_rows=[rows[d["id"]] for d in docs]
            a,b=(np.stack([v[i] for v in ordered_rows]) for i in (0,1))
            changed=sum(v[2] for v in ordered_rows)
        else:
            cases=dev if group=="select" else prepare((cohorts[group],group))
            old=[c.current for c,_ in cases]
            if fixed is None:
                y=old
            else:
                arm,_,_,kind,budget,gain,guard,threshold=choice
                y=score(cases,responses(cases,ARMS.index(arm),fixed),route_cases(cases,budget),kind,gain,guard,threshold)
            a,b=confusion(cases,y),confusion(cases,old)
            changed=sum(int((x!=z).sum()) for x,z in zip(y,old))
        out["splits"][group]=q={"cases":len(a),"candidate":row_summary(a.sum(0)),
                                  "baseline":row_summary(b.sum(0)),"paired":paired(a,b),"changed":changed}
        assert q["baseline"]==original[group]["candidate"]
        if group=="select":
            np.testing.assert_allclose(measure(a.sum(0))[:2],best[:2],atol=1.e-12,rtol=0)
        save_json(root/(STEM+".json"),out)
        print(group,q["candidate"]["macro_f1"],q["baseline"]["macro_f1"],changed,flush=True)


if __name__=="__main__":
    main()
