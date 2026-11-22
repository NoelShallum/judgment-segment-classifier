"""Compare closed-form LegalSeg repairs using frozen EmbeddingGemma and hierarchy states."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
from time import perf_counter

import numpy as np
from safetensors.numpy import load_file,save_file

from probe_embeddinggemma_q4 import ARTIFACT,Encoder,PROMPT,REVISION
from sentence_hidden_edit import (
    CRF,DONOR,EDGE,OUT,ROOT,HierModel,aligned_embeddings,cm,document,edit_documents,embeddings,
    file_hash,gold_cases,independent_validation,prepare_score,row_summary,sample_whole,save_json,torch,transition_scores,
)
from predict_role_edge import decode_profile
from sentence_joint_runtime import joint_inputs,joint_path,neighbors
from sentence_role_retention import measure,paired
from sentence_uncertainty_runtime import probability

PARENT=OUT/"known-sentence-joint-portable.safetensors"
RESULT=ROOT/"results/embeddinggemma-routed-write.json"
FREEZE=ROOT/"results/embeddinggemma-routed-frozen.json"
CAP=64
WORKERS=4
BATCH=4
PER_CASE=4
ARMS=("hierarchy","gemma","combined")


@dataclass
class Case:
    key: str
    gold: np.ndarray
    current: np.ndarray
    prior: np.ndarray
    structure: np.ndarray
    texts: list
    eligible: np.ndarray


def case_inputs(d,h,model,p,vocab):
    bounds=d["reference_units"]
    text=d["text"]
    phrases=[text[a:b].strip() for a,b in bounds]
    item=prepare_score({**d,"units":bounds},model.logits(h),{"exact":p})
    raw=item.logits["exact"]
    lexical=transition_scores(text,bounds,vocab,p)
    gain=float(p["edge.gain"])
    initial=decode_profile(raw,lexical,p,gain)
    lengths=np.asarray([len(s) for s in phrases])
    anchor,rank,utility,w=joint_inputs(raw,lexical,p,gain,initial,lengths,h)
    _,_,rate,strength=w["settings"]
    current=joint_path(anchor,rank,utility,w,strength,rate,p[CRF])
    best=utility.argmax(1)
    order=np.sort(utility,axis=1)
    dist=order[:,-1]-order[:,-2]
    # Refer only ambiguous positions already eligible for the joint edit.
    route=np.flatnonzero(rank<rate)
    route=route[np.argsort(dist[route],kind="stable")]
    z=utility-utility.max(1,keepdims=True)
    q=np.exp(4*z)
    q/=q.sum(1,keepdims=True)
    one=np.eye(7)[current]
    lhs,rhs=neighbors(one,np.arange(len(one))==0)
    structure=np.column_stack((q,one,lhs,rhs,rank,dist,best==current,
                               (np.arange(len(one))+.5)/len(one),np.log1p(lengths),h)).astype("f4")
    return Case(d["id"],np.asarray(d["reference_roles"]),current,.6*q+.4*one,structure,phrases,route)


def rows_for_fit(cases):
    rng=np.random.default_rng(245501)
    pools=[[] for _ in range(7)]
    for j,c in enumerate(cases):
        for i in c.eligible[:max(8,len(c.eligible)//3)]:
            pools[c.gold[i]].append((j,i))
    picks=[]
    for pool in pools:
        for idx in rng.choice(len(pool),size=min(len(pool),72),replace=False):
            picks.append(pool[idx])
    return picks


def rows_for_evaluation(cases):
    return [(j,i) for j,c in enumerate(cases) for i in c.eligible[:PER_CASE]]


def infer_shard(texts):
    engine=Encoder(cap=CAP,threads=1)
    # Explicit length bucketing fixes this worker's batch composition and avoids padding long outliers.
    tokenizer=engine.tokenizer
    seq=tokenizer.encode_batch([PROMPT+t for t in texts])
    order=sorted(range(len(texts)),key=lambda i:sum(seq[i].attention_mask))
    vectors=np.empty((len(texts),768),np.float32)
    for start in range(0,len(order),BATCH):
        ids=order[start:start+BATCH]
        vectors[ids]=engine.encode([texts[i] for i in ids])
    assert np.isfinite(vectors).all()
    return vectors


def embeddings_for(cases,rows,tag):
    texts=[cases[j].texts[i] for j,i in rows]
    root=ROOT/"cache/embeddinggemma-routed-64"
    root.mkdir(exist_ok=True)
    digest=sha256(json.dumps([REVISION,CAP,WORKERS,BATCH,texts],ensure_ascii=False).encode()).hexdigest()
    path=root/f"{tag}-{digest}.npy"
    if path.is_file():
        return np.load(path,allow_pickle=False)
    now=perf_counter()
    print("Encoding",tag,len(texts),flush=True)
    with ProcessPoolExecutor(WORKERS) as executor:
        result=list(executor.map(infer_shard,[texts[i::WORKERS] for i in range(WORKERS)]))
    data=np.empty((len(texts),768),np.float32)
    for i,x in enumerate(result):
        data[i::WORKERS]=x
    stage=path.with_suffix(".part")
    with stage.open("wb") as file:
        np.save(file,data,allow_pickle=False)
    stage.replace(path)
    print("Encoded",tag,len(data),round(perf_counter()-now,2),flush=True)
    return data


def design(cases,rows,semantic,arm,dim):
    structural=np.stack([cases[j].structure[i] for j,i in rows])
    if arm=="hierarchy":
        return structural
    v=semantic[:,:dim]
    words=v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1.e-8)
    return words if arm=="gemma" else np.column_stack((structural,words))


def fit(cases,rows,semantic,arm,dim,damping):
    x=design(cases,rows,semantic,arm,dim)
    mean=x.mean(0)
    scale=1/np.maximum(x.std(0),.015)
    z=np.column_stack(((x-mean)*scale,np.ones(len(x)))).astype(np.float64)
    labels=np.asarray([cases[j].gold[i] for j,i in rows])
    gram=z.T@z
    diag=np.r_[np.ones(x.shape[1]),.05]
    coefs=np.linalg.solve(gram+np.diag(diag*damping*gram.trace()/len(gram)),z.T@np.eye(7)[labels])
    return {"center":mean,"scale":scale,"weight":np.ascontiguousarray(coefs[:-1],dtype=np.float32),
            "bias":np.ascontiguousarray(coefs[-1],dtype=np.float32)}


def predict(cases,rows,estimate,mix,margin):
    out=[c.current.copy() for c in cases]
    for (j,i),estimate_row in zip(rows,estimate):
        c=cases[j]
        values=(1-mix)*c.prior[i]+mix*estimate_row
        choice=values.argmax()
        if values[choice]-values[c.current[i]]>=margin:
            out[j][i]=choice
    return out


def scores(cases,rows,labels):
    a=np.stack([cm(c.gold,y) for c,y in zip(cases,labels)])
    b=np.stack([cm(c.gold,c.current) for c in cases])
    by_case=[[] for c in cases]
    for j,i in rows:
        by_case[j].append(i)
    r=np.stack([cm(c.gold[items],y[items]) for c,y,items in zip(cases,labels,by_case)])
    s=np.stack([cm(c.gold[items],c.current[items]) for c,items in zip(cases,by_case)])
    changes=sum(int((c.current!=y).sum()) for c,y in zip(cases,labels))
    return a,b,r,s,changes


def summarize(cases,rows,labels):
    a,b,r,s,n=scores(cases,rows,labels)
    return {"cases":len(a),"sentences":int(a.sum()),"routed":len(rows),"changed":n,
            "full":row_summary(a.sum(0)),"full_parent":row_summary(b.sum(0)),"full_ci":paired(a,b),
            "routed_scores":row_summary(r.sum(0)),"routed_parent":row_summary(s.sum(0)),
            "routed_ci":paired(r,s)}


def main():
    torch.set_num_threads(1)
    hierarchy=HierModel(DONOR)
    parent=load_file(str(PARENT))
    vocab=json.loads((EDGE/"vocab.json").read_text())
    fit_docs=edit_documents()
    groups=sample_whole()
    def prepare(docs,source):
        latents=embeddings(hierarchy,docs,"aligned-source-train") if source=="train" else aligned_embeddings(hierarchy,docs,source)
        return [case_inputs(d,h,hierarchy,parent,vocab) for d,h in zip(docs,latents)]

    train=prepare(fit_docs,"train")
    train_rows=rows_for_fit(train)
    u=embeddings_for(train,train_rows,"fit")
    params=[(arm,dim,ridge) for arm in ARMS for dim in ((0,) if arm=="hierarchy" else (128,256))
            for ridge in (.15,.60)]
    weights={k:fit(train,train_rows,u,*k) for k in params}
    dev=prepare(groups["select"],"select")
    rows=rows_for_evaluation(dev)
    v=embeddings_for(dev,rows,"select")
    grid=[];best={}
    gold=np.asarray([dev[j].gold[i] for j,i in rows])
    initial=np.asarray([dev[j].current[i] for j,i in rows])
    control=measure(cm(gold,initial)).tolist()
    for (arm,dim,ridge),w in weights.items():
        estimate=probability(design(dev,rows,v,arm,dim),w)
        for mix,margin in product((.35,.60,.85),(.025,.075,.15)):
            y=predict(dev,rows,estimate,mix,margin)
            yy=np.asarray([y[j][i] for j,i in rows])
            metrics=measure(cm(gold,yy)).tolist()
            choice=(arm,dim,ridge,mix,margin)
            grid.append([*choice,*metrics,int((yy!=initial).sum())])
            score=(metrics[0],metrics[1],choice)
            if arm not in best or score>best[arm]:
                best[arm]=score
    chosen={k:x[2] for k,x in best.items()}
    names={}
    fixed={}
    for arm,setting in chosen.items():
        path=OUT/f"embeddinggemma-routed-{arm}.safetensors"
        payload={**weights[setting[:3]],"settings":np.asarray(setting[1:])}
        save_file(payload,str(path))
        fixed[arm]=load_file(str(path))
        for k,x in payload.items():
            np.testing.assert_array_equal(x,fixed[arm][k])
        names[arm]=file_hash(path)
    save_json(FREEZE,{"choices":chosen,"weights_sha256":names,"parent_sha256":file_hash(PARENT),
              "train_cases":[d["id"] for d in fit_docs],"train_rows":[[train[j].key,int(i)] for j,i in train_rows],
              "selection_cases":[c.key for c in dev],"selection_parent_routed":control,"grid":grid,
              "encoder_revision":REVISION,"token_cap":CAP,"per_case_route":PER_CASE,"encoder_provenance":json.loads((ARTIFACT/"provenance.json").read_text())})
    print("Frozen options",best,"control",control,flush=True)
    report={"freeze_sha256":file_hash(FREEZE),"choices":chosen,"splits":{}}
    for key in ("select","confirm"):
        cases=dev if key=="select" else prepare(groups[key],key)
        batch=rows if key=="select" else rows_for_evaluation(cases)
        sem=v if key=="select" else embeddings_for(cases,batch,key)
        variants={}
        for arm,w in fixed.items():
            dim,_,mix,gap=w["settings"]
            estimate=probability(design(cases,batch,sem,arm,int(dim)),w)
            labels=predict(cases,batch,estimate,mix,gap)
            variants[arm]=summarize(cases,batch,labels)
        report["splits"][key]=variants
        save_json(RESULT,report)
        print(key,[(k,v["routed_scores"]["macro_f1"],v["full"]["macro_f1"],v["changed"]) for k,v in variants.items()],flush=True)


if __name__=="__main__":
    main()
