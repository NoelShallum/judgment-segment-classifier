"""Write local corrections for novel rhetorical spans using net match gain."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import product
import json

import numpy as np
from safetensors.numpy import save_file

from role_edge_interpolation import EDGE,DONOR,HierModel,ROOT,bootstrap,document,file_hash,full_score,gold_cases,independent_validation,load_file,metric,prepare,sample_whole,save_json,torch
from hier_full import CACHE
from role_pair_weight_search import edit_documents,embeddings,prepare_score,transition_scores
from segment_span_runtime import density_features
from segment_weight_write import engine
from short_span_weight_edit import setup
from span_posterior_decode import spans

MAX_RUNS=4
MAX_TIER=5
COUNT_COST=.15


@dataclass
class Repairs:
    case: object
    base: np.ndarray
    arcs: np.ndarray
    tier: np.ndarray
    groups: np.ndarray
    run_count: int
    features: np.ndarray
    gain: np.ndarray
    count: np.ndarray
    exact: np.ndarray


def bank(item,edge,parents,incumbent,decode):
    tokens=item.token_role_counts.sum(1)
    arcs,tier,feat,value=density_features(
        item.logits,edge,parents,tokens,incumbent,MAX_TIER)
    core=tier==0
    n=len(tokens)
    base=decode(arcs[core],value[core]-float(incumbent["span_cost"]),n)
    original=spans(base)
    starts,ends,role=original.T
    ai,bi,label=arcs.T
    lo=np.searchsorted(starts,ai,side="right")-1
    hi=np.searchsorted(starts,bi-1,side="right")-1
    left_ok=(ai==0)|(base[(ai-1).clip(0,n-1)]!=label)
    right_ok=(bi==n)|(base[bi.clip(0,n-1)]!=label)
    keep=(tier>0)&(bi-ai<=16)&(hi-lo<MAX_RUNS)&left_ok&right_ok
    arcs=arcs[keep];tier=tier[keep];x=feat[keep];v=value[keep]
    a,b,k=arcs.T
    lo=lo[keep];hi=hi[keep]
    l0=starts[lo];r0=ends[hi]
    fragment_l=a>l0
    fragment_r=b<r0
    match=lambda u,w,j:(item.gold_label[u]==j)&(item.gold_end[u]==item.offsets[w])
    prev=match(starts,ends,role)
    mass=np.r_[0,np.cumsum(prev)]
    removed=mass[hi+1]-mass[lo]
    exact=match(a,b,k)
    kept_l=fragment_l&match(l0,a,role[lo])
    kept_r=np.zeros(len(b),dtype=bool)
    kept_r[fragment_r]=match(b[fragment_r],r0[fragment_r],role[hi[fragment_r]])
    gain=exact.astype(np.int8)+kept_l+kept_r-removed
    count=fragment_l.astype(np.int8)+fragment_r+1-(hi-lo+1)

    # These features describe inference geometry and frozen scores only.
    width=hi-lo+1
    boundary=np.column_stack((fragment_l,fragment_r,a-l0,r0-b,width,count,
                              role[lo]==k,role[hi]==k,v))
    context=np.column_stack((np.log1p(a-l0),np.log1p(r0-b),np.log1p((b-a)/(r0-l0)),
                            np.eye(7)[role[lo]],np.eye(7)[role[hi]]))
    features=np.column_stack((x,boundary,context)).astype("f4")
    return Repairs(item,base,arcs,tier,lo*MAX_RUNS+hi-lo,len(starts),features,gain,count,exact)


def fit(rows):
    x=np.concatenate([r.features for r in rows])
    g=np.concatenate([r.gain for r in rows])
    d=np.concatenate([r.count for r in rows])
    good=g>0
    bad=g<0
    m=x.mean(0,dtype=np.float64)
    scale=1/np.maximum(x.std(0,dtype=np.float64),.06)
    target=np.column_stack((g-COUNT_COST*d,good,bad))
    penalty=np.r_[np.ones(len(m)),.04]
    for emphasis in (1.,12.,35.):
        base=.3+.7*(g!=0)+emphasis*good
        gram=np.zeros((len(penalty),len(penalty)))
        rhs=np.zeros((len(penalty),3))
        for i in range(0,len(x),16000):
            c=x[i:i+16000]
            z=np.column_stack(((c-m)*scale,np.ones(len(c))))
            w=base[i:i+len(c)]
            u=z.T*w
            gram+=u@z
            rhs+=u@target[i:i+len(c)]
        for ridge in (.045,.18):
            w=np.linalg.solve(gram+np.diag(penalty*ridge*np.trace(gram)/len(gram)),rhs)
            yield emphasis,ridge,{"center":m,"scale":scale,"weight":np.ascontiguousarray(w[:-1]),"bias":w[-1].copy()}


def evidence(row,weights):
    v=((row.features-weights["center"])*weights["scale"])@weights["weight"]+weights["bias"]
    return v[:,0],np.clip(v[:,1],0,1)-1.3*np.clip(v[:,2],0,1)-COUNT_COST*row.count


def compress(row,utility,scope):
    eligible=np.flatnonzero((row.tier>=2) if scope=="short" else np.ones(len(row.tier),bool))
    groups=row.groups[eligible]
    best=np.full(row.run_count*MAX_RUNS,-np.inf)
    np.maximum.at(best,groups,utility[eligible])
    candidates=eligible[utility[eligible]==best[groups]]
    index=np.full(len(best),-1,dtype=np.int32)
    index[row.groups[candidates]]=candidates
    return best.reshape(-1,MAX_RUNS),index.reshape(-1,MAX_RUNS)


def choose(row,packed,threshold):
    utility,index=packed
    n=row.run_count
    best=np.zeros(n+1)
    selected=np.full(n,-1,dtype=np.int32)
    for l in range(n-1,-1,-1):
        right=min(n-l,MAX_RUNS)
        far=np.minimum(np.arange(l+2,l+right+2),n)
        options=utility[l,:right]-threshold+best[far]
        r=int(np.argmax(options))
        best[l]=best[l+1]
        if options[r]>best[l] and utility[l,r]>threshold:
            best[l]=options[r]
            selected[l]=r
    choices=[]
    l=0
    while l<n:
        r=int(selected[l])
        if r<0:
            l+=1
        else:
            choices.append(int(index[l,r]))
            l+=r+2
    return np.asarray(choices,dtype=np.int32)


def evaluate(rows,packed,threshold):
    stats=[];counts=np.zeros(5,np.int64)
    for row,p in zip(rows,packed):
        chosen=choose(row,p,threshold)
        y=row.base.copy()
        for a,b,k in row.arcs[chosen]:
            y[a:b]=k
        before=row.case.score(row.base)
        actual=row.case.score(y)
        if len(chosen):
            np.testing.assert_array_equal(actual[49:51]-before[49:51],[row.gain[chosen].sum(),row.count[chosen].sum()])
        counts += (len(chosen),int((row.gain[chosen]>0).sum()),int((row.gain[chosen]<0).sum()),
                   int(row.exact[chosen].sum()),int((row.tier[chosen]>=2).sum()))
        stats.append(actual)
    return np.stack(stats),counts


def replay(args):
    docs,weights,mode,scope,threshold=args
    torch.set_num_threads(1)
    parents,vocab,_,incumbent=setup()
    decode=engine()
    a=[];b=[];count=np.zeros(5,np.int64)
    for d in docs:
        with np.load(CACHE/(d["id"]+".npz")) as f:
            item=prepare_score(d,f["raw"],parents)
        edge=transition_scores(d["text"],d["units"],vocab,parents["exact"])
        row=bank(item,edge,parents,incumbent,decode)
        u=evidence(row,weights)[mode]
        value,events=evaluate([row],[compress(row,u,scope)],threshold)
        a.append(value[0])
        b.append(item.score(row.base))
        count+=events
    return np.stack(a),np.stack(b),count


def main():
    torch.set_num_threads(2)
    parents,vocab,ipath,incumbent=setup()
    model=HierModel(DONOR)
    decode=engine()

    def pack(bank_,name):
        rows=[bank(item,edge,parents,incumbent,decode) for item,edge in bank_]
        print("Novel spans prepared",name,len(rows),sum(len(x.arcs) for x in rows),flush=True)
        return rows

    train=edit_documents()
    raw=[(prepare_score(d,model.logits(x),parents),transition_scores(d["text"],d["units"],vocab,parents["exact"]))
         for d,x in zip(train,embeddings(model,train,"raw-emitter-train"))]
    train_rows=pack(raw,"train")
    weights={(e,r):w for e,r,w in fit(train_rows)}
    counts={"candidates":sum(len(r.arcs) for r in train_rows),"exact":sum(int(r.exact.sum()) for r in train_rows),
            "helpful":sum(int((r.gain>0).sum()) for r in train_rows),"harmful":sum(int((r.gain<0).sum()) for r in train_rows)}
    print("Novel training evidence",counts,flush=True)
    del train_rows
    groups=sample_whole()
    selection=pack(prepare(groups["select"],"select",model,parents,vocab,parents["exact"]),"select")
    base=metric(np.stack([r.case.score(r.base) for r in selection]).sum(0))
    grid=[]
    best=(-1.,-1.);choice=None
    fallback=(-1.,-1.);fallback_choice=None
    for key,w in weights.items():
        local=[evidence(row,w) for row in selection]
        for mode,scope in product((0,1),("short","all")):
            packed=[compress(r,v[mode],scope) for r,v in zip(selection,local)]
            for threshold in (.03,.08,.15,.23,.34,.50,.70,1.):
                a,n=evaluate(selection,packed,threshold)
                q=metric(a.sum(0))
                spec=(*key,mode,scope,threshold)
                grid.append([*spec,*q,*n.tolist()])
                if n[0] and q>fallback:
                    fallback=q;fallback_choice=spec
                if n[0] and q>best and q[1]>=base[1]-.008:
                    choice=spec;best=q
    if choice is None:
        choice,best=fallback_choice,fallback
    assert choice is not None
    freeze=ROOT/"results/novel-span-repair-frozen.json"
    out=ROOT/"checkpoints/hierarchical/whole-novel-span-repair.safetensors"
    e,r,mode,scope,threshold=choice
    tensors={**weights[e,r],"mode":np.asarray(mode),"wide":np.asarray(scope=="all"),"threshold":np.asarray(threshold)}
    save_file(tensors,str(out))
    written=load_file(str(out))
    assert all(np.array_equal(v,written[k]) for k,v in tensors.items())
    save_json(freeze,{"choice":choice,"grid":grid,"counts":counts,"base":base,"selected":best,
          "train_cases":[d["id"] for d in train],"selection":[d["id"] for d in groups["select"]],
          "incumbent_sha256":file_hash(ipath),"weights_sha256":file_hash(out)})
    print("Novel repair choice",choice,best,base,flush=True)
    mode=int(written["mode"])
    scope="all" if bool(written["wide"]) else "short"
    threshold=float(written["threshold"])
    used={d["id"] for t in groups.values() for d in t}
    exclusions=set(json.loads((EDGE/"overlap.json").read_text())["independent_exclusions_for_text_disjoint_check"])
    groups["independent"]=[d for d in independent_validation(used) if d["id"] not in exclusions]
    report={"choice":choice,"freeze_sha256":file_hash(freeze),"splits":{},
        "protocol":"Score local replacements against the complete segment-exact path. A replacement must produce one new short span and can touch at most four old spans. Fit either a direct change in exact-match utility or separate probabilities of fixing and breaking spans. Decode with at least one unedited original span between replacements."}
    for name in ("select","confirm","independent","test"):
        if name!="test":
            rows=selection if name=="select" else pack(prepare(groups[name],name,model,parents,vocab,parents["exact"]),name)
            a,n=evaluate(rows,[compress(row,evidence(row,written)[mode],scope) for row in rows],threshold)
            b=np.stack([r.case.score(r.base) for r in rows])
        else:
            cases=[document(k,r) for k,r in gold_cases().items()]
            with ProcessPoolExecutor(4) as pool:
                chunks=list(pool.map(replay,[(cases[i::4],written,mode,scope,threshold) for i in range(4)]))
            a=np.concatenate([q[0] for q in chunks])
            b=np.concatenate([q[1] for q in chunks])
            n=sum(q[2] for q in chunks)
            expected=json.loads((ROOT/"results/segment-weight-qa.json").read_text())["scores"]
            assert full_score(b)==expected
        if name=="select":
            np.testing.assert_allclose(metric(a.sum(0)),best,rtol=0,atol=1e-12)
        event=dict(zip(("accepted","helpful","harmful","new_span_exact","short_only_proposals"),map(int,n)))
        report["splits"][name]={"cases":len(b),"candidate":full_score(a),"parent":full_score(b),
              "gap_ci95":bootstrap(a,b),"edits":event}
        save_json(ROOT/"results/novel-span-repair.json",report)
        print(name,metric(a.sum(0)),metric(b.sum(0)),event,flush=True)


if __name__=="__main__":
    main()
