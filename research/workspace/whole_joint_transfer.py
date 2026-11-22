"""Use a frozen sentence decoder to propose and score raw-text rhetorical spans."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import product
import json

import numpy as np
from safetensors.numpy import save_file

from role_edge_interpolation import (
    CRF,DONOR,EDGE,ROOT,HierModel,bootstrap,document,file_hash,full_score,gold_cases,
    independent_validation,load_file,metric,sample_whole,save_json,torch,
)
from hier_full import CACHE
from hier_trial import OUT
from precompute_raw_joint_states import DEST as RAW_STATES
from predict_role_edge import decode_profile
from role_pair_weight_search import edit_documents,embeddings,prepare_score,transition_scores
from segment_weight_write import engine
from short_span_weight_edit import setup
from whole_joint_runtime import adjustment,candidates,select_labels

JOINT=OUT/"known-sentence-joint-portable.safetensors"
NEW=OUT/"whole-joint-transfer.safetensors"


@dataclass
class Entries:
    item: object
    arcs: np.ndarray
    original_score: np.ndarray
    novel: np.ndarray
    features: np.ndarray
    role_path: np.ndarray


def bank(d,states,raw,parents,incumbent,joint,vocab):
    item=prepare_score(d,raw,{**parents,"joint":joint})
    ext=transition_scores(d["text"],d["units"],vocab,joint)
    frozen=item.logits["joint"]
    gain=float(joint["edge.gain"])
    selected=decode_profile(frozen,ext,joint,gain)
    lengths=[len(d["text"][a:b].strip()) for a,b in d["units"]]
    edge=transition_scores(d["text"],d["units"],vocab,parents["exact"])
    counts=item.token_role_counts.sum(1)
    return Entries(item,*candidates(
        item.logits,edge,ext,parents,incumbent,joint,counts,lengths,states,selected))


def exact(row):
    a,b,k=row.arcs.T
    return (row.item.gold_label[a]==k)&(row.item.gold_end[a]==row.item.offsets[b])


def fit(train):
    x=np.concatenate([r.features for r in train])
    z=np.concatenate([exact(r) for r in train]).astype(np.float64)
    base=np.concatenate([r.original_score for r in train])
    novel=np.concatenate([r.novel for r in train])
    mean=x.mean(0)
    scale=1/np.maximum(x.std(0),.06)
    k=np.column_stack(((x-mean)*scale,np.ones(len(x))))
    p=np.r_[np.ones(k.shape[1]-1),.035]
    for prior in (0.,4.):
        w=1+prior*z+.9*novel
        gram=k.T@(w[:,None]*k)
        rhs=k.T@(w*(z-base))
        for penalty in (.08,.35):
            out=np.linalg.solve(gram+np.diag(p*penalty*np.trace(gram)/len(gram)),rhs)
            yield (prior,penalty),{"center":mean,"scale":scale,"coef":out[:-1].copy(),"bias":np.asarray(out[-1])}


def residual(row,weights):
    return adjustment(row.features,weights)


def label(row,decode,adjustment,strength,cost,fee,use_extra):
    return select_labels(row.arcs,row.original_score,row.novel,len(row.role_path),decode,
                         adjustment,strength,cost,fee,use_extra)


def metrics(rows,decode,adjust=None,strength=0.,cost=.24,fee=0.,use_extra=False):
    if adjust is None:
        adjust=[np.zeros(len(r.arcs)) for r in rows]
    return np.stack([r.item.score(label(r,decode,u,strength,cost,fee,use_extra)) for r,u in zip(rows,adjust)])


def replay(args):
    docs,choice,physical=args
    torch.set_num_threads(1)
    parents,vocab,_,incumbent=setup()
    joint=load_file(str(JOINT))
    decode=engine()
    result=[];base=[];transfer=[];coverage=np.zeros(4,np.int64)
    for d in docs:
        with np.load(CACHE/(d["id"]+".npz")) as f:
            raw=f["raw"]
        path=RAW_STATES/(d["id"]+".npy")
        if not path.is_file():
            raise FileNotFoundError(path)
        x=np.load(path,allow_pickle=False)
        row=bank(d,x,raw,parents,incumbent,joint,vocab)
        g,cost,fee,wide=physical["settings"]
        y=label(row,decode,residual(row,physical),g,cost,fee,bool(wide))
        z=label(row,decode,np.zeros(len(row.arcs)),0.,float(incumbent["span_cost"]),0.,False)
        known=exact(row)
        coverage+=(len(row.arcs),row.novel.sum(),known.sum(),(known&row.novel).sum())
        result.append(row.item.score(y))
        base.append(row.item.score(z))
        transfer.append(row.item.score(row.role_path))
    return np.stack(result),np.stack(base),np.stack(transfer),coverage


def main():
    torch.set_num_threads(2)
    d=engine()
    parents,vocab,incpath,inc=setup()
    joint=load_file(str(JOINT))
    encoder=HierModel(DONOR)
    groups=sample_whole()
    docs=edit_documents()
    training=[bank(q,h,encoder.logits(h),parents,inc,joint,vocab) for q,h in zip(docs,embeddings(encoder,docs,"raw-emitter-train"))]
    weights=dict(fit(training))
    count=sum(len(r.arcs) for r in training)
    added=sum(int(r.novel.sum()) for r in training)
    hits=sum(int((r.novel&exact(r)).sum()) for r in training)
    del training
    print("Raw joint training",len(docs),count,added,hits,flush=True)
    def prepare(which):
        cases=groups[which]
        rows=[bank(x,h,encoder.logits(h),parents,inc,joint,vocab) for x,h in
              zip(cases,embeddings(encoder,cases,"raw-joint-transfer-"+which))]
        print("Joint transfer prepared",which,len(rows),sum(r.novel.sum() for r in rows),flush=True)
        return rows

    selected=prepare("select")
    baseline=metric(metrics(selected,d,cost=float(inc["span_cost"])).sum(0))
    grid=[];best=None;fallback=None
    for key,w in weights.items():
        values=[residual(r,w) for r in selected]
        for strength,cost,wide in product((0.,.18,.42,.75),(.12,.20,.24,.30), (False,True)):
            for fee in ((.0,.06) if wide else (.0,)):
                q=metric(metrics(selected,d,values,strength,cost,fee,wide).sum(0))
                s=(*key,strength,cost,fee,wide)
                grid.append([*s,*q])
                if strength==0 and not wide and cost==float(inc["span_cost"]):
                    continue
                k=(*q,s)
                if fallback is None or k>fallback:
                    fallback=k
                if q[1]>=baseline[1]-.012 and (best is None or k>best):
                    best=k
    choice=(best or fallback)[2]
    physical={**weights[choice[:2]],"settings":np.asarray(choice[2:])}
    save_file(physical,str(NEW))
    restored=load_file(str(NEW))
    assert all(np.array_equal(v,restored[k]) for k,v in physical.items())
    f=ROOT/"results/whole-joint-transfer-frozen.json"
    save_json(f,{"choice":choice,"grid":grid,"preceding":baseline,"train_arcs":int(count),
        "train_novel_arcs":int(added),"train_exact_novel":int(hits),"weights_sha256":file_hash(NEW),
        "joint_sha256":file_hash(JOINT),"segment_sha256":file_hash(incpath),"selection_cases":[x["id"] for x in groups["select"]]})
    print("Joint transfer selected",choice,(best or fallback)[:2],baseline,flush=True)
    used={q["id"] for batch in groups.values() for q in batch}
    dropped=set(json.loads((EDGE/"overlap.json").read_text())["independent_exclusions_for_text_disjoint_check"])
    groups["independent"]=[x for x in independent_validation(used) if x["id"] not in dropped]
    report={"choice":choice,"freeze_sha256":file_hash(f),"splits":{},
        "protocol":"Use rule-selected raw units without supplied boundaries. Transfer the frozen joint sentence decoder to those raw units and propose intervals from its decisions with or without transition scores. The original segment candidates remain available. A closed-form correction estimates exact span matches from the original and new decoder scores."}
    for which in ("select","confirm","independent","test"):
        if which!="test":
            rows=selected if which=="select" else prepare(which)
            strength,cost,fee,wide=restored["settings"]
            a=metrics(rows,d,[residual(r,restored) for r in rows],strength,cost,fee,bool(wide))
            b=metrics(rows,d,cost=float(inc["span_cost"]))
            c=np.stack([r.item.score(r.role_path) for r in rows])
            coverage=sum(np.asarray([len(r.arcs),r.novel.sum(),exact(r).sum(),(r.novel&exact(r)).sum()]) for r in rows)
        else:
            cases=[document(k,r) for k,r in gold_cases().items()]
            with ProcessPoolExecutor(4) as pool:
                pieces=list(pool.map(replay,[(cases[i::4],choice,restored) for i in range(4)]))
            a,b,c=[np.concatenate([shard[i] for shard in pieces]) for i in range(3)]
            coverage=sum(shard[3] for shard in pieces)
            qa=json.loads((ROOT/"results/segment-weight-qa.json").read_text())["scores"]
            assert full_score(b)==qa
        if which=="select":
            np.testing.assert_allclose(metric(a.sum(0)),(best or fallback)[:2],atol=1e-12,rtol=0.)
        report["splits"][which]={"cases":len(b),"candidate":full_score(a),"preceding":full_score(b),
           "standalone_sentence_joint":full_score(c),"gap_ci95":bootstrap(a,b),
           "candidate_coverage":dict(zip(("all_arcs","joint_added_arcs","exact_arcs","joint_added_exact_arcs"),map(int,coverage)))}
        save_json(ROOT/"results/whole-joint-transfer-edit.json",report)
        print(which,metric(a.sum(0)),metric(b.sum(0)),metric(c.sum(0)),coverage.tolist(),flush=True)


if __name__=="__main__":
    main()
