"""Write role heads over an audited 20% quantized MiniLM referral cache."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from itertools import product
import json

import numpy as np
from safetensors.numpy import load_file,save_file

import compare_broad_edits_to_bert as broad
import sentence_minilm_fusion as original
import sentence_pairwise_consensus as consensus
import sentence_pairwise_repair as pair
import sentence_phrase_sequence as seq
from sentence_document_graph_write import OUT, cm, file_hash, paired, row_summary, save_json, torch

P=pair.P
TAG="sentence-minilm-whitened"
ORIGIN="sentence-minilm-fusion"
ARMS=("structure","semantic","whitened","whitened with neighbors")
SPLITS=("select","confirm","independent","test")
RATE=.20
DEFAULT_TAG=TAG


def source():
    plan=pair.receipt(ORIGIN)
    cfg=json.loads(plan.read_text())
    assert file_hash(original.GRAPH)==cfg["graph_sha256"]
    assert file_hash(original.MAIN)==cfg["baseline_sha256"]
    return cfg,file_hash(plan),P.ROOT/"cache"/f'minilm-sentence-fusion-{cfg["graph_sha256"][:12]}-{cfg["sentence_token_limit"]}'


def docs(split):
    if split=="fit":
        return original.edit_documents()
    if split=="independent":
        used={d["id"] for batch in original.sample_whole().values() for d in batch}
        excluded=set(json.loads((original.EDGE/"overlap.json").read_text())["independent_exclusions_for_text_disjoint_check"])
        return [d for d in original.independent_validation(used) if d["id"] not in excluded]
    return P.docs(split)


def replay_positions(job):
    docs, split, rate = job
    torch.set_num_threads(1)
    model=original.HierModel(original.DONOR)
    parent=load_file(str(original.MAIN))
    vocab=json.loads((original.EDGE/"vocab.json").read_text())
    states=(original.embeddings(model,docs,"aligned-source-train") if split=="fit"
            else original.aligned_embeddings(model,docs,split))
    return {d["id"]:original.case_view(d,h,model,parent,vocab,rate).query for d,h in zip(docs,states)}


def cache_rows(documents, cache_dir, case_rate, split, trusted=None):
    paths=[cache_dir/f'{d["id"]}-{case_rate:.2f}.npz' for d in documents]
    q,feat,receipt=[],[],[]
    for path,d in zip(paths,documents):
        n=len(d["reference_units"])
        with np.load(path,allow_pickle=False) as f:
            index,x=f["query"],f["x"]
            layout=f["layout"].item() if "layout" in f else None
        assert len(index)==int(np.ceil(case_rate*n))
        assert x.shape==(len(index),384) and np.isfinite(x).all()
        assert np.unique(index).size==len(index) and ((0<=index)&(index<n)).all()
        q.append(index)
        feat.append(x)
        receipt.append(layout)
    trace=sha256()
    for lo in range(0,len(documents),36):
        payload=[]
        for d,index in zip(documents[lo:lo+36],q[lo:lo+36]):
            bounds=d["reference_units"]
            texts=[d["text"][bounds[j][0]:bounds[j][1]].strip() for j in index]
            payload.append((d["id"],index.tolist(),texts))
        check=sha256(json.dumps(payload,ensure_ascii=False).encode()).hexdigest()
        assert all(s is None or s==check for s in receipt[lo:lo+36])
        trace.update(check.encode())
    for path in paths:
        trace.update(file_hash(path).encode())
    digest=trace.hexdigest()
    if any(s is None for s in receipt) and trusted!=digest:
        with ProcessPoolExecutor(4) as pool:
            expected={k:v for part in pool.map(replay_positions,[(documents[i::4],split,case_rate) for i in range(4)])
                      for k,v in part.items()}
        for d,idx in zip(documents,q):
            np.testing.assert_array_equal(expected[d["id"]],idx)
    print(split, "Verified cached MiniLM positions",sum(map(len,q)),flush=True)
    use=[int(np.ceil(RATE*len(d["reference_units"]))) for d in documents]
    return [idx[:n] for idx,n in zip(q,use)],[x[:n] for x,n in zip(feat,use)],digest


def build(docs,split):
    path=P.ROOT/"cache"/(DEFAULT_TAG+"-"+split+"-dense.npz")
    keys=np.array([d["id"] for d in docs])
    if not path.is_file():
        torch.set_num_threads(1)
        with ProcessPoolExecutor(4) as pool:
            by={r[0]:r for part in pool.map(P.dense.block,[(docs[i::4],split) for i in range(4)]) for r in part}
        ordered=[by[k] for k in keys]
        ends=np.r_[0,np.cumsum([len(r[1]) for r in ordered],dtype=np.int64)]
        stage=path.with_suffix(".part")
        with stage.open("wb") as stream:
            np.savez(stream,cases=keys,offsets=ends,**dict(zip(("gold","current","anchor","features"),
                    [np.concatenate([x[j] for x in ordered]) for j in (1,2,3,4)])))
        stage.replace(path)
    with np.load(path,allow_pickle=False) as f:
        bank=tuple(f[k] for k in ("cases","offsets","gold","current","anchor","features"))
    np.testing.assert_array_equal(keys,bank[0])
    np.testing.assert_array_equal(bank[2],np.concatenate([d["reference_roles"] for d in docs]))
    return bank


def independent_lexical(documents):
    proof=P.dense.read(P.TAG+"-lexical-frozen")
    vocab=OUT/(P.TAG+"-vocabulary.json")
    matrix=OUT/(P.TAG+"-lexical.safetensors")
    assert file_hash(vocab)==proof["vocabulary_sha256"] and file_hash(matrix)==proof["weights_sha256"]
    vocab=P.vectorizers(json.loads(vocab.read_text()))
    w=load_file(str(matrix))["weight"]
    texts=[P.canonical(d["text"][a:b].strip()) for d in documents for a,b in d["reference_units"]]
    return P.sparse(vocab,texts)@w


def study(split,cache_dir,recipe=None,pair_w=None,parent=None,trusted=None):
    documents=docs(split)
    cache_rate=.30 if split in ("fit","select") else .20
    q,x,signature=cache_rows(documents,cache_dir,cache_rate,split,trusted)
    bank=build(documents,split) if split in ("fit","independent") else P.dense.prepare(split)
    np.testing.assert_array_equal(bank[0],[d["id"] for d in documents])
    positions=np.concatenate([a+idx for a,idx in zip(bank[1],q)])
    base=None
    if split!="fit":
        lex=independent_lexical(documents) if split=="independent" else P.cached_scores(split)
        prev,unary,_=pair.incumbent(bank,lex,recipe)["sequence only"]
        mode,scale,top,cutoff,guard=parent["arms"]["sequence only"]["choice"]
        pair_scores=pair.score_pairs(pair.dense_basis(bank,lex),pair_w,mode)
        contest=pair.proposals(pair_scores,prev,unary,scale,top)
        base=pair.decide(contest,prev,np.zeros(len(prev),bool),cutoff,guard)
        if split!="independent":
            assert row_summary(cm(bank[2],base))==P.dense.read(pair.TAG)["splits"][split]["arms"]["sequence only"]["labels"]
    return bank,q,x,positions,base,signature


def whiten(v):
    center=v.mean(0,dtype=np.float64)
    residual=v.astype(np.float64)-center
    cov=(residual.T@residual)/len(v)
    val,vec=np.linalg.eigh(cov)
    use=val[-96:]
    transform=vec[:,-96:]/np.sqrt(use+.12*np.mean(use))
    return {"semantic_center":center.astype(np.float32),"semantic_transform":np.ascontiguousarray(transform,dtype=np.float32)}


def neighbor(values,positions):
    order=np.argsort(positions)
    ids=positions[order]
    z=values[order]
    prefix=np.vstack((np.zeros((1,z.shape[1])),np.cumsum(z,axis=0,dtype=np.float64)))
    count=np.arange(len(ids))
    left=np.searchsorted(ids,ids-12,side="left")
    right=np.searchsorted(ids,ids+12,side="right")
    past=(prefix[count]-prefix[left])/np.maximum(count-left,1)[:,None]
    future=(prefix[right]-prefix[count+1])/np.maximum(right-count-1,1)[:,None]
    merged=np.column_stack(((past+future)*.5,future-past)).astype(np.float32)
    return merged[np.argsort(order)]


def features(view,arm,w):
    bank,q,vectors,positions,_,_=view
    structure=bank[-1][positions,:54]
    if arm=="structure":
        return structure
    x=np.concatenate(vectors)
    if arm=="semantic":
        return np.column_stack((structure,x))
    z=(x-w["semantic_center"])@w["semantic_transform"]
    if arm=="whitened":
        return np.column_stack((structure,z))
    lengths=np.r_[0,np.cumsum([len(i) for i in q])]
    context=np.concatenate([neighbor(z[a:b],idx) for a,b,idx in zip(lengths,lengths[1:],q)])
    return np.column_stack((structure,z,context))


def edit(view,response,gain,margin):
    bank,q,_,positions,base,_=view
    current=base[positions]
    prior=.65*bank[4][positions]+.35*np.eye(7)[current]
    probabilities=(1.-gain)*prior+gain*response
    y=probabilities.argmax(1)
    allowed=probabilities[np.arange(len(y)),y]-probabilities[np.arange(len(y)),current]>=margin
    result=base.copy()
    result[positions[allowed]]=y[allowed]
    return result


def fit(diagnostic=False):
    origin,source_hash,cache_dir=source()
    p,w,p_sha=consensus.inputs()
    recipe=pair.load_recipe()
    prior=P.dense.read(DEFAULT_TAG+"-frozen") if diagnostic else {}
    if prior:
        assert prior["miniLM_source_freeze_sha256"]==source_hash
    train=study("fit",cache_dir,trusted=prior.get("fit_layout_sha256"))
    val=study("select",cache_dir,recipe,w,p,trusted=prior.get("selection_layout_sha256"))
    assert train[0][0].tolist()==origin["training_cases"] and val[0][0].tolist()==origin["selection_cases"]
    assert not set(train[0][0]) & set(val[0][0])
    cfg=whiten(np.concatenate(train[2]))
    roles=train[0][2][train[3]]
    gold=val[0][2]
    base=row_summary(cm(gold,val[4]))
    choices={}
    for i,arm in enumerate(ARMS):
        train_x=features(train,arm,cfg)
        x=features(val,arm,cfg)
        best=[base["macro_f1"],base["accuracy"],0]
        winner=None
        signatures=[]
        for rarity in (0.,.28,.55):
            for damp,weight in pair.fusion.bert.fit(train_x,roles,rarity):
                response=pair.fusion.bert.response(x,weight)
                for gain,gap in product((.22,.4,.62,.85,1.),(.02,.06,.13,.24)):
                    out=edit(val,response,gain,gap)
                    r=row_summary(cm(gold,out))
                    stats=[r["macro_f1"],r["accuracy"],-int(np.count_nonzero(out!=val[4]))]
                    setting=(rarity,damp,gain,gap)
                    signatures.append([setting,stats])
                    acceptable=(stats[2]<=-40 and (winner is None or stats>best)) if diagnostic else (
                        stats>best and r["accuracy"]>=base["accuracy"]-.0015)
                    if acceptable:
                        best,winner=stats,setting
                        patch=OUT/(TAG+f"-{i}.safetensors")
                        tensors={k:np.ascontiguousarray(v) for k,v in {**cfg,**weight}.items()}
                        save_file(tensors,str(patch))
                        assert all(np.array_equal(v,load_file(str(patch))[k]) for k,v in tensors.items())
        choices[arm]={"settings":winner,"selection":best,"sha256":file_hash(patch) if winner else None,"grid":signatures}
        print(arm,best,winner,flush=True)
    save_json(pair.receipt(TAG),{"miniLM_source_freeze_sha256":source_hash,"pair_freeze_sha256":p_sha,
                "graph_sha256":origin["graph_sha256"],"train_cases":train[0][0].tolist(),"select_cases":val[0][0].tolist(),
                "fit_layout_sha256":train[5],"selection_layout_sha256":val[5],"gradient_steps":0,"referral_rate":RATE,
                "fit_rows":len(roles),"minimum_development_revisions":40 if diagnostic else 0,"arms":choices})


def evaluate(splits):
    _,source_hash,cache_dir=source()
    parent,w,p_sha=consensus.inputs()
    recipe=pair.load_recipe()
    path=pair.receipt(TAG)
    frozen=json.loads(path.read_text())
    assert frozen["miniLM_source_freeze_sha256"]==source_hash and frozen["pair_freeze_sha256"]==p_sha
    chosen=max(frozen["arms"],key=lambda name:frozen["arms"][name]["selection"])
    overlay_path=pair.receipt(broad.overlay.TAG)
    overlay=json.loads(overlay_path.read_text())
    assert overlay["sources"][pair.TAG]==p_sha
    bert_cfg=broad.B.read(broad.B.TAG+"-frozen")
    assert overlay["sources"][broad.B.TAG]==file_hash(pair.receipt(broad.B.TAG))
    bert_rule=overlay["arms"]["BERT only"]["choice"]
    out={"freeze_sha256":file_hash(path),"development_selected_arm":chosen,
         "bert_overlay_sha256":file_hash(overlay_path),"splits":{}}
    previous=P.dense.read(DEFAULT_TAG) if TAG!=DEFAULT_TAG else {"splits":{}}
    for split in splits:
        known=previous["splits"].get(split,{}).get("layout_sha256")
        view=study(split,cache_dir,recipe,w,parent,trusted=known)
        bank=view[0]
        assert not set(frozen["train_cases"]) & set(bank[0])
        if split!="select":
            assert not set(frozen["select_cases"]) & set(bank[0])
        else:
            assert view[5]==frozen["selection_layout_sha256"]
        base=P.dense.paired_confusion(bank,view[4])
        span=seq.span_tally(bank,view[4])
        repaired=view[4].copy()
        repaired[view[3]]=bank[2][view[3]]
        result=out["splits"][split]={"cases":len(bank[0]),"referrals":len(view[3]),"layout_sha256":view[5],
                                    "base":row_summary(base.sum(0)),"base_spans":seq.span_info(span),"arms":{},
                                    "oracle_macro_f1":row_summary(cm(bank[2],repaired))["macro_f1"],
                                    "errors_reachable":int(np.count_nonzero(repaired!=view[4]))}
        for i,(arm,a) in enumerate(frozen["arms"].items()):
            p=view[4]
            if a["settings"]:
                patch=OUT/(TAG+f"-{i}.safetensors")
                assert file_hash(patch)==a["sha256"]
                cfg=load_file(str(patch))
                response=pair.fusion.bert.response(features(view,arm,cfg),cfg)
                p=edit(view,response,*a["settings"][-2:])
            diff=np.flatnonzero(p!=view[4])
            assert np.isin(diff,view[3]).all()
            m=P.dense.paired_confusion(bank,p)
            scores=row_summary(m.sum(0))
            if split=="select":
                assert [scores["macro_f1"],scores["accuracy"],-len(diff)]==a["selection"]
            edited_span=seq.span_tally(bank,p)
            result["arms"][arm]={"scores":scores,"paired":paired(m,base),"changed":len(diff),
                                 "spans":seq.span_info(edited_span),"paired_spans":seq.span_intervals(edited_span,span)}
            if arm==chosen and split!="independent":
                bert=broad.bert_reference(split,bank,view[4],bert_cfg,bert_rule)
                hybrid=np.where(bert!=view[4],bert,p)
                prior,candidate=(P.dense.paired_confusion(bank,z) for z in (bert,hybrid))
                result["composed_with_bert"]={"base":row_summary(prior.sum(0)),"scores":row_summary(candidate.sum(0)),
                                             "paired":paired(candidate,prior),"changed":int(np.count_nonzero(hybrid!=bert))}
        save_json(P.RESULT/(TAG+".json"),out)
        print(split,{k:(v["scores"]["macro_f1"],v["scores"]["accuracy"],v["changed"])
                     for k,v in result["arms"].items()},flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("fit","evaluate"))
    parser.add_argument("splits",nargs="*")
    parser.add_argument("--diagnostic",action="store_true")
    args=parser.parse_args()
    if args.diagnostic:
        TAG=DEFAULT_TAG+"-active"
    if args.action=="fit":
        fit(args.diagnostic)
    else:
        evaluate(args.splits or SPLITS)
