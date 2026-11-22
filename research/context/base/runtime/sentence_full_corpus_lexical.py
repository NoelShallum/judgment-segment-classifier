"""Write a corpus-scale lexical memory and a closed-form judgment readout."""
import argparse
from collections import Counter
import csv
from itertools import product
import json

import numpy as np
from safetensors.numpy import load_file,save_file
from sklearn.feature_extraction.text import HashingVectorizer

import sentence_pairwise_consensus as consensus
import sentence_pairwise_repair as repair
import sentence_phase_readout_write as phase
from sentence_document_graph_write import OUT, cm, file_hash, measure, paired, row_summary, save_json
from program_trial import TO_ID

P=repair.P
TAG="sentence-full-corpus-lexical"
MEMORY=TAG+"-memory"
DATA=P.ROOT/"references/official/train.csv"
SPLITS=("select","confirm","holdout","phrase-prospect","test")
CACHE=P.ROOT/"cache"/TAG
WIDTH=1<<17
ARMS=("existing features","full corpus lexical")
VEC=HashingVectorizer(n_features=WIDTH,alternate_sign=False,binary=True,norm=None,
                      ngram_range=(1,3),lowercase=False,dtype=np.float32)


def rows():
    with DATA.open(newline="",encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["Text"].strip():
                yield row["Index"],row["Text"],TO_ID[row["Label"]]


def save_patch(tensors,path):
    tensors={k:np.ascontiguousarray(v) for k,v in tensors.items()}
    save_file(tensors,str(path))
    check=load_file(str(path))
    assert all(np.array_equal(v,check[k]) for k,v in tensors.items())


def write_memory():
    plan=repair.receipt(repair.TAG)
    frozen=json.loads(plan.read_text())
    excluded=set(frozen["fit_cases"])
    sizes=Counter(case for case,_,_ in rows() if case not in excluded)
    seen=Counter()
    counts=np.zeros((3*7,WIDTH),np.float64)
    role_counts=np.zeros(3*7,np.int64)
    words,modes=[],[]
    def flush():
        x=VEC.transform(words)
        idx=np.repeat(np.array(modes,dtype=np.int64),np.diff(x.indptr))*WIDTH+x.indices
        counts[:]+=np.bincount(idx,minlength=3*7*WIDTH).reshape(3*7,WIDTH)
        role_counts[:]+=np.bincount(modes,minlength=3*7)
        words.clear()
        modes.clear()
    for case,text,role in rows():
        if case in excluded:
            continue
        where=min((3*seen[case]+1)//sizes[case],2)
        seen[case]+=1
        words.append(P.canonical(text))
        modes.append(7*where+role)
        if len(words)==8192:
            flush()
            if int(role_counts.sum())%131072==0:
                print("Counted",int(role_counts.sum()),"sentences",flush=True)
    if words:
        flush()
    assert seen==sizes
    label_counts=counts.reshape(3,7,WIDTH).sum(0)
    pm=(label_counts.sum(0)+.5)
    pm/=pm.sum()
    posterior=(label_counts+64*pm[None,:]+.1)
    posterior/=posterior.sum(1,keepdims=True)
    unconditional=np.log(posterior)-np.log(pm)
    phase_specific=counts.reshape(3,7,WIDTH)+8000*posterior[None,:,:]+.1
    phase_specific/=phase_specific.sum(2,keepdims=True)
    conditional=np.log(phase_specific)-np.log(pm[None,None,:])
    path=OUT/(MEMORY+".safetensors")
    save_patch({"global":unconditional.T.astype(np.float32),"phase":conditional.transpose(0,2,1).astype(np.float32)},path)
    record={"train_csv_sha256":file_hash(DATA),"pair_freeze_sha256":file_hash(plan),"model_sha256":file_hash(path),
            "training_cases":sorted(seen),"calibration_cases_excluded":frozen["fit_cases"],"feature_slots":WIDTH,
            "training_rows":int(role_counts.sum()),"role_phase_counts":role_counts.reshape(3,7).tolist(),"gradient_steps":0}
    save_json(repair.receipt(MEMORY),record)
    print("Wrote memory on",len(seen),"judgments,",record["training_rows"],"sentences",flush=True)


def memory():
    plan=repair.receipt(MEMORY)
    spec=json.loads(plan.read_text())
    path=OUT/(MEMORY+".safetensors")
    assert spec["model_sha256"]==file_hash(path)
    assert spec["train_csv_sha256"]==file_hash(DATA)
    assert spec["pair_freeze_sha256"]==file_hash(repair.receipt(repair.TAG))
    return spec,load_file(str(path)),file_hash(plan)


def lexical(bank,split,spec,w):
    CACHE.mkdir(exist_ok=True)
    path=CACHE/(split+"-"+spec["model_sha256"][:14]+".npz")
    if path.is_file():
        with np.load(path,allow_pickle=False) as cache:
            np.testing.assert_array_equal(cache["cases"],bank[0])
            np.testing.assert_array_equal(cache["offsets"],bank[1])
            return cache["features"]
    ids,offsets,labels,_,texts=P.corpus(split)
    np.testing.assert_array_equal(ids,bank[0])
    np.testing.assert_array_equal(offsets,bank[1])
    np.testing.assert_array_equal(labels,bank[2])
    phase_ids=np.concatenate([np.minimum((3*np.arange(b-a)+1)//(b-a),2) for a,b in zip(offsets,offsets[1:])])
    feats=[]
    for a in range(0,len(texts),8192):
        x=VEC.transform(texts[a:a+8192])
        n=np.sqrt(np.maximum(np.diff(x.indptr),6))[:,None]
        c=phase_ids[a:a+len(n)]
        one=x@w["global"]
        two=np.empty_like(one)
        for j in range(3):
            two[c==j]=x[c==j]@w["phase"][j]
        feats.append(np.column_stack((one,two))/n)
    packed=np.concatenate(feats)
    features=np.empty((len(packed),14*5),np.float32)
    for a,b in zip(offsets,offsets[1:]):
        values=packed[a:b]
        features[a:b]=np.column_stack((values,*tuple(P.dense.windows(values))[:4]))
    stage=path.with_suffix(".part")
    with stage.open("wb") as f:
        np.savez(f,cases=ids,offsets=offsets,features=features)
    stage.replace(path)
    print("Scored",split,len(features),"sentences",flush=True)
    return features


def design(bank,lex,extra,arm):
    plain=repair.dense_basis(bank,lex)
    return np.column_stack((plain,extra)) if arm==ARMS[1] else plain


def standardized(x,w):
    z=np.clip((x-w["center"])*w["scale"],-5.,5.)
    return np.column_stack((z,np.ones(len(z),np.float32)))


def fit():
    spec,weight,m_sha=memory()
    parent,pair_w,p_sha=consensus.inputs()
    recipe=repair.load_recipe()
    train=P.dense.prepare("fit")
    train_lex=P.cached_scores("fit")
    val,val_lex,inc,u=phase.view("select",parent,pair_w,recipe)
    assert set(train[0])==set(spec["calibration_cases_excluded"])
    assert not set(val[0]) & (set(train[0])|set(spec["training_cases"]))
    lex_t=lexical(train,"fit",spec,weight)
    lex_s=lexical(val,"select",spec,weight)
    gold=train[2]
    m,a,*_=measure(cm(val[2],inc))
    standard=[float(m),float(a),0]
    choice={k:{"settings":None,"selection":standard} for k in ARMS}
    for j,arm in enumerate(ARMS):
        x=design(train,train_lex,lex_t,arm)
        pref={"center":x.mean(0,dtype=np.float64).astype(np.float32),
              "scale":(1/np.maximum(x.std(0,dtype=np.float64),.025)).astype(np.float32)}
        fit_basis=standardized(x,pref)
        del x
        valid=standardized(design(val,val_lex,lex_s,arm),pref)
        prior=np.maximum(np.bincount(gold,minlength=7)/len(gold),.015)
        for rare in (0.,.3):
            mass=(prior[gold]**-rare).astype(np.float32)
            gram,link=P.dense.fit_moments(fit_basis,gold,train[4],mass)
            for damping in (.02,.09,.26):
                matrix=phase.solve(gram,link,damping).astype(np.float32)
                estimate=valid@matrix
                for cfg in product(("posterior","residual"),(.22,.50,.85,1.3,2.),(.02,.075,.16,.27),("all","keep issue")):
                    pred=phase.edit(val,inc,u,estimate,cfg)
                    m,a,*_=measure(cm(val[2],pred))
                    stat=[float(m),float(a),-int(np.count_nonzero(pred!=inc))]
                    if stat>choice[arm]["selection"] and a>=standard[1]-.0015:
                        path=OUT/(TAG+f"-{j}.safetensors")
                        save_patch({**pref,"weight":matrix},path)
                        choice[arm]={"settings":[rare,damping,*cfg],"selection":stat,"sha256":file_hash(path)}
        print(arm,choice[arm],flush=True)
    save_json(repair.receipt(TAG),{"memory_freeze_sha256":m_sha,"pairwise_freeze_sha256":p_sha,"gradient_steps":0,
              "calibration_cases":train[0].tolist(),"selection_cases":val[0].tolist(),"arms":choice,"baseline":standard})


def evaluate(splits):
    spec,w,m_sha=memory()
    parent,pair_w,p_sha=consensus.inputs()
    recipe=repair.load_recipe()
    path=repair.receipt(TAG)
    plan=json.loads(path.read_text())
    assert plan["memory_freeze_sha256"]==m_sha and plan["pairwise_freeze_sha256"]==p_sha
    frozen=file_hash(path)
    report={"freeze_sha256":frozen,"splits":{}}
    for split in splits:
        bank,lex,base,u=phase.view(split,parent,pair_w,recipe)
        excluded=set(spec["training_cases"])|set(plan["calibration_cases"])
        if split!="select":
            excluded.update(plan["selection_cases"])
        assert not set(bank[0]) & excluded
        z=lexical(bank,split,spec,w)
        original=P.dense.paired_confusion(bank,base)
        result=report["splits"][split]={"cases":len(bank[0]),"baseline":row_summary(original.sum(0)),"arms":{}}
        for j,(arm,entry) in enumerate(plan["arms"].items()):
            p=base
            if entry["settings"]:
                patch=OUT/(TAG+f"-{j}.safetensors")
                assert file_hash(patch)==entry["sha256"]
                trained=load_file(str(patch))
                x=standardized(design(bank,lex,z,arm),trained)@trained["weight"]
                p=phase.edit(bank,base,u,x,entry["settings"][2:])
            c=P.dense.paired_confusion(bank,p)
            n=int(np.count_nonzero(p!=base))
            if split=="select":
                m,a,*_=measure(c.sum(0))
                assert [float(m),float(a),-n]==entry["selection"]
            result["arms"][arm]={"labels":row_summary(c.sum(0)),"paired":paired(c,original),"changed":n}
        save_json(P.RESULT/(TAG+".json"),report)
        print(split,{k:(r["labels"]["macro_f1"],r["labels"]["accuracy"],r["changed"])
                     for k,r in result["arms"].items()},flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("action",choices=("memory","fit","evaluate"))
    cli.add_argument("splits",nargs="*")
    args=cli.parse_args()
    if args.action=="memory":
        write_memory()
    elif args.action=="fit":
        fit()
    else:
        evaluate(args.splits or SPLITS)
