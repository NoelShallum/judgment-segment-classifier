"""Write a sparse seven-role memory for recurring sentences and templates."""
import argparse
from collections import Counter
import csv
from hashlib import blake2b,sha256
from itertools import groupby,product
import json
from pathlib import Path
import re

import numpy as np
from safetensors.numpy import load_file,save_file

import sentence_pairwise_consensus as consensus
import sentence_pairwise_repair as repair
import sentence_phase_readout_write as phase
from program_trial import TO_ID
from sentence_document_graph_write import OUT,cm,file_hash,paired,row_summary,save_json

P=repair.P
TAG="sentence-template-memory"
WEIGHTS=OUT/(TAG+".safetensors")
DATA=P.ROOT/"references/official/train.csv"
MODES=("exact","template","previous template","next template")
SPLITS=("select","confirm","holdout","phrase-prospect","test")
DIGITS=re.compile(r"\d+(?:[.,/]\d+)*")
SPACES=re.compile(r"\s+")
WORD=re.compile(r"\w+")


def canonical(text):
    return SPACES.sub(" ",text.casefold().strip())


def keys(texts):
    exact=[canonical(t) for t in texts]
    masked=[DIGITS.sub("#",t) for t in exact]
    def signature(value):
        return int.from_bytes(blake2b(value.encode(),digest_size=8,person=b"AttoFlow-role-v1").digest(),"big")
    return [[signature(t) for t in exact], [signature(t) for t in masked],
            [signature(a+"\t"+b) if a else 0 for a,b in zip([""]+masked,masked)],
            [signature(a+"\t"+b) if b else 0 for a,b in zip(masked,masked[1:]+[""])]]


def text_digest(texts):
    h=sha256()
    for raw in texts:
        b=canonical(raw).encode()
        h.update(len(b).to_bytes(4,"big"))
        h.update(b)
    return h.hexdigest()


def dataset(split):
    path=P.ROOT/"references/official"/(split+".csv")
    with path.open(newline="",encoding="utf-8-sig") as f:
        source=csv.DictReader(f)
        seen=set()
        for k,rows in groupby(source,key=lambda r:r["Index"]):
            assert k not in seen,(split,k)
            seen.add(k)
            items=[(r["Text"],TO_ID[r["Label"]]) for r in rows if r["Text"].strip()]
            if items:
                yield k,items


def write():
    sources={s:file_hash(DATA.with_name(s+".csv")) for s in ("train","val","test")}
    deny={text_digest([t for t,_ in rows]) for split in ("val","test") for _,rows in dataset(split)}
    counts=[{} for _ in MODES]
    rejected=[]
    role_counts=np.zeros(7,np.int64)
    cases=[]
    for key,rows in dataset("train"):
        texts,labels=zip(*rows)
        if text_digest(texts) in deny:
            rejected.append(key)
            continue
        cases.append(key)
        role_counts+=np.bincount(labels,minlength=7)
        allow=[sum(1 for _ in WORD.finditer(t))>=4 for t in texts]
        for j,values in enumerate(keys(texts)):
            unique={}
            for marker,y,ok in zip(values,labels,allow):
                if marker and ok:
                    unique[marker]=unique.get(marker,0)|(1<<y)
            table=counts[j]
            for h,roles in unique.items():
                if roles&(roles-1):
                    continue
                label=roles.bit_length()-1
                # One vote per judgment for any recurring key of known role.
                aggregate=table.get(h)
                if aggregate is None:
                    table[h]=label
                elif isinstance(aggregate,int):
                    v=[0]*7
                    v[aggregate]+=1
                    v[label]+=1
                    table[h]=v
                else:
                    aggregate[label]+=1
        if len(cases)%800==0:
            print("Tallied",len(cases),"training judgments",flush=True)
    tensors={}
    measures={}
    for i,(mode,table) in enumerate(zip(MODES,counts)):
        keep=sorted(k for k,v in table.items() if isinstance(v,list))
        count=np.asarray([table[k] for k in keep],dtype=np.uint16)
        assert not np.any(count.astype(np.uint64).sum(1)<2)
        tensors[f"keys_{i}"]=np.asarray(keep,dtype=np.uint64)
        tensors[f"counts_{i}"]=count
        measures[mode]=len(keep)
    save_file(tensors,str(WEIGHTS))
    check=load_file(str(WEIGHTS))
    assert all(np.array_equal(value,check[k]) for k,value in tensors.items())
    save_json(P.RESULT/(TAG+"-source.json"),{"csv_sha256":sources,"weights_sha256":file_hash(WEIGHTS),
              "train_cases":cases,"excluded_exact_duplicate_judgments":rejected,
              "train_role_counts":role_counts.tolist(),"stored_keys":measures,"gradient_steps":0})
    print("Wrote",measures,"excluded judgment duplicates",len(rejected),flush=True)


def read_weights(check_corpus=True):
    path=P.RESULT/(TAG+"-source.json")
    plan=json.loads(path.read_text())
    assert file_hash(WEIGHTS)==plan["weights_sha256"]
    if check_corpus:
        assert plan["csv_sha256"]=={s:file_hash(DATA.with_name(s+".csv")) for s in ("train","val","test")}
    return plan,load_file(str(WEIGHTS)),file_hash(path)


def lookup(texts,w,off):
    q=np.empty((len(texts),4),dtype=np.uint64)
    for a,b in zip(off,off[1:]):
        q[a:b]=np.array(keys(texts[a:b]),dtype=np.uint64).T
    matches=np.zeros((len(q),4,7),dtype=np.uint16)
    for i in range(4):
        indices=w[f"keys_{i}"]
        positions=np.searchsorted(indices,q[:,i])
        positions=np.minimum(positions,len(indices)-1)
        valid=indices[positions]==q[:,i]
        matches[valid,i]=w[f"counts_{i}"][positions[valid]]
    return matches


def get_view(split,parent,pair_w,recipe,w):
    bank,_,base,u=phase.view(split,parent,pair_w,recipe)
    docs=P.docs(split)
    np.testing.assert_array_equal(bank[0],[d["id"] for d in docs])
    # Check against the same supplied sentence boundaries as the main classifier.
    strings=[d["text"][a:b] for d in docs for a,b in d["reference_units"]]
    assert len(strings)==len(base)
    m=lookup(strings,w,bank[1])
    return bank,base,u,m


def propose(base,unary,m,source,minimum,accuracy,max_margin):
    q=m[:,source].astype(np.float32)
    totals=q.sum(1)
    labels=q.argmax(1)
    rows=np.arange(len(base))
    valid=((totals>=minimum)&(q[rows,labels]>=totals*accuracy)&
           (unary[rows,base]-unary[rows,labels]<=max_margin))
    return np.where(valid,labels,base)


def fit():
    plan,w,source=read_weights()
    parent,pair_w,p_sha=consensus.inputs()
    recipe=repair.load_recipe()
    bank,base,u,m=get_view("select",parent,pair_w,recipe,w)
    assert not set(bank[0])&set(plan["train_cases"])
    start=row_summary(cm(bank[2],base))
    best=(start["macro_f1"],start["accuracy"],0)
    settings=None
    grid=[]
    for cfg in product(range(4),(2,3,5,10),(.75,.86,.95,1.),(.18,.45,.85,1.6,4.)):
        p=propose(base,u,m,*cfg)
        r=row_summary(cm(bank[2],p))
        entry=(r["macro_f1"],r["accuracy"],-int(np.count_nonzero(p!=base)))
        grid.append([cfg,entry])
        if entry>best and r["accuracy"]>=start["accuracy"]-.001:
            best,settings=entry,cfg
    save_json(repair.receipt(TAG),{"source_sha256":source,"pairwise_freeze_sha256":p_sha,"selection_cases":bank[0].tolist(),
             "choice":settings,"selected_metrics":best,"gradient_steps":0,"grid":grid})
    print("Frozen",settings,best,flush=True)


def evaluate(splits,with_bert=False):
    source,w,digest=read_weights()
    parent,pair_w,p_sha=consensus.inputs()
    recipe=repair.load_recipe()
    plan=repair.receipt(TAG)
    frozen=json.loads(plan.read_text())
    assert frozen["source_sha256"]==digest and frozen["pairwise_freeze_sha256"]==p_sha
    answer={"freeze_sha256":file_hash(plan),"splits":{}}
    if with_bert:
        import compare_broad_edits_to_bert as broad
        path=repair.receipt(broad.overlay.TAG)
        cfg=json.loads(path.read_text())
        assert cfg["sources"][repair.TAG]==p_sha
        rule=cfg["arms"]["BERT only"]["choice"]
        bert=broad.B.read(broad.B.TAG+"-frozen")
        assert cfg["sources"][broad.B.TAG]==file_hash(repair.receipt(broad.B.TAG))
        answer["bert_overlay_sha256"]=file_hash(path)
    for split in splits:
        bank,base,u,m=get_view(split,parent,pair_w,recipe,w)
        assert not set(bank[0])&set(source["train_cases"])
        if split!="select":
            assert not set(bank[0])&set(frozen["selection_cases"])
        p=propose(base,u,m,*frozen["choice"]) if frozen["choice"] else base
        if with_bert:
            original=broad.bert_reference(split,bank,base,bert,rule)
            p=np.where(original!=base,original,p)
            base=original
        before,after=(P.dense.paired_confusion(bank,v) for v in (base,p))
        changed=p!=base
        r={"cases":len(bank[0]),"baseline":row_summary(before.sum(0)),"candidate":row_summary(after.sum(0)),
           "paired":paired(after,before),"changed":int(changed.sum()),
           "fixed":int((changed&(p==bank[2])).sum()),"broken":int((changed&(base==bank[2])).sum()),
           "lookup_hits":dict(zip(MODES,(m.sum(2)>0).sum(0).tolist()))}
        if split=="select" and not with_bert:
            assert [r["candidate"]["macro_f1"],r["candidate"]["accuracy"],-r["changed"]]==frozen["selected_metrics"]
        answer["splits"][split]=r
        save_json(P.RESULT/(TAG+("-bert" if with_bert else "")+".json"),answer)
        print(split,r["candidate"]["macro_f1"],r["changed"],r["fixed"],r["broken"],r["lookup_hits"],flush=True)


def predict(text,raw,bert_device=None):
    import predict_pairwise_judgment as runtime
    import sentence_phrase_sequence as seq

    _,w,digest=read_weights(check_corpus=False)
    plan=json.loads(repair.receipt(TAG).read_text())
    assert plan["source_sha256"]==digest
    assert plan["pairwise_freeze_sha256"]==file_hash(repair.receipt(repair.TAG))
    sentences=[text[a:b] for a,b in seq.raw_document(text)["units"]] if raw else text
    initial=runtime.predictor("direct")
    review=None
    if bert_device:
        from contextlib import redirect_stdout
        import sys
        import compare_broad_edits_to_bert as broad
        from openvino_worker import embed_ids,get_engine

        B=broad.B
        policy=json.loads(repair.receipt(broad.overlay.TAG).read_text())
        cfg=B.read(B.TAG+"-frozen")
        assert policy["sources"][B.TAG]==file_hash(repair.receipt(B.TAG))
        assert policy["sources"][repair.TAG]==plan["pairwise_freeze_sha256"]
        head=OUT/(B.TAG+"-2.safetensors")
        assert cfg["checkpoints"][broad.overlay.BERT]==file_hash(head)
        assert cfg["document_weights_sha256"]==file_hash(B.PARENT)
        assert cfg["backbone_weights_sha256"]==file_hash(B.ORIGINAL)
        item=B.document("input",[(j,s,0) for j,s in enumerate(sentences)])
        model=B.HierModel(B.DONOR)
        input_strings=[item["text"][a:b] for a,b in item["reference_units"]]
        states=model.contextualize(model.words(input_strings))
        item=B.case_view(item,states,model,load_file(str(B.PARENT)),json.loads((B.EDGE/"vocab.json").read_text()))
        if len(item[3]):
            with redirect_stdout(sys.stderr):
                engine,tokenizer=get_engine(bert_device)
            encoded=embed_ids(engine,B.pack_ids(tokenizer,[item],B.MODES[1]),max_length=B.LIMIT)
            w0=load_file(str(head))
            mode=cfg["frozen_modes"][broad.overlay.BERT][0]
            prob=B.response(B.design([item],encoded,mode,w0),w0)
            gain,margin,veto=policy["arms"]["BERT only"]["choice"]
            review=(broad.overlay.edit,item[3],item[5],prob,gain,margin,item[2][item[3]] if veto else None)

    def edit(bank,lex,unary,base):
        direct=initial(bank,lex,unary,base)
        candidate=direct
        if plan["choice"] is not None:
            match=lookup(sentences,w,bank[1])
            candidate=propose(direct,unary,match,*plan["choice"])
        if review:
            apply,*args=review
            expert=apply(direct,*args)
            candidate=np.where(expert!=direct,expert,candidate)
        return candidate

    return seq.run_input(text,raw,"spans",edit)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("write","fit","evaluate","compose","predict"))
    p.add_argument("splits",nargs="*")
    p.add_argument("--input",type=Path)
    p.add_argument("--raw",action="store_true")
    p.add_argument("--bert-device",choices=("CPU","GPU"))
    a=p.parse_args()
    if a.action=="write":
        write()
    elif a.action=="fit":
        fit()
    elif a.action=="compose":
        evaluate(a.splits or ("select","confirm","holdout","test"),with_bert=True)
    elif a.action=="predict":
        if a.input is None:
            p.error("predict requires --input")
        text=a.input.read_bytes().decode("utf-8")
        print(json.dumps(predict(text if a.raw else json.loads(text),a.raw,a.bert_device),ensure_ascii=False))
    else:
        evaluate(a.splits or SPLITS)
