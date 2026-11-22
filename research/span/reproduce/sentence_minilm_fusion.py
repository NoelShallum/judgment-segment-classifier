"""Write a second-encoder correction on low-confidence supplied LegalSeg sentences."""
import argparse
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json

import numpy as np
import onnxruntime as ort
from safetensors.numpy import load_file,save_file
from transformers import AutoTokenizer

from sentence_hidden_edit import (
    DONOR,EDGE,OUT,ROOT,HierModel,aligned_embeddings,cm,document,edit_documents,
    embeddings,file_hash,gold_cases,independent_validation,prepare_score,row_summary,
    sample_whole,save_json,torch,transition_scores,
)
from predict_role_edge import decode_profile
from sentence_role_retention import measure,paired
from sentence_uncertainty_runtime import apply_uncertainty,probability,uncertainty_evidence

REVISION="1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
GRAPH=ROOT/"runtime/minilm-sentence-fusion/model-quint8-avx2.onnx"
RATES=(.10,.20,.30)
MAIN=OUT/"known-sentence-label-change.safetensors"
TOKEN_LIMIT=96
SOURCE=ROOT/"references/minilm/model.safetensors"


@dataclass
class RowView:
    case: str
    texts: list
    gold: np.ndarray
    base: np.ndarray
    prior: np.ndarray
    structural: np.ndarray
    query: np.ndarray
    semantic: np.ndarray|None=None


class MiniLM:
    def __init__(self):
        self.tok=AutoTokenizer.from_pretrained(ROOT/"references/minilm",local_files_only=True)
        opts=ort.SessionOptions()
        opts.intra_op_num_threads=6
        opts.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(GRAPH),sess_options=opts,providers=["CPUExecutionProvider"])
        self.inputs={a.name for a in self.session.get_inputs()}
        self.hash=file_hash(GRAPH)
        self.cache_key=self.hash[:12]

    def encode(self,texts):
        ids=self.tok(texts,add_special_tokens=False,verbose=False)["input_ids"]
        ids=[v if len(v)<=TOKEN_LIMIT-2 else v[:72]+v[-22:] for v in ids]
        rows=np.empty((len(ids),384),dtype=np.float32)
        order=sorted(range(len(ids)),key=lambda i:len(ids[i]))
        cls,sep=self.tok.cls_token_id,self.tok.sep_token_id
        for j in range(0,len(order),48):
            batch=order[j:j+48]
            arm=self.tok.pad([{"input_ids":[cls,*ids[i],sep]} for i in batch],
                             padding=True,return_tensors="np")
            values={name:arm[name] if name in arm else np.zeros_like(arm["input_ids"]) for name in self.inputs}
            z=self.session.run(["last_hidden_state"],values)[0]
            mask=values["attention_mask"][:,:,None]
            avg=(z*mask).sum(1)/mask.sum(1)
            rows[batch]=(avg/np.maximum(np.linalg.norm(avg,axis=1,keepdims=True),1.e-8)).astype(np.float32)
        return rows

    def fill(self,rows,tag,gate):
        cache=ROOT/"cache"/f"minilm-sentence-fusion-{self.cache_key}-{TOKEN_LIMIT}"
        cache.mkdir(exist_ok=True)
        step=36
        for start in range(0,len(rows),step):
            chunk=rows[start:start+step]
            dest=[]
            for row in chunk:
                inputs=[row.texts[i] for i in row.query]
                path=cache/f"{row.case}-{gate:.2f}.npz"
                dest.append((row,path,inputs))
            layout=sha256(json.dumps([(r.case,r.query.tolist(),x) for r,_,x in dest],
                             ensure_ascii=False).encode()).hexdigest()
            complete=True
            for row,path,_ in dest:
                if not path.is_file():
                    complete=False
                    break
                with np.load(path,allow_pickle=False) as r:
                    if ("layout" not in r or r["layout"].item()!=layout
                            or not np.array_equal(r["query"],row.query)):
                        complete=False
                        break
                    row.semantic=r["x"]
            if complete:
                continue
            # Dynamic quantization depends on the other examples in the batch.
            # Recreate the entire original block after a partial write or cache miss.
            inputs=[s for _,_,texts in dest for s in texts]
            response=self.encode(inputs)
            i=0
            for row,path,texts in dest:
                end=i+len(texts)
                row.semantic=response[i:end].copy()
                tmp=path.with_suffix(".part")
                with tmp.open("wb") as f:
                    np.savez(f,query=row.query,x=row.semantic,layout=layout)
                tmp.replace(path)
                i=end
            print("MiniLM",tag,min(len(rows),start+step),"/",len(rows),len(inputs),"new sentences",flush=True)


class TokenTable(MiniLM):
    def __init__(self):
        self.tok=AutoTokenizer.from_pretrained(ROOT/"references/minilm",local_files_only=True)
        values=load_file(str(SOURCE))
        self.table=values["embeddings.word_embeddings.weight"]
        self.hash=file_hash(SOURCE)
        self.idf=None

    def calibrate(self,texts):
        ids=self.tok(texts,add_special_tokens=False,verbose=False)["input_ids"]
        df=np.bincount([j for row in ids for j in set(row)],minlength=len(self.table))
        self.idf=(1+np.log((len(ids)+1)/(df+1))).astype(np.float32)
        self.cache_key=self.hash[:12]+"-"+sha256(self.idf.tobytes()).hexdigest()[:10]

    def encode(self,texts):
        ids=self.tok(texts,add_special_tokens=False,verbose=False)["input_ids"]
        rows=[]
        for seq in ids:
            if not seq:
                seq=[self.tok.unk_token_id]
            if len(seq)>TOKEN_LIMIT-2:
                seq=seq[:72]+seq[-22:]
            values=self.table[seq]
            v=(values*self.idf[seq][:,None]).sum(0)
            rows.append(v/max(float(np.linalg.norm(v)),1.e-8))
        return np.stack(rows)


def case_view(d,state,model,parent,vocab,rate):
    supplied={**d,"units":d["reference_units"]}
    text=d["text"]
    parts=[text[a:b].strip() for a,b in supplied["units"]]
    item=prepare_score(supplied,model.logits(state),{"exact":parent})
    x=item.logits["exact"]
    edge=transition_scores(text,supplied["units"],vocab,parent)
    gain=float(parent["edge.gain"])
    selected=decode_profile(x,edge,parent,gain)
    query,prior,structural=uncertainty_evidence(x,edge,parent,gain,selected,[len(t) for t in parts],rate)
    return RowView(str(d["id"]),parts,np.asarray(d["reference_roles"]),selected,prior,structural,query)


def features(row,arm):
    base=row.structural[row.query]
    if arm=="semantic":
        return row.semantic
    if arm=="combined":
        return np.column_stack((base,row.semantic))
    return base


def fit(rows):
    gold=np.concatenate([r.gold[r.query] for r in rows])
    prior=np.bincount(gold,minlength=7)/len(gold)
    label=np.eye(7)[gold]
    for arm in ("semantic","combined","structural"):
        x=np.concatenate([features(r,arm) for r in rows])
        center=x.mean(0).astype("f4")
        scale=(1/np.maximum(x.std(0),.025)).astype("f4")
        z=np.column_stack(((x-center)*scale,np.ones(len(x)))).astype(np.float64)
        pen=np.r_[np.ones(x.shape[1]),.05]
        for rarity in (0.,.28):
            w=prior[gold]**-rarity
            gram=z.T@(z*w[:,None])
            projection=np.linalg.solve(gram+np.diag(pen*.12*gram.trace()/len(gram)),z.T@(label*w[:,None]))
            yield (arm,rarity),{"center":center,"scale":scale,"weight":projection[:-1].astype("f4"),
                                 "bias":projection[-1].astype("f4")}


def predictions(row,arm,w):
    return probability(features(row,arm),w)


def apply(row,p,rate,mix,margin):
    return apply_uncertainty(row.base,row.query,row.prior,p,rate,mix,margin)


def run(rows,p,rate,mix,gap):
    labels=[apply(r,a,rate,mix,gap) for r,a in zip(rows,p)]
    return np.stack([cm(r.gold,y) for r,y in zip(rows,labels)]),sum(int((y!=r.base).sum()) for r,y in zip(rows,labels))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-table",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(2)
    encoder=TokenTable() if args.token_table else MiniLM()
    stem="sentence-minilm-token-table" if args.token_table else "sentence-minilm-fusion"
    model=HierModel(DONOR)
    parent=load_file(str(MAIN))
    vocab=json.loads((EDGE/"vocab.json").read_text())
    groups=sample_whole()
    train=edit_documents()
    train_vectors=embeddings(model,train,"aligned-source-train")
    rows=[case_view(d,s,model,parent,vocab,.30) for d,s in zip(train,train_vectors)]
    print("Training query sentences",sum(len(r.query) for r in rows),flush=True)
    if args.token_table:
        encoder.calibrate([r.texts[i] for r in rows for i in r.query])
    encoder.fill(rows,"train",.30)
    w=dict(fit(rows))
    print("Adapted MiniLM correction fit",len(w),flush=True)
    del rows,train_vectors
    def prepare(docs,split,rate,use_semantic=True):
        rows=[case_view(d,s,model,parent,vocab,rate) for d,s in zip(docs,aligned_embeddings(model,docs,split))]
        print("Hierarchy prepared",split,len(rows),sum(len(r.query) for r in rows),flush=True)
        if use_semantic:
            encoder.fill(rows,split,rate)
        return rows

    selected=prepare(groups["select"],"select",.30)
    base=measure(sum(cm(r.gold,r.base) for r in selected)).tolist()
    grid=[];champion={};choices={}
    for (arm,rarity),weights in w.items():
        scored=[predictions(r,arm,weights) for r in selected]
        for rate,mix,margin in product(RATES,(.35,.60,.85,1.),(.02,.07,.15,.23)):
            a,changed=run(selected,scored,rate,mix,margin)
            q=measure(a.sum(0)).tolist()
            spec=(arm,rarity,rate,mix,margin)
            grid.append([*spec,*q,changed])
            if changed>0 and q[1]>=base[1]-.006 and q[0]>champion.get(arm,(-1.,))[0]:
                champion[arm]=q
                choices[arm]=spec
    semantic=max((x for x in choices if x!="structural"),key=lambda x:champion[x][0],default=None)
    if semantic is None:
        feasible=[v for v in grid if v[0]!="structural" and v[-1]]
        assert feasible
        best=max(feasible,key=lambda v:v[5])
        semantic=best[0]
        choices[semantic]=tuple(best[:5])
        champion[semantic]=best[5:9]
    winners={"semantic_fusion":choices[semantic]}
    if "structural" in choices:
        winners["hierarchy_only"]=choices["structural"]
    file=OUT/(("known-"+stem)+".safetensors")
    payload={}
    for name,value in winners.items():
        payload.update({name+"."+k:v for k,v in w[value[:2]].items()})
        payload[name+".settings"]=np.asarray(value[2:])
    if args.token_table:
        payload["token_table.idf"]=encoder.idf
    save_file(payload,str(file))
    physical=load_file(str(file))
    assert all(np.array_equal(v,physical[k]) for k,v in payload.items())
    frozen=ROOT/"results"/(stem+"-frozen.json")
    save_json(frozen,{"choices":winners,"selection_scores":{k:champion[v[0]] for k,v in winners.items()},
           "preceding":base,"grid":grid,"graph_sha256":encoder.hash,"weights_sha256":file_hash(file),
           "minilm_revision":REVISION,"sentence_token_limit":TOKEN_LIMIT,"baseline_sha256":file_hash(MAIN),
           "backend":"idf-weighted input embeddings" if args.token_table else "quantized transformer",
           "training_cases":[d["id"] for d in train],"selection_cases":[d["id"] for d in groups["select"]]})
    print("MiniLM choices",winners,{k:champion[v[0]] for k,v in winners.items()},base,flush=True)
    used={d["id"] for batch in groups.values() for d in batch}
    omitted=set(json.loads((EDGE/"overlap.json").read_text())["independent_exclusions_for_text_disjoint_check"])
    groups["independent"]=[d for d in independent_validation(used) if d["id"] not in omitted]
    method="its frozen input-token embeddings with IDF weights" if args.token_table else "a frozen, quantized MiniLM transformer"
    output={"freeze_sha256":file_hash(frozen),"choices":winners,"splits":{},
        "protocol":f"Score potentially incorrect supplied-sentence labels using {method}. The risk rank uses only the two frozen hierarchical CRF posteriors and the released predicted role. Only the selected highest-risk fraction of each judgment can change at inference. Fit closed-form role probabilities and include a matched hierarchy-only ablation."}
    rate=max(v[2] for v in winners.values())
    for split in ("select","confirm","independent","test"):
        if split=="test":
            groups[split]=[document(k,r) for k,r in gold_cases().items()]
        subset=selected if split=="select" else prepare(groups[split],split,rate)
        b=np.stack([cm(r.gold,r.base) for r in subset])
        if split=="test":
            expect=json.loads((ROOT/"results/sentence-label-change-qa.json").read_text())["scores"]
            assert row_summary(b.sum(0))==expect
        q={"cases":len(b),"preceding":row_summary(b.sum(0)),
           "sentences_encoded":int(sum(len(r.query) for r in subset)),"profiles":{}}
        for name,choice in winners.items():
            params={k.split(".",1)[1]:v for k,v in physical.items() if k.startswith(name+".")}
            a,n=run(subset,[predictions(r,choice[0],params) for r in subset],*params["settings"])
            if split=="select":
                np.testing.assert_allclose(measure(a.sum(0)),champion[choice[0]],rtol=0,atol=1e-12)
            q["profiles"][name]={"scores":row_summary(a.sum(0)),"gap_ci95":paired(a,b),"changed":int(n)}
        output["splits"][split]=q
        save_json(ROOT/"results"/(stem+"-edit.json"),output)
        print(split,{k:v["scores"]["macro_f1"] for k,v in q["profiles"].items()},q["preceding"]["macro_f1"],flush=True)


if __name__=="__main__":
    main()
