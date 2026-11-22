"""Select and test authored sequence weights on raw, unmarked passages from complete seven-role LegalSeg judgments."""
import argparse
from collections import Counter,defaultdict
import csv,hashlib,json,random,re,sqlite3
from pathlib import Path
import numpy as np
from safetensors.numpy import save_file
from encoder import TOK,components
from mixed_encoder_trial import CONTEXT
from program_trial import TO_ID,fold,save_json
from span_model import authored_transitions,context_inputs,decode,merge,units
from weight_edit import LABELS,ORIGINAL,ROOT,file_hash,tensor
from windowing import token_windows

SEQ=ROOT/"seven-spans"
MODELS={"a":{"source":ORIGINAL,"head":ROOT/"checkpoints/associative-direct/model.safetensors"},
        "b":{"source":CONTEXT,"head":ROOT/"checkpoints/associative-context/model.safetensors"}}
METHODS={"A isolated":("a",0),"B isolated":("b",0),"B preceding two":("b",2)}
def document(doc,rows):
    texts=[];gold=[];offset=0
    for index,text,role in rows:
        t=text.strip()
        if not t:continue
        if texts:offset+=1
        gold.append((offset,role));texts.append(t);offset+=len(t)
    text=" ".join(texts);known=[(a,b) for a,b in zip([x[0] for x in gold],[x[0] for x in gold][1:]+[len(text)])]
    return {"id":doc,"text":text,"reference_units":known,"reference_roles":[x[1] for x in gold],
            "units":units(text),"gold_spans":merge(known,[x[1] for x in gold],LABELS)}

def sets():
    result={}; exclusion=set()
    for name in ("gemini-seven-pilot","gemini-confirmation","gemini-seven-paragraphs","gemini-seven-joined"):
        f=ROOT/name/"approved.json"
        if f.exists():exclusion|={r["doc"] for r in json.loads(f.read_text())}
    p=SEQ/"cloud.json"
    if p.exists():exclusion|={d["doc"] for d in json.loads(p.read_text())}
    for split,code,n,seed in (("val","select",10,28591),("val","confirm",10,72587),("test","test",20,49118)):
        docs=defaultdict(list)
        with (ROOT/"references/official"/(split+".csv")).open(encoding="utf-8-sig",newline="") as stream:
            for i,r in enumerate(csv.DictReader(stream)):
                d=r["Index"]
                if split=="val" and (fold(d)==0)!=(code=="confirm"):continue
                if split=="test" and d in exclusion:continue
                docs[d].append((i,r["Text"] or "",TO_ID[r["Label"]]))
        eligible=sorted(d for d,rows in docs.items() if 25<=len(rows)<=100)
        ids=random.Random(seed).sample(eligible,n)
        result[code]=[document(d,docs[d]) for d in ids]
    if p.exists():result["cloud"]=[document(b["doc"],[(r["row"],r["text"],LABELS.index(r["gold"])) for r in b["rows"]]) for b in json.loads(p.read_text())]
    return result

class EncoderCache:
    def __init__(self,source,backend):
        self.source=source;self.backend=backend;self.engine=None;self.sum=file_hash(source)
        self.db=sqlite3.connect(ROOT/"cache"/("span-window-fp32-"+self.sum[:16]+".sqlite3"),timeout=90)
        self.db.execute("CREATE TABLE IF NOT EXISTS vectors (sha TEXT PRIMARY KEY, value BLOB NOT NULL)")
    def encode(self,texts):
        keys=[hashlib.sha256(t.encode()).hexdigest() for t in texts];found={};missing={}
        for k,t in zip(keys,texts):
            if k in found or k in missing:continue
            rec=self.db.execute("SELECT value FROM vectors WHERE sha=?",(k,)).fetchone()
            if rec:found[k]=np.frombuffer(rec[0],dtype="<f4")
            else:missing[k]=t
        if missing:
            if self.engine is None:
                if self.backend=="intel":
                    from openvino_worker import ART,get_engine
                    ir=ART if self.source==ORIGINAL else ROOT/"runtime"/("encoder-"+self.sum[:15])
                    self.engine,self.tok=get_engine("GPU",self.source,ir)
                else:self.engine,self.tok,self.device=components(checkpoint=self.source)
            pairs=list(missing.items())
            for start in range(0,len(pairs),192):
                group=pairs[start:start+192]
                seqs=self.tok([v for _,v in group],add_special_tokens=False,return_attention_mask=False,return_token_type_ids=False,verbose=False)["input_ids"]
                flat=[(j,ids,w) for j,seq in enumerate(seqs) for ids,w in token_windows(self.tok,seq,512)]
                if self.backend=="intel":
                    from openvino_worker import embed_ids
                    vec=embed_ids(self.engine,[i for _,i,_ in flat])
                else:
                    import torch
                    vec=np.empty((len(flat),768),dtype=np.float32)
                    order=sorted(range(len(flat)),key=lambda j:len(flat[j][1]))
                    with torch.inference_mode():
                        for off in range(0,len(order),4):
                            indx=order[off:off+4]
                            pack=self.tok.pad([{"input_ids":flat[j][1]} for j in indx],padding=True,return_tensors="pt").to(self.device)
                            vec[indx]=self.engine.bert(**pack).pooler_output.detach().cpu().numpy()
                total=np.zeros((len(group),768),dtype=np.float64);mass=np.zeros(len(group))
                for v,(j,_,weight) in zip(vec,flat):total[j]+=v.astype(float)*weight;mass[j]+=weight
                total=(total/mass[:,None]).astype("<f4")
                self.db.executemany("INSERT OR REPLACE INTO vectors (sha,value) VALUES (?,?)",[(k,v.tobytes()) for (k,_),v in zip(group,total)])
                self.db.commit()
                for (k,_),v in zip(group,total):found[k]=v
        return np.stack([found[k] for k in keys])
    def close(self):self.db.close()

def matches(a,b):
    a=set(a);b=set(b);return [len(a&b),len(a),len(b)]
def counts(doc,offsets,y):
    predicted=merge(offsets,y,LABELS);gold=doc["gold_spans"]
    assert predicted[0]["start"]==0 and predicted[-1]["end"]==len(doc["text"])
    toks=[m.start() for m in re.finditer(r"\S+",doc["text"])]
    boundaries=lambda spans:[z["start"] for z in spans[1:]]
    starts=np.array(toks)
    def labels(spans):
        indexes=np.searchsorted([s["start"] for s in spans],starts,side="right")-1
        roles=np.asarray([LABELS.index(z["label"]) for z in spans]);return roles[indexes]
    a=labels(predicted);b=labels(gold)
    exact=matches([(z["start"],z["end"],z["label"]) for z in predicted],[(z["start"],z["end"],z["label"]) for z in gold])
    bp=matches(boundaries(predicted),boundaries(gold))
    return np.concatenate((np.bincount(b*7+a,minlength=49),np.asarray(exact+bp)))
def summarize(c):
    cm=c[:49].reshape(7,7);tp=cm.diagonal();den=cm.sum(1)+cm.sum(0)
    def pr(d):
        h,p,r=d;return {"matched":int(h),"predicted":int(p),"gold":int(r),"precision":float(h/p) if p else 0.,"recall":float(h/r) if r else 0.,"f1":float(2*h/(p+r)) if p+r else 1.}
    return {"tokens":int(cm.sum()),"token_accuracy":float(tp.sum()/cm.sum()),"token_macro_f1":float(np.divide(2*tp,den,out=np.zeros(7),where=den>0).mean()),
            "exact_labeled_spans":pr(c[49:52]),"exact_role_boundaries":pr(c[52:55]),"token_confusion":cm.tolist()}
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--backend",choices=["intel","torch"],default="torch");args=ap.parse_args()
    groups=sets();all_docs=[d for group in groups.values() for d in group]
    methods={};oracles={}
    for parent,details in MODELS.items():
        engine=EncoderCache(details["source"],args.backend)
        heads=(tensor(details["head"],"classifier.weight"),tensor(details["head"],"classifier.bias"))
        for name,(p,left) in METHODS.items():
            if p!=parent:continue
            for kind,collection in (("units",methods),("reference_units",oracles)):
                inp=[v for d in all_docs for v in context_inputs(d["text"],d[kind],left)]
                print("Encoding",name,kind,len(inp),"local units",flush=True)
                xx=engine.encode(inp);out=xx@heads[0].T+heads[1]
                blocks={};i=0
                for d in all_docs:
                    blocks[d["id"]]=out[i:i+len(d[kind])];i+=len(d[kind])
                collection[name]=blocks
        engine.close()
    def score(group,name,s,o,neighbor,oracle=False,detail=False):
        tr=authored_transitions(s,o);ker=np.array([neighbor,1-2*neighbor,neighbor],np.float32)
        dcounts=[]
        for d in groups[group]:
            xx=(oracles if oracle else methods)[name][d["id"]];y=decode(xx,tr,ker)
            dcounts.append(counts(d,d["reference_units" if oracle else "units"],y))
        summary=summarize(np.sum(dcounts,axis=0))
        return (summary,np.stack(dcounts)) if detail else summary
    choices=[]
    for name in METHODS:
        for sticky in (0,.5,1.5,3):
            for order in (0,.375):
                for n in (0,.15):
                    q=score("select",name,sticky,order,n)
                    choices.append((q["exact_labeled_spans"]["f1"],q["token_accuracy"],name,sticky,order,n,q))
    # A tie goes to the lower-compute and less-edited candidate in enumeration order.
    best=max(choices,key=lambda z:z[:2]);_,_,name,sticky,order,neighbor,_=best
    tr=authored_transitions(sticky,order);ker=np.asarray([neighbor,1-2*neighbor,neighbor],dtype=np.float32)
    dest=ROOT/"checkpoints/document-decoder";dest.mkdir(exist_ok=True)
    save_file({"transition":tr,"neighbor_kernel":ker},str(dest/"sequence.safetensors"))
    program={"method":name,"context_left":METHODS[name][1],"parent":METHODS[name][0],"source_head_sha256":file_hash(MODELS[METHODS[name][0]]["head"]),
             "sticky":sticky,"order":order,"neighbor":neighbor,"tensor_sha256":file_hash(dest/"sequence.safetensors"),"labels":LABELS,
             "selection":"Maximize exact labeled-span F1 on 10 whole validation judgments, then token accuracy. No gradient or teacher outputs."}
    save_json(dest/"sequence.json",program)
    result={}
    for g,docs in groups.items():
        main,per=score(g,name,sticky,order,neighbor,detail=True)
        base,bper=score(g,"A isolated",0,0,0,detail=True)
        rng=np.random.default_rng(11306);draw=[]
        for _ in range(1000):
            sel=rng.integers(len(docs),size=len(docs))
            ca,cb=summarize(per[sel].sum(0)),summarize(bper[sel].sum(0))
            draw.append([ca["token_accuracy"]-cb["token_accuracy"],ca["exact_labeled_spans"]["f1"]-cb["exact_labeled_spans"]["f1"]])
        result[g]={"cases":len(docs),"source_rows":sum(len(d["reference_units"]) for d in docs),"detected_units":sum(len(d["units"]) for d in docs),
            "case_ids":[d["id"] for d in docs],"independent_a":base,"programmed":main,
            "all_base_input_methods":{v:score(g,v,0,0,0) for v in METHODS},
            "programmed_with_reference_row_boundaries":score(g,name,sticky,order,neighbor,oracle=True),
            "paired_accuracy_and_exact_span_f1_gain_95pct":np.quantile(draw,[.025,.975],axis=0).T.tolist()}
    if "cloud" in groups:
        rows=[]
        for d in groups["cloud"]:
            xx=methods[name][d["id"]];y=decode(xx,tr,ker)
            rows.append({"doc":d["id"],"unit_labels":[LABELS[i] for i in y],"spans":merge(d["units"],y,LABELS)})
        save_json(SEQ/"local-cloud.json",rows)
    save_json(ROOT/"results/document-segmentation.json",{"selection":program,"segments_use_half_open_unicode_offsets":True,
        "split":"Complete 25–100-row LegalSeg cases, reconstructed with one ordinary space between nonblank source rows. Original PDF layout is unavailable. The raw decoder does not receive gold row boundaries.",
        "candidate_count":len(choices),"selection_candidates":[[v[2],v[3],v[4],v[5],v[0],v[1]] for v in choices],"groups":result})
    print("Selected",program["method"],sticky,order,neighbor,"official test",result["test"]["programmed"]["exact_labeled_spans"],flush=True)
if __name__=="__main__":main()
