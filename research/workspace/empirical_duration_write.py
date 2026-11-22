"""Write empirical role-run length corrections into an augmented-state CRF."""
import numpy as np
from safetensors.numpy import load_file,save_file
import torch

from discourse_span_edit import text_features
from hier_full import CACHE,gold_cases
from hier_model import CRF,HierModel,path_labels
from hier_trial import DONOR,OUT
from preserved_emitter_edit import edit_documents,embeddings
from program_trial import save_json
from seven_paths import ROOT
from span_eval import counts,document,summarize
from transition_span_edit import sample_whole
from weight_edit import file_hash

AGES=32
BUCKETS=np.array([1,2,4,8,16,32])

def length_bins(labels):
    x=np.asarray(labels,dtype=int)
    if not len(x):return np.zeros((7,len(BUCKETS)),dtype=int)
    starts=np.r_[0,np.flatnonzero(x[:-1]!=x[1:])+1]
    size=np.diff(np.r_[starts,len(x)])
    b=np.searchsorted(BUCKETS,size,side="right")-1
    return np.bincount(x[starts]*len(BUCKETS)+b,minlength=7*len(BUCKETS)).reshape(7,len(BUCKETS))

def estimate(model,parents):
    rec=edit_documents()
    features=embeddings(model,rec,"aligned-source-train")
    gold=np.zeros((7,len(BUCKETS)),dtype=int)
    predicted={k:gold.copy() for k in parents}
    for d,x in zip(rec,features):
        gold+=length_bins(d["reference_roles"])
        raw=model.logits(x)
        for name,v in parents.items():predicted[name]+=length_bins(path_labels(raw,v[CRF]))
    def chance(x):
        y=x.astype(np.float32)+8.
        return y/y.sum(1,keepdims=True)
    curves={}
    for name,r in predicted.items():
        local=np.clip(np.log(chance(gold)/chance(r)),-2,2)
        pooled=np.tile(np.clip(np.log(chance(gold.sum(0,keepdims=True))/chance(r.sum(0,keepdims=True))),-2,2),(7,1))
        positions=np.searchsorted(BUCKETS,np.arange(1,AGES+1),side="right")-1
        curves[name]={"role_conditioned":local[:,positions],"pooled":pooled[:,positions]}
    return curves, gold,predicted

def decode(logits,trans,offset):
    if not np.any(offset):return path_labels(logits,trans)
    a=np.asarray(logits,np.float32);days=offset.shape[1]
    graph=trans[3:,3:].copy();self=np.diag(graph).copy()
    np.fill_diagonal(graph,-np.inf)
    s=np.full((7,days),-np.inf,dtype=np.float32);s[:,0]=a[0]+trans[1,3:]
    roles=np.arange(7);stay=np.arange(days,dtype=np.int16)
    path=np.full((len(a),7,days),-1,dtype=np.int16)
    ages_base=roles*days
    for i,e in enumerate(a[1:],1):
        end=s+offset
        previous=end.argmax(1)
        switches=end[roles,previous][:,None]+graph
        route=switches.argmax(0)
        nxt=np.empty_like(s)
        nxt[:,0]=switches[route,roles]+e
        path[i,:,0]=route*days+previous[route]
        sea=np.where(s[:,-2]>=s[:,-1],days-2,days-1)
        nxt[:,1:]=s[:,:-1]+self[:,None]+e[:,None]
        nxt[:,-1]=s[roles,sea]+self+e
        path[i,:,1:]=ages_base[:,None]+stay[None,:-1]
        path[i,:,-1]=ages_base+sea
        s=nxt
    state=int(np.argmax(s+offset+trans[3:,2,None]))
    labels=np.empty(len(a),dtype=np.int16)
    for i in range(len(a)-1,-1,-1):
        label,age=divmod(state,days);labels[i]=label;state=int(path[i,label,age])
    return labels

def pair(a,b):
    rng=np.random.default_rng(13030)
    rows=[]
    for ix in rng.integers(len(a),size=(1600,len(a))):
        q=summarize(a[ix].sum(0));v=summarize(b[ix].sum(0))
        rows.append([q["exact_labeled_spans"]["f1"]-v["exact_labeled_spans"]["f1"],q["token_accuracy"]-v["token_accuracy"]])
    return np.quantile(rows,[.025,.975],axis=0).T.tolist()

def main():
    torch.set_num_threads(2)
    m=HierModel(DONOR)
    parents={k:load_file(str(OUT/f"whole-discourse-{k}.safetensors"))
             for k in ("exact","balanced")}
    curves,gold,source=estimate(m,parents)
    print("Froze training-only empirical duration estimates",int(gold.sum()),flush=True)
    def stage(doc,raw):
        cue,loc=text_features(doc)
        return doc,{k:raw+cue@v["cue_head"]+loc@v["position_head"] for k,v in parents.items()}
    docs=sample_whole();banks={}
    for category,items in docs.items():
        states=embeddings(m,items,"raw-boundary-write-"+category)
        banks[category]=[stage(d,m.logits(f)) for d,f in zip(items,states)]
    def measure(records,name,offset):
        return np.stack([counts(d,d["units"],decode(o[name],parents[name][CRF],offset)) for d,o in records])
    reports={}
    chosen={}
    zero=np.zeros((7,AGES),np.float32)
    for name in parents:
        old=summarize(measure(banks["select"],name,zero).sum(0))
        rank=old["exact_labeled_spans"]["f1"],old["token_accuracy"]
        spec=("none",0.);choice=zero
        grid=[]
        for scheme,curve in curves[name].items():
            for gain in (.25,.5,1.):
                q=summarize(measure(banks["select"],name,curve*gain).sum(0))
                pos=q["exact_labeled_spans"]["f1"],q["token_accuracy"];grid.append([scheme,gain,*pos])
                if name=="balanced" and pos[1]<old["token_accuracy"]-.01:continue
                if pos>rank:rank=pos;choice=curve*gain;spec=(scheme,gain)
        path=OUT/f"whole-empirical-duration-{name}.safetensors"
        save_file({**parents[name],"duration.offset":choice.astype("f4")},str(path))
        reports[name]={"selected":list(spec),"weights_sha256":file_hash(path),"grid":grid,"scores":{}}
        chosen[name]=choice
    freeze=ROOT/"results/empirical-duration-frozen.json"
    save_json(freeze,{"train_gold_runs":gold.tolist(),"train_baseline_predicted_runs":{k:v.tolist() for k,v in source.items()},"selected":reports,
                      "selection":"Select on 72 raw development judgments before loading full-test inputs."})
    print("Frozen duration profiles",[(k,v["selected"]) for k,v in reports.items()],flush=True)
    test=[]
    for i,(k,rows) in enumerate(gold_cases().items(),1):
        d=document(k,rows)
        with np.load(CACHE/(k+".npz")) as raw:test.append(stage(d,raw["raw"]))
        if i%144==0:print("Read test raw inputs",i,"/ 712",flush=True)
    banks["test"]=test
    banks["long"]=[x for x in test if len(x[0]["text"].split())>=8000]
    for name,r in reports.items():
        for k,data in banks.items():
            a=measure(data,name,chosen[name]);b=measure(data,name,zero)
            r["scores"][k]={"cases":len(a),"candidate":summarize(a.sum(0)),"parent":summarize(b.sum(0)),
                            "span_token_gap_95pct":pair(a,b)}
        print(name,r["scores"]["test"]["candidate"]["exact_labeled_spans"]["f1"],flush=True)
    save_json(ROOT/"results/empirical-duration-write.json",{"profiles":reports,"selection_receipt_sha256":file_hash(freeze),
              "method":"A bounded duration correction comes from the gold-to-predicted conditional run-length frequency ratio in official training rows. The program stores up to 32 age offsets per role physically in one tensor. Validation selects role-specific pooling or a shared curve. The donor encoder and role labels remain fixed. There were zero optimizer steps."})
if __name__=="__main__":
    main()
