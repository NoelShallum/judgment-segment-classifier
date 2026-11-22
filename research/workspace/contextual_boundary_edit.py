"""Write a boundary-specific residual into the CRF's off-diagonal transitions."""
import itertools
import numpy as np
from safetensors.numpy import load_file,save_file
from scipy.linalg import solve
import torch

from discourse_span_edit import text_features
from boundary_runtime import boundary_features,variable_decode
from hier_full import CACHE, gold_cases
from hier_model import CRF, HierModel
from hier_trial import DONOR,OUT
from preserved_emitter_edit import edit_documents, embeddings
from program_trial import save_json
from seven_paths import ROOT
from span_eval import counts,document,summarize
from transition_span_edit import sample_whole
from weight_edit import file_hash

STRENGTHS=(-.35,.35,.7,1.4,2.5)


def write_base(h,y):
    mean=h.mean(0);scale=h.std(0).clip(1e-5)
    x=(h-mean)/scale
    prototypes=[x[y==label].mean(0) for label in (0,1)]
    centered=[x[y==label]-prototypes[label] for label in (0,1)]
    cov=sum((a.T@a)/max(len(a)-1,1) for a in centered)/2
    prior=float(y.mean())
    for ridge in (.01,.1,.5):
        matrix=cov+np.eye(len(mean))*(ridge*cov.trace()/len(cov))
        w=solve(matrix,prototypes[1]-prototypes[0],assume_a="pos")
        b=-(prototypes[1]+prototypes[0])@w/2
        w=w/scale
        yield (ridge,w.astype(np.float32),np.float32(b-w@mean),prior)


def feed(train,model):
    x=embeddings(model,train,"aligned-source-train")
    vectors=[];y=[]
    for d,states in zip(train,x):
        if len(states)<2:continue
        logits=model.logits(states)
        vectors.append(boundary_features(logits))
        g=np.array(d["reference_roles"])
        y.append(g[1:]!=g[:-1])
    return np.concatenate(vectors),np.concatenate(y).astype(np.int8)


def test_score(a,b):
    rng=np.random.default_rng(99127)
    diff=[]
    for ii in rng.integers(len(a),size=(2000,len(a))):
        p=summarize(a[ii].sum(0));q=summarize(b[ii].sum(0))
        diff.append([p["exact_labeled_spans"]["f1"]-q["exact_labeled_spans"]["f1"],p["token_accuracy"]-q["token_accuracy"]])
    return np.quantile(diff,[.025,.975],axis=0).T.tolist()


def main():
    torch.set_num_threads(2)
    m=HierModel(DONOR)
    train=edit_documents()
    h,y=feed(train,m)
    presets=list(write_base(h,y))
    authority={k:load_file(str(OUT/f"whole-discourse-{k}.safetensors")) for k in ("exact","balanced")}
    cohorts=sample_whole()
    # The train files contain official provided cuts. The target benchmark candidate cuts are generated from raw text.
    cohorts["test"]=[document(k,rows) for k,rows in gold_cases().items()]
    evidence={}
    for split,docs in cohorts.items():
        table=[]
        if split!="test":
            states=embeddings(m,docs,"raw-boundary-write-"+split)
            raw=[m.logits(x) for x in states]
        else:
            raw=[]
            for d in docs:
                with np.load(CACHE/(d["id"]+".npz")) as pkg:raw.append(pkg["raw"])
        for d,array in zip(docs,raw):
            u,v=text_features(d)
            base={k:array+u@g["cue_head"]+v@g["position_head"] for k,g in authority.items()}
            z=boundary_features(array)
            table.append((d,base,[np.clip(z@p[1]+p[2],-6,6) for p in presets]))
        evidence[split]=table
        print("Read",split,len(table),flush=True)
    def aggregate(id,mode,preset,weight,shuffle=False):
        out=[]
        for d,s,z in evidence[id]:
            signal=z[preset]*weight
            if shuffle and len(signal)>1:
                signal=np.roll(signal,max(1,len(signal)//2))
            guess=variable_decode(s[mode],authority[mode][CRF],signal)
            out.append(counts(d,d["units"],guess))
        return np.stack(out)
    report={};grid=[]
    token_floor=summarize(aggregate("select","balanced",0,0).sum(0))["token_accuracy"]-.01
    winner={"exact":None,"balanced":None}
    for mode,block,strength in itertools.product(authority,range(len(presets)),(0.,*STRENGTHS)):
        q=summarize(aggregate("select",mode,block,strength).sum(0))
        rank=q["exact_labeled_spans"]["f1"],q["token_accuracy"]
        grid.append([mode,block,strength,*rank])
        for mode2 in winner:
            if mode2=="balanced" and rank[1]<token_floor:continue
            rec=(rank,mode,block,strength)
            if winner[mode2] is None or rank>winner[mode2][0]:winner[mode2]=rec
    for key,(_,mode,block,strength) in winner.items():
        path=OUT/(f"whole-contextual-boundary-{key}.safetensors")
        ridge,w,b,_=presets[block]
        save_file({**authority[mode],"boundary.weight":w,"boundary.bias":np.array(b,np.float32),"boundary.gain":np.array(strength,np.float32)},str(path))
        r={}
        for fold in evidence:
            a=aggregate(fold,mode,block,strength)
            c=aggregate(fold,mode,block,0)
            s=aggregate(fold,mode,block,strength,shuffle=True)
            q=summarize(a.sum(0))
            r[fold]={"cases":len(a),"candidate":q,"incumbent":summarize(c.sum(0)),"shuffled_boundary_control":summarize(s.sum(0)),
                     "gap_vs_incumbent_95pct":test_score(a,c),"gap_vs_shuffled_95pct":test_score(a,s)}
        report[key]={"incumbent_profile":mode,"ridge":ridge,"strength":strength,"weights_sha256":file_hash(path),"scores":r}
        x=r["test"]["candidate"]
        print(key,mode,ridge,strength,x["exact_labeled_spans"]["f1"],x["token_accuracy"],flush=True)
    save_json(ROOT/"results/contextual-boundary-write.json",{"training_cases":len(train),"training_edges":len(y),"training_positive_edges":int(y.sum()),
        "method":"A closed-form regularized LDA fit from pre-existing donor activations on the official training split. The score is written as a context-dependent residual on off-diagonal transition scores. Every judgment is decoded from raw text and proposed units. The zero residual is eligible. Shuffling within the case preserves the intervention distribution while breaking alignment.",
        "coefficients":grid,"profiles":report})
if __name__=="__main__":
    main()
