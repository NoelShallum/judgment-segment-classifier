"""Store sparse, corpus-derived residual role keys without changing the pretrained encoder."""
from itertools import product
import numpy as np
from scipy.special import softmax
from safetensors.numpy import load_file,save_file
import torch

from discourse_span_edit import text_features
from hier_model import CRF,HierModel,path_labels
from hier_trial import DONOR,OUT
from preserved_emitter_edit import edit_documents,embeddings
from program_trial import save_json
from seven_paths import ROOT
from span_eval import counts,sets,summarize
from transition_span_edit import sample_whole
from weight_edit import file_hash

def normalized(x):
    return x/np.maximum(1e-6,np.linalg.norm(x,axis=1,keepdims=True))

def key_tables(z,y,max_keys=32):
    rng=np.random.default_rng(705128)
    roles=[]
    for k in range(7):
        eligible=np.flatnonzero(y==k)
        pool=np.sort(rng.choice(eligible,size=min(len(eligible),420),replace=False))
        basis=z[pool]
        seeds=[]
        maximum=np.full(len(pool),-np.inf)
        index=int(np.argmax(basis@basis.mean(0)))
        for _ in range(max_keys):
            seeds.append(int(pool[index]))
            maximum=np.maximum(maximum,basis@basis[index])
            # Exclude previously chosen keys even when two source spans agree.
            maximum[index]=np.inf
            index=int(np.argmin(maximum))
        roles.append(seeds)
    return np.asarray(roles,np.int64)

def lookup(x,keys,values,temperature):
    cosine=x@keys.T
    top=np.argpartition(cosine,-6,axis=1)[:,-6:]
    score=np.take_along_axis(cosine,top,axis=1)
    w=softmax(score/temperature,axis=1)
    return np.einsum("nk,nkc->nc",w,values[top])

def main():
    torch.set_num_threads(2)
    m=HierModel(DONOR)
    documents=edit_documents()
    x=np.concatenate(embeddings(m,documents,"aligned-source-train"))
    y=np.concatenate([d["reference_roles"] for d in documents])
    mean=x.mean(0);scale=x.std(0).clip(.012)
    x2=normalized((x-mean)/scale)
    selections=key_tables(x2,y)
    patches={}
    for n in (8,16,32):
        ids=selections[:,:n].reshape(-1)
        encoded=x2[ids]
        target=np.eye(7)[y[ids]];logits=m.logits(x[ids])
        patches[n]={"k":encoded,"v":(target-softmax(logits/2,axis=1)).astype("f4"),
                    "neutral":(target-(1/7)).astype("f4")}
    orders=(.05,.15)
    approaches=("residual","labels")
    groups=sample_whole();groups["test_screen"]=sets()["test"]
    base={k:load_file(str(OUT/f"whole-discourse-{k}.safetensors")) for k in ("exact","balanced")}
    data={}
    for k,docs in groups.items():
        sid="whole-test" if k=="test_screen" else "raw-boundary-write-"+k
        xx=embeddings(m,docs,sid)
        data[k]=[]
        for d,raw in zip(docs,xx):
            cue,where=text_features(d)
            z=normalized((raw-mean)/scale)
            prior=m.logits(raw)
            logits={q:prior+cue@p["cue_head"]+where@p["position_head"] for q,p in base.items()}
            shift={}
            for n,table in patches.items():
                for tau,mode in product(orders,approaches):
                    shift[n,tau,mode]=lookup(z,table["k"],table["v" if mode=="residual" else "neutral"],tau)
            data[k].append((d,logits,shift))
    def score(group,parent,setting):
        n,tau,mode,gain=setting
        measurements=[]
        for d,logits,shift in data[group]:
            q=logits[parent] if not gain else logits[parent]+gain*shift[n,tau,mode]
            measurements.append(counts(d,d["units"],path_labels(q,base[parent][CRF])))
        return summarize(np.sum(measurements,axis=0))
    ctr=(8,.05,"residual",0.)
    report={}
    for parent in base:
        specs=[ctr,*product(patches,orders,approaches,(.2,.6,1.0))]
        val=score("select",parent,ctr);best=None;trial=[]
        for s in specs:
            q=score("select",parent,s);key=q["exact_labeled_spans"]["f1"],q["token_accuracy"]
            trial.append([*s,*key])
            if parent=="balanced" and key[1]<val["token_accuracy"]-.01:continue
            if best is None or key>best[0]:best=key,s
        _,selected=best
        n,tau,mode,gain=selected
        path=OUT/f"whole-corpus-residual-memory-{parent}.safetensors"
        mem=patches[n];keys=mem["k"];values=mem["v" if mode=="residual" else "neutral"]
        save_file({**base[parent],"memory.keys":keys.astype("f4"),"memory.values":(gain*values).astype("f4"),
                   "memory.center":mean.astype("f4"),"memory.invscale":(1/scale).astype("f4"),
                   "memory.temperature":np.array(tau,np.float32)},str(path))
        obs={k:{"candidate":score(k,parent,selected),"incumbent":score(k,parent,ctr),"cases":len(v)} for k,v in groups.items()}
        report[parent]={"settings":selected,"weights_sha256":file_hash(path),"candidates":trial,"scores":obs}
        print(parent,*selected,obs["test_screen"]["candidate"]["exact_labeled_spans"]["f1"],obs["test_screen"]["incumbent"]["exact_labeled_spans"]["f1"],flush=True)
    save_json(ROOT/"results/corpus-residual-memory.json",{"approach":"Direct sparse key-value memory with up to 224 selected source training activations. Choose representative dissimilar keys per role using greedy diversity. Store expected role residuals against parent logits or centered expected role values. Use training cases only for keys; development chooses gain, memory size and scoring. No gradients or nearest-reference retrieval on held-out cases.",
             "source_train_cases":len(documents),"source_train_rows":len(y),"profiles":report})
if __name__=="__main__":
    main()
