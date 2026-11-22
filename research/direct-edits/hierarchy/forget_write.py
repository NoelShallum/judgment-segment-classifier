"""Program empirical legal-role run lengths into native document-level LSTM forget biases."""
import argparse
import json
import numpy as np
from scipy.special import softmax
from safetensors.numpy import save_file,load_file
from native_memory import HERE,ROOT,DONOR,BASE,HierModel,data,native,confusion_docs,metric,decode,dump,file_hash
from embedding_write import digest
from replay_native import read_native

BIASES=("emitter.lstm.bias_ih_l0","emitter.lstm.bias_ih_l0_reverse")
def plan():
    path=HERE/"forget-plan.npz"
    if path.exists():return np.load(path)
    m=HierModel(DONOR);src=native(m)
    f=np.load(HERE/"native-base-fit-240.npz")
    raw=np.load(ROOT/"cache/sentence-recurrent-inverse/fit.npz")
    np.testing.assert_array_equal(f["keys"],raw["keys"])
    words=raw["words"];h=f["states"];y=f["gold"];off=f["ends"]
    start=np.zeros(len(y),bool);start[off[:-1]]=True;end=np.zeros(len(y),bool);end[off[1:]-1]=True
    transitions=np.bincount(y[:-1][~start[1:]]*7+y[1:][~start[1:]],minlength=49).reshape(7,7).astype(float)+1
    perrole=[];all_dir=[];cur=[];target=[]
    for d,s in enumerate(("","_reverse")):
        hist=transitions if d==0 else transitions.T
        stay=hist.diagonal()/hist.sum(1)
        logits=np.log(stay/(1-stay))
        r=m.weight[3:,d*100:(d+1)*100].astype(float)
        r-=r.mean(0)
        # Score role sensitivity regardless of sign; centered, squared readout weights identify affected cells.
        att=(r*r+1e-5);att/=att.sum(0)
        desired=logits@att
        prev=src["emitter.lstm.weight_hh_l0"+s].numpy()[100:200]
        inp=src["emitter.lstm.weight_ih_l0"+s].numpy()[100:200]
        mu=(h[~(end if d==0 else start),100*d:100*(d+1)].sum(0,dtype=np.float64))/len(y)
        observed=inp@words.mean(0,dtype=np.float64)+prev@mu+src[BIASES[d]].numpy()[100:200]+src["emitter.lstm.bias_hh_l0"+s].numpy()[100:200]
        cur.append(observed);target.append(desired);perrole.append(logits)
        all_dir.append(np.clip(desired-observed,-2.5,2.5))
    full=np.array(all_dir)
    rng=np.random.default_rng(4441);shuf=np.stack([rng.permutation(v) for v in full])
    with path.with_suffix(".part").open("wb") as fp:np.savez(fp,full=full,permuted=shuf,uniform=np.broadcast_to(full.mean(1,keepdims=True),full.shape),centered=full-full.mean(1,keepdims=True),current=cur,target=target,logit_persist=perrole)
    path.with_suffix(".part").replace(path)
    return np.load(path)

def run(split):
    limit={"select":71,"confirm":70,"shared":16}[split];docs=data(split,limit)
    name=f"forget-{split}"
    m=HierModel(DONOR);read,_=read_native(HERE/"native-base-240-71.safetensors")
    before=native(m);baseline=[before[k].numpy().copy() for k in BIASES]
    signs=plan()
    if split=="select":
        arms=[("base",0.),*((tag,g) for tag in ("full","permuted","uniform","centered") for g in (.12,.35,.80)),("full",-.35)]
    else:
        pre=json.loads((HERE/"forget-frozen.json").read_text());arms=[("base",0.)]+[tuple(x) for x in pre["arms"].values()]
        arms=list(dict.fromkeys(arms))
    keys=[];words=[];ys=[];ends=[0]
    for k,tx,gold,cached in docs:
        words.append(m.words(tx) if cached is None else cached);keys.append(k);ys.extend(gold);ends.append(len(ys))
    ys=np.array(ys)
    out={"cases":keys,"rows":len(ys),"gradient_steps":0,"split":split,"tensors":list(BIASES),"subtensor":"zero-based gate rows 100 through 199 of each bias. Remainder stays numerically identical.","arms":{}}
    raw=None
    for kind,g in arms:
        for i,key in enumerate(BIASES):
            target=before[key].numpy();target[:]=baseline[i]
            if kind!="base":target[100:200]+=np.asarray(signs[kind][i]*g,np.float32)
        h=np.concatenate([m.contextualize(w) for w in words])
        pred=decode(read.logits(h),ends,m.transitions)
        if raw is None:raw=pred.copy()
        conf=confusion_docs(ys,pred,ends)
        old=confusion_docs(ys,raw,ends);rng=np.random.default_rng(262);boot=[]
        for j in rng.integers(len(keys),size=(1000,len(keys))):
            a,b=metric(conf[j].sum(0)),metric(old[j].sum(0));boot.append((a["macro_f1"]-b["macro_f1"],a["accuracy"]-b["accuracy"]))
        detail={k:dict(sha256=digest(before[k].numpy()),delta_L2=float(np.linalg.norm(before[k].numpy()-baseline[i])),changed=int(np.count_nonzero(before[k].numpy()!=baseline[i]))) for i,k in enumerate(BIASES)}
        out["arms"][str((kind,g))]=dict(**metric(conf.sum(0)),changed=int((pred!=raw).sum()),correct=int(((pred!=raw)&(pred==ys)).sum()),broken=int(((pred!=raw)&(raw==ys)).sum()),ci95=np.quantile(boot,[.025,.975],axis=0).T.tolist(),weights=detail)
        dump(HERE/(name+".json"),out)
        print(split,kind,g,out["arms"][str((kind,g))]["macro_f1"],out["arms"][str((kind,g))]["accuracy"],flush=True)
    if split=="select":
        selected={}
        for method in ("full","permuted","uniform","centered"):
            pool=[arm for arm in arms if arm[0]==method and arm[1]>0]
            selected[method]=max(pool,key=lambda arm:out["arms"][str(arm)]["macro_f1"])
        outpath=HERE/"forget-frozen.json";dump(outpath,dict(arms=selected,trained_on=240,selection_cases=keys,source=file_hash(BASE),
             gate_plan=file_hash(HERE/"forget-plan.npz"),control_readout=file_hash(HERE/"native-base-240-71.safetensors")))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("split",choices=("select","confirm","shared"));run(p.parse_args().split)
