"""Score a frozen native checkpoint against its exact donor on the same judgment rows."""
import argparse
import json
import numpy as np
from safetensors.numpy import load_file
from native_memory import HERE,ROOT,DONOR,BASE,HierModel,confusion_docs,decode,metric,file_hash,dump,data,native

def read_native(path):
    info=json.loads(path.with_suffix(".json").read_text())
    assert info["weights_sha256"]==file_hash(path) and info["sha256_base"]==file_hash(BASE)
    m=HierModel(DONOR); fields=native(m); updates=load_file(str(path))
    for k,b in updates.items():
        a=fields[k].numpy()
        assert a.shape==b.shape and a.dtype==b.dtype
        a[:]=b
    return m,info

def run(patch,split,n):
    m,info=read_native(patch); old=HierModel(DONOR); docs=data(split,n)
    k=[];ends=[0];ys=[];pred=[];orig=[];logits=[];base_logits=[]
    for key,txt,gold,cache in docs:
        assert key not in info["train_cases"] and key not in info["select_cases"]
        prev=old.logits(old.contextualize(old.words(txt) if cache is None else cache))
        z=m.logits(m.contextualize(m.words(txt)))
        p=decode(z,[0,len(gold)],m.transitions)
        q=decode(prev,[0,len(gold)],old.transitions)
        k.append(key);ys.extend(gold);pred.extend(p);orig.extend(q);ends.append(len(ys));logits.extend(z);base_logits.extend(prev)
    ys,pred,orig=map(np.asarray,(ys,pred,orig))
    new=confusion_docs(ys,pred,ends); base=confusion_docs(ys,orig,ends)
    rng=np.random.default_rng(6169)
    deltas=[]
    for i in rng.integers(len(k),size=(2000,len(k))):
        a,b=metric(new[i].sum(0)),metric(base[i].sum(0))
        deltas.append([a["macro_f1"]-b["macro_f1"],a["accuracy"]-b["accuracy"]])
    output=dict(cases=k,rows=len(ys),split=split,base=metric(base.sum(0)),edited=metric(new.sum(0)),
                corrected=int(((pred!=orig)&(pred==ys)).sum()),harmed=int(((pred!=orig)&(orig==ys)).sum()),
                changed=int((orig!=pred).sum()),tensor_checkpoint_sha256=file_hash(patch),
                paired_judgment_95_macro_F1_accuracy=np.quantile(deltas,[.025,.975],axis=0).T.tolist(),gradient_steps=0)
    name=f"replay-{patch.stem}-{split}-{n}"
    dump(HERE/(name+".json"),output)
    temp=HERE/(name+".part")
    with temp.open("wb") as fp:
        np.savez(fp,keys=k,ends=ends,gold=ys,base=orig,pred=pred,logits=np.asarray(logits),base_logits=np.asarray(base_logits))
    temp.replace(HERE/(name+".npz"))
    print(name,output["rows"],output["base"]["macro_f1"],output["edited"]["macro_f1"],output["edited"]["accuracy"],flush=True)
if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("checkpoint");a.add_argument("split",choices=("confirm","holdout","shared"));a.add_argument("n",type=int)
    args=a.parse_args();run(HERE/args.checkpoint,args.split,args.n)
