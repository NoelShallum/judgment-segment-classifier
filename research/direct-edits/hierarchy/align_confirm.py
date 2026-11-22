import json
import numpy as np
from embedding_write import HERE,confusion,metric,dump
from native_memory import confusion_docs
from lex_counts import file_hash
def main():
    ref=np.load(HERE/"strong-confirm.npz")
    lookup={str(k):slice(a,b) for k,a,b in zip(ref["keys"],ref["ends"],ref["ends"][1:])}
    for name in ("widened-confirm-70","production-word-table-confirm-24"):
        p=HERE/(name+".json");data=json.loads(p.read_text());widen=name.startswith("widened")
        f=np.load(HERE/(name+("" if widen else "-base")+".npz"))
        keys=f["keys"];y=f["gold"];idx=np.concatenate([np.arange(ref["pred"].size)[lookup[str(k)]] for k in keys])
        np.testing.assert_array_equal(ref["gold"][idx],y);h=f["pred"];inc=ref["pred"][idx]
        if widen:
            a=confusion_docs(y,h,f["ends"]);b=confusion_docs(y,inc,f["ends"])
            rng=np.random.default_rng(989);de=[]
            for ii in rng.integers(len(keys),size=(2000,len(keys))):
                x,z=metric(a[ii].sum(0)),metric(b[ii].sum(0));de.append((x["macro_f1"]-z["macro_f1"],x["accuracy"]-z["accuracy"]))
            data["metrics"]["incumbent"]=metric(b.sum(0));data["paired"]["incumbent"]=dict(corrected=int(((h!=inc)&(h==y)).sum()),harmed=int(((h!=inc)&(inc==y)).sum()),changed=int((h!=inc).sum()),ci95=np.quantile(de,[.025,.975],axis=0).T.tolist())
        else:
            data["incumbent_20pct"]=metric(confusion(y,inc))
        data["reference_20pct_sha256"]=file_hash(HERE/"strong-confirm.npz");dump(p,data)
        print(name,metric(confusion(y,inc))["macro_f1"],flush=True)
if __name__=="__main__":main()
