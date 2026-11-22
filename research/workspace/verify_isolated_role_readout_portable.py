"""Score the isolated CPU runtime on official rows and same-row author predictions."""
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np
from safetensors.numpy import save_file

from isolated_role_readout_write import TAG,NGRAM,OUT,ROOT,EDGE,DONOR,HierModel,cm,paired,row_summary,save_json,torch,document,load_file,file_hash,gold_cases,CACHE
from isolated_role_readout_runtime import label_ids,infer
from hier_model import LABELS

DEST=OUT/"sentence-isolated-portable.safetensors"


def run(args):
    docs,live=args
    torch.set_num_threads(1)
    w=load_file(str(DEST))
    model=HierModel(DONOR)
    items=[]
    for d,mask in docs:
        u=d["reference_units"]
        texts=[d["text"][a:b].strip() for a,b in u]
        with np.load(CACHE/(d["id"]+".npz")) as f:
            h=f["excerpt"][mask]
        z=label_ids(texts,h,w)
        receipt=None
        if d["id"] in live:
            assert infer(texts,model,w)["labels"]==[LABELS[k] for k in z]
            examples=texts[:3]
            assert infer(examples,model,w)["labels"]==[infer([p],model,w)["labels"][0] for p in examples]
            receipt={"case":d["id"],"rows":len(z)}
        items.append((d["id"],cm(np.asarray(d["reference_roles"]),z),receipt))
    return items


def main():
    root=ROOT/"results"
    f=root/(TAG+"-frozen.json")
    plan=json.loads(f.read_text())
    assert plan["choice"][:2]==["combined","lda"]
    model=OUT/(TAG+".safetensors")
    lexical=OUT/(NGRAM+"-memory.safetensors")
    assert plan["weights_sha256"]==file_hash(model) and plan["lexical_weights_sha256"]==file_hash(lexical)
    raw=load_file(str(model))
    p={**{"isolated."+k:raw[k] for k in ("center","scale","bias")},
       "isolated.matrix":np.ascontiguousarray(raw["weight"]),
       "isolated.prior_shift":plan["choice"][-1]*raw["prior"],
       "isolated.lexical":np.ascontiguousarray(load_file(str(lexical))["lexical"][0])}
    save_file(p,str(DEST))
    verified=load_file(str(DEST))
    for key,w in p.items():
        np.testing.assert_array_equal(w,verified[key])
    rows=gold_cases()
    docs=[(document(k,v),np.asarray([bool(s.strip()) for _,s,_ in v])) for k,v in rows.items()]
    live={v["case"] for v in json.loads((EDGE/"qa.json").read_text())["cases"]}
    with ProcessPoolExecutor(3) as pool:
        parts=list(pool.map(run,[(docs[i::3],live) for i in range(3)]))
    bank={key:(m,r) for shard in parts for key,m,r in shard}
    assert len(bank)==len(docs)==712
    actual=np.stack([bank[d["id"]][0] for d,_ in docs])
    scores=row_summary(actual.sum(0))
    prior=root/(TAG+".json")
    study=json.loads(prior.read_text())
    assert study["freeze_sha256"]==file_hash(f)
    assert scores==study["splits"]["test"]["candidate"]
    record=json.loads((root/"known-sentence-numerical-archives.json").read_text())
    arrays=root/"known-sentence-numerical-archives.npz"
    assert record["score_file_sha256"]==file_hash(arrays)
    with np.load(arrays,allow_pickle=False) as o:
        np.testing.assert_array_equal(o["cases"],[d["id"] for d,_ in docs])
        authored=o["standalone"]
    samples=[r for _,r in bank.values() if r]
    assert {r["case"] for r in samples}==live
    out={"scores":scores,"standalone_author_archive":row_summary(authored.sum(0)),
         "paired_to_standalone_author":paired(actual,authored),"cases":len(docs),"verified_live_cases":samples,
         "freeze_sha256":file_hash(f),"source_weights_sha256":file_hash(model),"weights_sha256":file_hash(DEST),
         "width":int(raw["weight"].shape[0]),"lexical_width":len(p["isolated.lexical"])}
    save_json(root/(TAG+"-portable-qa.json"),out)
    print("Verified isolated checkpoint and per-sentence interface",scores["macro_f1"],out["paired_to_standalone_author"]["macro_f1"],flush=True)

if __name__=="__main__":
    main()
