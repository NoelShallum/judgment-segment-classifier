"""Replay the frozen local and graph feature choices on identical judgments."""
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np
from safetensors.numpy import save_file

from sentence_document_graph_write import (
    STEM,ARMS,OUT,PARENT,ROOT,EDGE,torch,load_file,sample_whole,prepare,score,route_cases,confusion,fits,
    independent_validation,document,gold_cases,row_summary,paired,file_hash,save_json,training_documents,measure,
)


def run(args):
    docs,group,choices=args
    cases=prepare((docs,group))
    z=[c.current for c,_ in cases]
    out={"baseline":confusion(cases,z)}
    edits={}
    for name,spec in choices.items():
        root=OUT/(STEM+".safetensors") if name=="local" else OUT/(STEM+"-"+name+".safetensors")
        w=load_file(str(root))
        _,_,_,kind,budget,gain,guard,cut=spec
        # Materialize the unstandardized affine map before replay.
        coef=w["scale"][:,None]*w["matrix"]
        intercept=w["bias"]-w["center"]@coef
        values=[h[ARMS.index(name)]@coef+intercept for _,h in cases]
        y=score(cases,values,route_cases(cases,budget),kind,gain,guard,cut)
        edits[name]=sum(int((a!=b).sum()) for a,b in zip(y,z))
        out[name]=confusion(cases,y)
    return [c.key for c,_ in cases],out,edits


def main():
    torch.set_num_threads(1)
    where=ROOT/"results"
    path=where/(STEM+"-frozen.json")
    f=json.loads(path.read_text())
    freeze_sha=file_hash(path)
    assert f["choice"][0]=="local"
    assert f["weights_sha256"]==file_hash(OUT/(STEM+".safetensors"))
    assert f["parent_sha256"]==file_hash(PARENT)
    choices=f["choices_by_arm"]
    cases=training_documents()
    assert [d["id"] for d in cases]==f["train_cases"]
    with ProcessPoolExecutor(3) as pool:
        shards=list(pool.map(prepare,[(cases[i::3],"graph-fit") for i in range(3)]))
    found={c.key:(c,h) for p in shards for c,h in p}
    data=[found[d["id"]] for d in cases]
    weights={}
    for name in ("semantic","two-hop"):
        _,rarity,lam,*_=choices[name]
        p=next(w for b,l,w in fits(data,ARMS.index(name)) if b==rarity and l==lam)
        target=OUT/(STEM+"-"+name+".safetensors")
        save_file(p,str(target))
        serialized=load_file(str(target))
        for k,v in p.items():
            np.testing.assert_array_equal(v,serialized[k])
        weights[name]=file_hash(target)
    del data,found,shards
    print("Restored frozen graph choices",choices,flush=True)
    groups=sample_whole()
    used={d["id"] for g in groups.values() for d in g}
    excluded=set(json.loads((EDGE/"overlap.json").read_text())["independent_exclusions_for_text_disjoint_check"])
    groups["independent"]=[d for d in independent_validation(used) if d["id"] not in excluded]
    groups["test"]=[document(k,r) for k,r in gold_cases().items()]
    archive_name=json.loads((where/"known-sentence-numerical-archives.json").read_text())
    archive_path=where/"known-sentence-numerical-archives.npz"
    assert file_hash(archive_path)==archive_name["score_file_sha256"]
    top=archive_name["best_by_macro_f1"]
    with np.load(archive_path,allow_pickle=False) as ar:
        np.testing.assert_array_equal(ar["cases"],[d["id"] for d in groups["test"]])
        author=ar[top]
    out={"freeze_sha256":freeze_sha,"comparison_weights_sha256":weights,"author_archive":top,"splits":{}}
    goals=json.loads((where/"sentence-joint-edit.json").read_text())["splits"]
    for split in ("select","confirm","independent","test"):
        docs=groups[split]
        if split=="test":
            with ProcessPoolExecutor(3) as pool:
                chunks=list(pool.map(run,[(docs[i::3],split,choices) for i in range(3)]))
        else:
            chunks=[run((docs,split,choices))]
        where_in={key:i for i,key in enumerate(k for ids,_,_ in chunks for k in ids)}
        sequence=[where_in[d["id"]] for d in docs]
        records={k:np.concatenate([q[k] for ids,q,n in chunks])[sequence] for k in ("baseline",*choices)}
        row={}
        for k in choices:
            part=records[k]
            q=row[k]={"scores":row_summary(part.sum(0)),"paired_to_baseline":paired(part,records["baseline"]),
               "paired_to_local":paired(part,records["local"]),"changed":sum(n[k] for ids,q,n in chunks)}
            if split=="test":
                q["paired_to_aligned_archive"]=paired(part,author)
            if split=="select":
                m=next(q for opt,q in f["grid"] if opt==choices[k])
                np.testing.assert_allclose(measure(part.sum(0)),m,rtol=0,atol=1.e-12)
        assert row_summary(records["baseline"].sum(0))==goals[split]["candidate"]
        root=where/(STEM+".json")
        if root.is_file() and split in (main_record:=json.loads(root.read_text()))["splits"]:
            q=main_record["splits"][split]
            assert q["candidate"]==row["local"]["scores"] and q["changed"]==row["local"]["changed"]
        out["splits"][split]={"cases":len(docs),"baseline":row_summary(records["baseline"].sum(0)),"arms":row}
        save_json(where/(STEM+"-comparison.json"),out)
        print(split,[(k,r["scores"]["macro_f1"],r["changed"]) for k,r in row.items()],flush=True)


if __name__=="__main__":
    main()
