"""Test whether a written risk head routes recoverable LegalSeg errors."""
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np
from safetensors.numpy import save_file

from audit_author_complementarity import author_outputs
from embeddinggemma_routed_write import (
    DONOR,EDGE,OUT,PARENT,ROOT,HierModel,aligned_embeddings,case_inputs,cm,document,edit_documents,
    embeddings,file_hash,gold_cases,independent_validation,load_file,paired,row_summary,
    sample_whole,save_json,torch,
)

STEM="sentence-joint-repair-routing"
BUDGETS=("3 per case","5%","10%","20%","30%")
HEURISTICS=("smallest utility gap","posterior entropy","low assigned probability","even spacing")
STRENGTHS=(.06,.22,.80)


def load_views(args):
    docs,split=args
    torch.set_num_threads(1)
    model=HierModel(DONOR)
    parent=load_file(str(PARENT))
    vocab=json.loads((EDGE/"vocab.json").read_text())
    states=embeddings(model,docs,"aligned-source-train") if split=="train" else aligned_embeddings(model,docs,split)
    return [case_inputs(d,h,model,parent,vocab) for d,h in zip(docs,states)]


def design(c):
    p=c.structure[:,:7]
    assigned=p[np.arange(len(p)),c.current]
    entropy=-(p*np.log(np.maximum(p,1.e-8))).sum(1)
    top=p.max(1)
    return np.column_stack((c.structure,assigned,entropy,top,
                            np.square(assigned),np.square(entropy)))


def fits(rows):
    h=np.concatenate([design(c) for c in rows]).astype(np.float64)
    y=np.concatenate([c.current!=c.gold for c in rows]).astype(np.float64)
    label=np.concatenate([c.gold for c in rows])
    center=h.mean(0)
    scale=1/np.maximum(h.std(0),.035)
    x=np.column_stack(((h-center)*scale,np.ones(len(h))))
    freq=np.maximum(np.bincount(label,minlength=7)/len(label),.015)
    regularizer=np.r_[np.ones(h.shape[1]),.025]
    for balanced in (0.,.30):
        w=freq[label]**-balanced
        gram=x.T@(w[:,None]*x)
        rhs=x.T@(w*y)
        for damping in STRENGTHS:
            coef=np.linalg.solve(gram+np.diag(regularizer*damping*np.trace(gram)/len(gram)),rhs)
            yield balanced,damping,{"center":center,"scale":scale,"weight":np.ascontiguousarray(coef[:-1]),
                                   "bias":np.asarray(coef[-1])}


def size(c,b):
    if b=="3 per case":
        return min(3,len(c.eligible))
    return min(len(c.eligible),max(1,int(np.ceil(len(c.gold)*int(b[:-1])/100))))


def ordered(c,method,weights=None):
    if method=="smallest utility gap":
        return c.eligible
    if method=="even spacing":
        ix=np.sort(c.eligible)
        # A deterministic van der Corput order produces nested, evenly spread subsets.
        width=max(1,(len(ix)-1).bit_length())
        values=np.array([int(f"{i:0{width}b}"[::-1],2) for i in range(len(ix))])
        return ix[np.argsort(values,kind="stable")]
    p=c.structure[:,:7]
    if method=="posterior entropy":
        risk=-(p*np.log(np.maximum(p,1.e-8))).sum(1)
    elif method=="low assigned probability":
        risk=-p[np.arange(len(p)),c.current]
    else:
        risk=((design(c)-weights["center"])*weights["scale"])@weights["weight"]+weights["bias"]
    return c.eligible[np.argsort(-risk[c.eligible],kind="stable")]


def capture(cases,method,budget,weights=None):
    return sum(int(np.count_nonzero(c.current[q]!=c.gold[q])) for c in cases
               for q in [ordered(c,method,weights)[:size(c,budget)]])


def study(cases,choices,archive):
    baseline=np.stack([cm(c.gold,c.current) for c in cases])
    errors=sum(int((c.current!=c.gold).sum()) for c in cases)
    by_budget={}
    for budget in BUDGETS:
        methods={}
        index={name:[ordered(c,name,w)[:size(c,budget)] for c in cases] for name,w in choices.items()}
        control=index["smallest utility gap"]
        for name,indices in index.items():
            reference=[];perfect=[];actual_reference=[];roles=np.zeros(7,np.int64)
            only=all_found=total=0
            for c,q,old in zip(cases,indices,control):
                mistake=c.current!=c.gold
                ref=c.current.copy()
                good=c.current.copy()
                old_ref=c.current.copy()
                found=q[mistake[q]]
                roles+=np.bincount(c.gold[found],minlength=7)
                total+=len(q)
                all_found+=len(found)
                good[q]=c.gold[q]
                perfect.append(cm(c.gold,good))
                if archive is not None:
                    key=archive[c.key]
                    keep=q[key[q]==c.gold[q]]
                    keep_old=old[key[old]==c.gold[old]]
                    ref[keep]=c.gold[keep]
                    old_ref[keep_old]=c.gold[keep_old]
                    reference.append(cm(c.gold,ref))
                    actual_reference.append(cm(c.gold,old_ref))
                    only+=int(mistake[keep].sum())
            z=np.stack(perfect)
            item={"queries":total,"errors_found":all_found,"error_recall":all_found/errors,
                  "errors_found_by_role":roles.tolist(),"perfect_repair":row_summary(z.sum(0))}
            if archive is not None:
                a,b=np.stack(reference),np.stack(actual_reference)
                item.update({"archive_recoverable_errors_found":only,"archive_assisted":row_summary(a.sum(0)),
                             "archive_assisted_gap_to_current_route_ci":paired(a,b)})
            methods[name]=item
        by_budget[budget]=methods
    return {"cases":len(cases),"baseline":row_summary(baseline.sum(0)),"errors":errors,"budgets":by_budget}


def main():
    torch.set_num_threads(1)
    root=ROOT/"results"
    cohorts=sample_whole()
    core=edit_documents()
    train=load_views((core,"train"))
    candidates=dict(((a,b),w) for a,b,w in fits(train))
    del train
    select=load_views((cohorts["select"],"select"))
    totals={k:[capture(select,"written risk",b,w) for b in ("3 per case","5%","10%")] for k,w in candidates.items()}
    choice=max(totals,key=lambda k:(sum(totals[k]),totals[k]))
    physical=OUT/(STEM+".safetensors")
    save_file(candidates[choice],str(physical))
    weights=load_file(str(physical))
    for k,expected in candidates[choice].items():
        np.testing.assert_array_equal(expected,weights[k])
    freeze=root/(STEM+"-frozen.json")
    save_json(freeze,{"parent_sha256":file_hash(PARENT),"weights_sha256":file_hash(physical),
                      "training_cases":[d["id"] for d in core],"selection_cases":[c.key for c in select],
                      "selection_objective":"Pooled incumbent errors found at fixed budgets of 3 seats, 5% and 10% per case.",
                      "choice":choice,"selection_grid":[[list(k),value] for k,value in totals.items()]})
    print("Frozen error-risk head",choice,totals,flush=True)
    choices={name:None for name in HEURISTICS}|{"written risk":weights}
    result={"freeze_sha256":file_hash(freeze),"splits":{}}
    verified=json.loads((root/"sentence-joint-edit.json").read_text())["splits"]
    for split in ("select","confirm","test"):
        archive=None
        if split=="test":
            source=gold_cases()
            docs=[document(k,r) for k,r in source.items()]
            y,ref,provenance=author_outputs(source)
            values=ref[provenance["best_by_macro_f1"]]
            offsets=np.r_[0,np.cumsum([len(d["reference_units"]) for d in docs])]
            archive={d["id"]:values[i:j] for d,i,j in zip(docs,offsets,offsets[1:])}
            assert len(y)==len(values)==offsets[-1]
            with ProcessPoolExecutor(3) as pool:
                parts=list(pool.map(load_views,[(docs[i::3],split) for i in range(3)]))
            entries={c.key:c for group in parts for c in group}
            frames=[entries[d["id"]] for d in docs]
        else:
            frames=select if split=="select" else load_views((cohorts[split],split))
        result["splits"][split]=out=study(frames,choices,archive)
        assert out["baseline"]==verified[split]["candidate"]
        save_json(root/(STEM+".json"),result)
        print(split,len(frames),[(b,[(name,d["errors_found"],d.get("archive_recoverable_errors_found"))
                                   for name,d in out["budgets"][b].items()]) for b in BUDGETS],flush=True)


if __name__=="__main__":
    main()
