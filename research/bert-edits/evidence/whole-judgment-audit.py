"""Verify remote receipts against the local official labels and estimate paired uncertainty."""
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
from zipfile import ZipFile

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent))
from span_comparison import tally


ROLES=("None","Facts","Issue","Arguments of Petitioner","Arguments of Respondent","Reasoning","Decision")
ROLE_ID={r:i for i,r in enumerate(ROLES)}


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def unpack(path, destination):
    with ZipFile(path) as bundle:
        names=bundle.namelist()
        assert len(names)==len(set(names)) and bundle.testzip() is None
        assert all(not PurePosixPath(n).is_absolute() and ".." not in PurePosixPath(n).parts for n in names)
        expected=json.loads(bundle.read("SHA256SUMS.json"))
        assert set(names)==set(expected)|{"SHA256SUMS.json"}
        for name,h in expected.items():
            assert hashlib.sha256(bundle.read(name)).hexdigest()==h
        bundle.extractall(destination)


def cm(g,p):
    assert g.shape==p.shape and np.isin(p,np.arange(7)).all()
    return np.bincount(g.astype(np.int64)*7+p,minlength=49).reshape(7,7)


def scores(c):
    hit=np.diagonal(c,axis1=-2,axis2=-1)
    den=c.sum(-1)+c.sum(-2)
    role=2*hit/den.clip(min=1)
    return role.mean(-1),hit.sum(-1)/c.sum(axis=(-2,-1)),role


def verified_rows(plan, group, val):
    ids=plan["groups"][group]
    golds,indices,review=[],[],[]
    starts=[0]
    seen=set()
    for case in ids:
        parent=HERE/"new-cases" if group=="fresh" else HERE.parent/"bert-t4/cases"
        path=parent/(case+".npz")
        with np.load(path,allow_pickle=False) as x:
            rows=val[case]
            g=np.array([r[1] for r in rows],dtype=np.int8)
            ix=np.array([r[0] for r in rows],dtype=np.int64)
            np.testing.assert_array_equal(g,x["gold"])
            np.testing.assert_array_equal(ix,x["original_row"])
            h=hashlib.sha256()
            for _,_,s in rows:
                h.update(" ".join(s.casefold().split()).encode())
                h.update(b"\0")
            assert h.hexdigest()==x["case_sha256"].item()==plan["case_text_sha256"][case]
            assert h.hexdigest() not in seen
            seen.add(h.hexdigest())
            packed=x["positions"]
            assert len(set(packed.tolist()))==len(packed) and len(packed)==len(x["ids"])
            golds.append(g); indices.append(ix); review.append(packed+starts[-1])
            starts.append(starts[-1]+len(g))
    return ids,np.concatenate(golds),np.array(starts),np.concatenate(review)


ARM_MAP={
    "query-results":{"incumbent":"current","temperature-edited":"query_temperature","temperature-shuffled":"query_temperature_shuffled",
                     "conditional-edited":"query_conditional","conditional-shuffled":"query_conditional_shuffled"},
    "affine-results":{"incumbent":"current","parent_without_bert":"without_review","attention_nonzero":"norm_attention","attention_shuffled":"norm_attention_shuffled",
                     "attention_selected":"norm_attention_selected","ffn_nonzero":"norm_ffn","ffn_shuffled":"norm_ffn_shuffled","ffn_selected":"norm_ffn_selected"},
    "combination-results":{"incumbent":"current","joint_selected":"joint_selected","pair_nonzero":"pair_nonzero","pair_shuffled":"pair_shuffled"},
    "previous-controls":{"current_actual":"current","targeted":"previous_value","shuffled":"previous_value_shuffled",
                         "pooled refit":"previous_pooled_head","attention-aware refit":"previous_attention_head"},
    "sparse-transfer":{"incumbent":"current","role_pairs":"excerpt_ffn","shuffled":"excerpt_ffn_shuffled","classifier_refit":"excerpt_classifier"}
}
CONTROLS={"query_temperature":"query_temperature_shuffled","query_conditional":"query_conditional_shuffled",
          "norm_attention":"norm_attention_shuffled","norm_ffn":"norm_ffn_shuffled","pair_nonzero":"pair_shuffled","previous_value":"previous_value_shuffled",
          "excerpt_ffn":"excerpt_ffn_shuffled"}


def audit_group(out, group, plan, val):
    ids,gold,starts,review=verified_rows(plan,group,val)
    predictions={}
    for directory,mapping in ARM_MAP.items():
        path=out/directory/(group+"-predictions.npz")
        with np.load(path,allow_pickle=False) as a:
            np.testing.assert_array_equal(a["case_ids"],ids)
            np.testing.assert_array_equal(a["gold"],gold)
            np.testing.assert_array_equal(a["offsets" if "offsets" in a else "case_starts"],starts)
            np.testing.assert_array_equal(a["review" if "review" in a else "reviewed"],review)
            curves=read(out/directory/(group+("-report.json" if directory=="affine-results" else "-scores.json")))["scores"]
            for field,tag in mapping.items():
                if tag in predictions:
                    np.testing.assert_array_equal(predictions[tag],a[field])
                predictions[tag]=a[field]
                r=curves[field]
                if "whole_judgment" in r:
                    r=r["whole_judgment"]
                f,acc,_=scores(cm(gold,a[field]))
                np.testing.assert_allclose([r["macro_f1"],r["accuracy"]],[f,acc],atol=1e-12,rtol=0)
    base=predictions["current"]
    isreview=np.zeros(len(base),dtype=bool); isreview[review]=True
    draws=np.random.default_rng(722641).integers(len(ids),size=(4000,len(ids)))
    results,boot={},{}
    for tag,pred in predictions.items():
        np.testing.assert_array_equal(pred[~isreview],base[~isreview])
        matrices=[]; routed=[]; runs=[]
        for a,b in zip(starts,starts[1:]):
            g,p=gold[a:b],pred[a:b]; keep=isreview[a:b]
            matrices.append(cm(g,p)); routed.append(cm(g[keep],p[keep])); runs.append(tally(g,p))
        matrix,route,spans=map(np.stack,(matrices,routed,runs))
        f,acc,per=scores(matrix.sum(0)); r,ra,rper=scores(route.sum(0))
        s=spans.sum(0)
        bs=spans[draws].sum(1)
        bf,ba,_=scores(matrix[draws].sum(1)); br,bra,_=scores(route[draws].sum(1))
        boot[tag]=np.column_stack((bf,ba,br,bra,2*bs[:,0]/(bs[:,1]+bs[:,2])))
        results[tag]={"macro_f1":float(f),"accuracy":float(acc),"routed_macro_f1":float(r),"routed_accuracy":float(ra),
                      "exact_span_f1":float(2*s[0]/(s[1]+s[2])),"matched_spans":int(s[0]),"gold_spans":int(s[2]),
                      "by_role":dict(zip(ROLES,per.tolist())),"routed_by_role":dict(zip(ROLES,rper.tolist())),
                      "changed":int((base!=pred).sum()),
                      "fixed":int(((base!=gold)&(pred==gold)).sum()),"harmed":int(((base==gold)&(pred!=gold)).sum())}
    for tag,values in results.items():
        for other in dict.fromkeys(["current",CONTROLS.get(tag,"current"),"previous_pooled_head","previous_attention_head"]):
            v=np.quantile(boot[tag]-boot[other],[.025,.975],axis=0).T.tolist()
            values.setdefault("paired_95",{})[other]=dict(zip(["macro_f1","accuracy","routed_macro_f1","routed_accuracy","exact_span_f1"],v))
    excerpt_sets={g:{str(r["case"]) for r in read(HERE/"excerpt/capture-input"/(g+"-rows.json"))} for g in ("select","confirm")}
    restricted={}
    for label,excluded in (("exclude_excerpt_selector",excerpt_sets["select"]),
                           ("exclude_excerpt_selector_and_first_confirmation",excerpt_sets["select"]|excerpt_sets["confirm"])):
        ix=[j for j,k in enumerate(ids) if k not in excluded]
        if not ix:
            continue
        samples=np.random.default_rng(11713).integers(len(ix),size=(4000,len(ix)))
        groups={}
        for tag in ("current","excerpt_ffn","excerpt_ffn_shuffled","excerpt_classifier"):
            cells=np.stack([cm(gold[starts[j]:starts[j+1]],predictions[tag][starts[j]:starts[j+1]]) for j in ix])
            f,ac,_=scores(cells.sum(0)); bf,_,_=scores(cells[samples].sum(1))
            if tag=="current":
                reference=bf
            groups[tag]={"macro_f1":float(f),"accuracy":float(ac),"delta_from_current_ci95":np.quantile(bf-reference,[.025,.975]).tolist()}
        restricted[label]={"judgments":len(ix),"sentences":sum(int(starts[j+1]-starts[j]) for j in ix),"arms":groups}
    return {"n_judgments":len(ids),"n_sentences":len(gold),"reviewed":len(review),"role_counts":np.bincount(gold,minlength=7).tolist(),
            "routed_role_counts":np.bincount(gold[review],minlength=7).tolist(),"methods":results,"excerpt_case_exclusions":restricted}


def main():
    archive=HERE/"all-returned.zip"
    out=HERE/"verified"
    unpack(archive,out)
    plan=read(HERE/"round2-plan.json")
    assert digest(HERE/"round2-plan.json")==digest(out/"data/round2-plan.json")
    assert digest(ROOT/"references/official/val.csv")==plan["source"]["val"]
    keys=set(plan["groups"]["fresh"]+plan["groups"]["confirm"])
    val={k:[] for k in keys}
    with (ROOT/"references/official/val.csv").open(encoding="utf-8-sig",newline="") as f:
        for i,row in enumerate(csv.DictReader(f)):
            if row["Index"] in keys and row["Text"].strip():
                val[row["Index"]].append((i,ROLE_ID[row["Label"]],row["Text"]))
    result={g:audit_group(out,g,plan,val) for g in ("confirm","fresh")}
    result["returned_sha256"]=digest(archive)
    result["bootstrap"]="4,000 paired bootstrap resamples of whole judgments, pooled role confusion matrices and span counts."
    path=HERE/"results-audit.json"
    path.write_text(json.dumps(result,indent=2)+"\n")
    print("verified",result["returned_sha256"])
    for group in ("confirm","fresh"):
        print(group,{k:round(s["macro_f1"],5) for k,s in result[group]["methods"].items()})


if __name__=="__main__":
    main()
