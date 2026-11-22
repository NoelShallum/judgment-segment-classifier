"""Freeze train-only method choices and do one aligned replay on prespecified official validation IDs."""
import argparse
import hashlib
import json
import os
import time
import numpy as np
from safetensors.numpy import load_file
from safetensors import safe_open
from extract import groups, path_for
from prepare import ROOT, save
from study import SOURCE, data, h, load_source, metrics, norm, KEY_PERM, compare, cw, ridge
from joint import decode, position, join, scale
from mlm import layer, ROWS, MIX
from uncertainty import interval


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run_freeze(stage):
    dest=ROOT/f"common-freeze-{stage}.json"
    if dest.exists():return
    methods=[]
    for alg in ["write","joint","mlm"]:
        f=ROOT/f"{alg}-{stage}-context-384"
        s=json.loads((f/"report.json").read_text())
        if alg=="write":
            methods.append(("error_correction",s["results"]["select"]["direct_write"]["macro_f1_7"],f/"edit.safetensors"))
        if alg=="joint":
            for name,r in s["experiments"].items():
                methods.append(("joint_"+name,r["results"]["select"]["direct"]["macro_f1_7"],f/f"{name}.safetensors"))
        if alg=="mlm":
            for key in ["preceding_projection_write","decoder_row_replacement"]:
                methods.append((key,s["scores"]["select"][key]["macro_f1_7"],f/"writes.safetensors"))
    name,_,_=max(methods,key=lambda a:a[1])
    rec={"selection":"highest seven-class macro-F1 on train-only selection judgments across these physically written candidates",
         "primary":name,"stage":stage,"official_validation_labels_scored_before_freeze":False,
         "cohorts":"all nonblank rows of first 4 / first 16 of previously sealed small_validation_judgment_ids; no row subsampling",
         "manifest_sha256":digest(ROOT.parent/"shared-confirmation.json"),
         "methods":[{"name":n,"select_macro_f1_seven":s,"checkpoint":str(p.relative_to(ROOT)),"sha256":digest(p)} for n,s,p in methods]}
    save(dest,rec);print("freeze",name,methods,flush=True)


def eval_shared(stage,n):
    f=ROOT/f"common-freeze-{stage}.json"
    z=json.loads(f.read_text())
    gs=groups("common")["common"][:n]
    d=data(gs,384,"context");y=d["y"]
    out={"status":"complete","cohort":f"first {n} precommitted official validation judgments","cases":len(gs),"rows":len(y),
         "case_ids":[v["id"] for v in gs],"tokens":int(d["tok"].sum()),"freeze_sha256":digest(f),
         "freeze":str(f.relative_to(ROOT)),"stage":stage,"prospective_primary_name":z["primary"],"methods":{}}
    with safe_open(str(ROOT/"model/model.safetensors"),framework="numpy") as ck:
        one=ck.get_tensor("head.linear1.weight");base2=ck.get_tensor("head.linear2.weight")
    original=load_source()
    kt=layer(d["native"],one)
    incumbent=[]
    for v in gs:
        p=ROOT.parent/"pooler/shared-cases"/(v["id"]+".json")
        src=json.loads(p.read_text())
        assert src["manifest_sha256"]==z["manifest_sha256"] and src["rows"]==len(v["rows"])
        incumbent.extend(src["predictions"]["unchanged"])
    incumbent=np.asarray(incumbent)
    out["incumbent_identity"]="frozen Attoflow 20-percent InLegalBERT reviewer route with unchanged pooler"
    out["incumbent"]=metrics(y,incumbent)
    # The reoptimized head and hyperparameters come from the original train selection receipt.
    splits=groups(stage)
    fitting=join(data(splits["fit"],384,"context"),data(splits["edit"],384,"context"))
    jf=ROOT/f"joint-{stage}-context-384"
    jr=json.loads((jf/"report.json").read_text())
    blobs={"case":d["case"],"source_csv_row":d["row"],"sentence_index":d["ix"],"char_start":d["start"],"char_end":d["end"],"gold":y,"incumbent":incumbent}
    for item in z["methods"]:
        p=ROOT/item["checkpoint"];assert digest(p)==item["sha256"]
        ckpt=load_file(str(p));name=item["name"];optimized=None
        if name=="error_correction":
            W=ckpt["classifier.weight"].T;b=ckpt["classifier.bias"]
            target=(ckpt[SOURCE]-original).astype(float)
            base=(norm(d["e"])@W+b).argmax(1)
            primary=(norm(d["e"]+d["k"]@target.T)@W+b).argmax(1)
            shuffled=(norm(d["e"]+d["k"]@target[:,KEY_PERM].T)@W+b).argmax(1)
        elif name.startswith("joint_"):
            W=ckpt["classifier.weight"].T
            b=position(d)@ckpt["position.weight"].T+ckpt["classifier.bias"]
            tw=float(ckpt["transition.gain"][0]);tx=ckpt["transition.log_probabilities"]
            # Compute factored key weights once. This matches the full edited native projection.
            target=(ckpt[SOURCE]-original).astype(float);keyw=target.T@W.astype(float)
            zero=d["e"]@W+b
            base=decode(d,zero,tx,tw)
            primary=decode(d,zero+d["k"]@keyw,tx,tw)
            shuffled=decode(d,zero+d["k"]@keyw[KEY_PERM],tx,tw)
            key=name.removeprefix("joint_")
            cfg=jr["experiments"][key]["best_configs"]["independent_head_only"]
            _,cp,ck,reg,powr,twh=cfg
            assert ck==0
            fe=fitting["e"];fk=fitting["k"];fp=position(fitting)
            se,sk,sp=scale(fe),scale(fk),scale(fp)
            mat=np.column_stack([fe/se,ck*fk/sk,cp*fp/sp])
            wh,bh=ridge(mat,np.eye(7)[fitting["y"]],cw(fitting["y"],powr),reg,True)
            optimized=decode(d,d["e"]@(wh[:768]/se)+position(d)@(wh[1792:]*cp/sp)+bh,tx,twh)
        else:
            C=ckpt["seven_output_rows.original"];Cfit=ckpt["seven_output_rows.replacement"]
            frozen_h=layer(kt,base2)
            base=(frozen_h@C.T+ckpt["decoder_baseline.bias"]).argmax(1)
            if name=="decoder_row_replacement":
                primary=(frozen_h@Cfit.T+ckpt["decoder_fitted.bias"]).argmax(1)
                shuffled=(frozen_h@(C+(Cfit-C)[:,MIX]).T+ckpt["decoder_fitted.bias"]).argmax(1)
            else:
                target=ckpt["head.linear2.weight"];r=base2+(target-base2)[:,MIX]
                primary=(layer(kt,target)@C.T+ckpt["decoder_baseline.bias"]).argmax(1)
                shuffled=(layer(kt,r)@C.T+ckpt["decoder_baseline.bias"]).argmax(1)
        result={"primary":metrics(y,primary),"matched_zero":metrics(y,base),"equal_norm_shuffle":metrics(y,shuffled),
                "changes_vs_zero":compare(y,base,primary),"changes_vs_incumbent":compare(y,incumbent,primary),
                "delta_macro_f1_vs_incumbent":metrics(y,primary)["macro_f1_7"]-out["incumbent"]["macro_f1_7"]}
        if optimized is not None:
            result["independently_reoptimized_head_only"]=metrics(y,optimized)
            result["paired_bootstrap_vs_optimized_head_only"]=interval(d["case"],y,optimized,primary)
            result["changes_vs_optimized_head_only"]=compare(y,optimized,primary)
            blobs[name+"_reoptimized_head_only"]=optimized
        result["paired_bootstrap_vs_incumbent"]=interval(d["case"],y,incumbent,primary)
        result["paired_bootstrap_vs_matched_zero"]=interval(d["case"],y,base,primary)
        out["methods"][name]=result
        blobs[name]=primary;blobs[name+"_paired_zero"]=base;blobs[name+"_shuffle"]=shuffled
    out["primary"]=out["methods"][z["primary"]]
    # Reoptimized alternative to the identical-head zero ablation is computed with its own frozen checkpoint where applicable.
    save(ROOT/f"common{n}-report.json",out)
    np.savez_compressed(ROOT/f"common{n}-predictions.npz",**blobs)
    print("common",n,len(y),z["primary"],out["primary"]["primary"]["macro_f1_7"],out["incumbent"]["macro_f1_7"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("op",choices=["freeze","eval"]);p.add_argument("--stage",default="screen")
    p.add_argument("--n",type=int,default=16);z=p.parse_args()
    if z.op=="freeze":run_freeze(z.stage)
    else:eval_shared(z.stage,z.n)
