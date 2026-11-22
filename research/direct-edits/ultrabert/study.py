"""Closed-form writes to a multiclass readout and a pretrained MLP projection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import scipy.linalg as lin
import torch
from safetensors import safe_open
from safetensors.numpy import save_file

from extract import ARCH, Encoder, groups, identifier, path_for, read
from prepare import LABELS, ROOT, save

EYE = np.eye(7)
SOURCE = "ultrabert.layers.23.mlp.down_proj.weight"
KEY_PERM = np.random.default_rng(5611022).permutation(1024)


def data(items, cap, mode):
    fields = ["e","k","m","native","ix","row","start","end","tok","y"]
    vals = {f:[] for f in fields}
    vals["case"]=[];vals["ncase"]=[]
    used=set()
    for item in items:
        rec=read(item,cap,mode)
        fresh=[j for j,r in enumerate(rec["row"]) if (item["id"],int(r)) not in used]
        used.update((item["id"],int(rec["row"][j])) for j in fresh)
        for f in fields:vals[f].append(rec[f][fresh])
        vals["case"].append(np.repeat(item["id"],len(fresh)))
        vals["ncase"].append(np.repeat(item["ncase"],len(fresh)))
    return {k:np.concatenate(v).astype(np.float64) if k in ["e","k","m","native"] else np.concatenate(v) for k,v in vals.items()}


def rms(x):
    return np.sqrt(np.mean(x*x,axis=1,keepdims=True)+1e-8)


def norm(x):
    return x/rms(x)


def metrics(y,p):
    c=np.zeros((7,7),int)
    np.add.at(c,(y,p),1)
    d=c.diagonal()
    f=np.divide(2*d,c.sum(0)+c.sum(1),out=np.zeros(7,dtype=float),where=(c.sum(0)+c.sum(1))>0)
    return {"rows":len(y),"accuracy":float(d.sum()/len(y)),"macro_f1_7":float(f.mean()),"role_f1":f.tolist(),
            "true_counts":c.sum(1).tolist(),"predicted_counts":c.sum(0).tolist(),"confusion":c.tolist()}


def cw(y, exponent):
    n=np.bincount(y,minlength=7)
    w=np.power(np.maximum(n[y],1),-exponent)
    return w/w.mean()


def ridge(x,t,w,reg, center):
    if center:
        xm=np.average(x,axis=0,weights=w)
        tm=np.average(t,axis=0,weights=w)
    else:
        xm=np.zeros(x.shape[1]);tm=np.zeros(t.shape[1])
    a=(x-xm)*np.sqrt(w)[:,None]
    b=(t-tm)*np.sqrt(w)[:,None]
    lam=max(reg*np.sum(a*a)/len(a),1e-8)
    if a.shape[0] < a.shape[1]:
        g=a@a.T
        g.flat[::len(g)+1]+=lam
        beta=a.T@lin.solve(g,b,assume_a="pos",check_finite=False)
    else:
        g=a.T@a
        g.flat[::len(g)+1]+=lam
        beta=lin.solve(g,a.T@b,assume_a="pos",check_finite=False)
    return beta, tm-xm@beta


def features(d,kind):
    return norm(d[kind])


def select_head(fit,sel,kind):
    x=features(fit,kind);xx=features(sel,kind)
    outcomes=[]
    for exponent in (0.,0.5,1.):
        for reg in (.03,.3,1.5):
            w,b=ridge(x,EYE[fit["y"]],cw(fit["y"],exponent),reg,True)
            score=metrics(sel["y"],(xx@w+b).argmax(1))
            outcomes.append((score["macro_f1_7"],(exponent,reg),w,b,score))
    best=max(outcomes,key=lambda s:s[0])
    return best[2],best[3],{"kind":kind,"power":best[1][0],"ridge":best[1][1],"selection":best[4],
                            "trials":[{"power":a[1][0],"ridge":a[1][1],"macro_f1_7":a[0]} for a in outcomes]}


def correction_factors(fit,edit,w,b,obj,reg,keep):
    x=edit["k"]
    current=norm(edit["e"])@w+b
    top=current.argmax(1); gold=edit["y"]
    if obj=="mistakes":
        target=(EYE[gold]-EYE[top])*((current.max(1)-current[np.arange(len(top)),gold]+.25)[:,None]/2)
    elif obj=="residual":
        target=(EYE[gold]-current)
        target-=target.mean(1,keepdims=True)
    target*=rms(edit["e"])
    wei=cw(gold,.5)
    if keep:
        x=np.vstack([x,fit["k"]])
        target=np.vstack([target,np.zeros((len(fit["k"]),7))])
        wei=np.r_[wei,.35*cw(fit["y"],.5)]
    a,_=ridge(x,target,wei,reg,False)
    u=lin.pinv(w,rtol=1e-6,check_finite=False)
    return a,u


def scaled_score(d,w,b,a,u,gain,perm=False):
    keys=d["k"]
    change=(keys@(a[KEY_PERM] if perm else a))@u
    return norm(d["e"]+gain*change)@w+b


def compare(y,base,new):
    return {"changed":int(np.sum(base!=new)),"corrected":int(np.sum((base!=y)&(new==y))),
            "broken":int(np.sum((base==y)&(new!=y)))}


def h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def summary(a):
    return {"shape":list(a.shape),"dtype":str(a.dtype),"sha256":h(a),
            "frobenius":float(np.linalg.norm(a.astype(float)))}


def load_source():
    with safe_open(str(ROOT/"model/model.safetensors"),framework="numpy") as f:
        return f.get_tensor(SOURCE)


def main():
    pa=argparse.ArgumentParser()
    pa.add_argument("--stage",default="quick");pa.add_argument("--mode",default="context")
    pa.add_argument("--cap",type=int,default=384);pa.add_argument("--skip-verify",action="store_true")
    pa.add_argument("--common",action="store_true")
    z=pa.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","3")));torch.set_num_interop_threads(1)
    gs=groups(z.stage)
    sets={n:data(items,z.cap,z.mode) for n,items in gs.items() if n in ["fit","edit","select","confirm"]}
    if "natural_select" in gs and all(path_for(v,z.cap,z.mode).is_file() for v in gs["natural_select"]):
        # Prespecified selection includes both short and unfiltered cases where extracted.
        own={v["id"] for v in gs["select"]}
        extras=[v for v in gs["natural_select"] if v["id"] not in own]
        sets["select"]=data(gs["select"]+extras,z.cap,z.mode)
    if z.common:
        sets["common"]=data(groups("common")["common"],z.cap,z.mode)
    fit,edit,sel=sets["fit"],sets["edit"],sets["select"]
    w,b,winner=select_head(fit,sel,"e")
    native_w,native_b,native_selection=select_head(fit,sel,"native")
    candidates=[]
    for obj in ["mistakes","residual"]:
        for reg in [.02,.3,2.]:
            for keep in [False,True]:
                a,u=correction_factors(fit,edit,w,b,obj,reg,keep)
                for gain in [.25,.75,1.5,3.,6.]:
                    p=scaled_score(sel,w,b,a,u,gain).argmax(1)
                    result=metrics(sel["y"],p)
                    candidates.append((result["macro_f1_7"],gain,obj,reg,keep,a,u))
    cand=max(candidates,key=lambda c:(c[0],-c[1]))
    _,gain,obj,reg,keep,a,u=cand
    delta=(gain*(a@u).T).astype(np.float32)
    reference=load_source()
    changed=(reference+delta).astype(np.float32)
    diff=changed-reference
    original=summary(reference);updated=summary(changed)
    singular=lin.svdvals(lin.qr(a,mode="economic")[1]@lin.qr(u.T,mode="economic")[1].T)*gain
    base_report={"role_order":LABELS,"feature_version":ARCH,"stage":z.stage,"mode":z.mode,"token_cap":z.cap,
                 "case_split":"official train only, whole-case hash before role-balanced anchor samples on fit/edit; short selection/confirmation sampled label-blind",
                 "head":"sentence mean of final residual before native final RMSNorm, then sentence RMS normalization; unmodified native final readout is a separate control",
                 "natural_select_included":len(np.unique(sel["case"]))>len(gs["select"]),
                 "head_selection":winner,"native_head_selection":native_selection,
                 "readout_tensors":{"classifier.weight":summary(w.T.astype(np.float32)),"classifier.bias":summary(b.astype(np.float32))},
                 "internal_edit":{"parameter":SOURCE,"rank_at_1e-6_relative":int(np.sum(singular>singular[0]*1e-6)),
                                  "projection_formula":"write Delta = gain * pinv(W_head.T).T @ A.T; regularized delta-logit targets from separate edit-fit corpus",
                                  "selected":{"objective":obj,"ridge":reg,"preserve_fit_with_zero_targets":keep,"gain":gain},
                                  "selection_chooses_zero":winner["selection"]["macro_f1_7"]>=cand[0],
                                  "source":original,"modified":updated,"actual_fp32_delta":summary(diff),
                                  "raw_delta_singular_values":singular.tolist(),
                                  "changed_entries":int(np.count_nonzero(diff)),"max_abs_delta":float(abs(diff).max()),
                                  "relative_frobenius":float(np.linalg.norm(diff)/np.linalg.norm(reference)),
                                  "controls":"zero edit same head; shuffled input dimensions of the update with fixed seed 5611022 and same singular values"},
                 "trials":[{"macro_f1_7":i[0],"gain":i[1],"objective":i[2],"ridge":i[3],"preservation":i[4]} for i in candidates],
                 "results":{}}
    root=ROOT/f"write-{z.stage}-{z.mode}-{z.cap}"
    root.mkdir(exist_ok=True)
    save_file({SOURCE:changed,"classifier.weight":np.ascontiguousarray(w.T,dtype=np.float32),
               "classifier.bias":np.asarray(b,dtype=np.float32),"edit.keys_to_logits":np.asarray(a,dtype=np.float32),
               "edit.logits_to_hidden":np.asarray(u,dtype=np.float32),"edit.gain":np.asarray([gain],dtype=np.float32)},
              str(root/"edit.safetensors"), metadata={"base_revision":"545ac9d9c7aece444f014ada5a1e9818132b4d35","tensor":"fully materialized edited matrix, replace exactly"})
    for name,d in sets.items():
        y=d["y"];base=(norm(d["e"])@w+b).argmax(1)
        pred=scaled_score(d,w,b,a,u,gain).argmax(1)
        rnd=scaled_score(d,w,b,a,u,gain,True).argmax(1)
        nativ=(norm(d["native"])@native_w+native_b).argmax(1)
        base_report["results"][name]={"cases":len(np.unique(d["case"])),"tokens":int(d["tok"].sum()),
                  "head_only":metrics(y,base),"direct_write":metrics(y,pred),"shuffled_control":metrics(y,rnd),
                  "native_head_only":metrics(y,nativ),"vs_head":compare(y,base,pred)}
        npp=root/(name+"-predictions.npz")
        np.savez_compressed(npp,case=d["case"],csv_row=d["row"],sentence_index=d["ix"],char_start=d["start"],
                 char_end=d["end"],tokens=d["tok"],gold=y,head_only=base,direct_write=pred,shuffled_control=rnd,native_head=nativ,
                 edited_logits=scaled_score(d,w,b,a,u,gain).astype(np.float32),native_logits=(norm(d["native"])@native_w+native_b).astype(np.float32))
    save(root/"report.json",base_report)
    print("selection",base_report["internal_edit"]["selected"],"rank",base_report["internal_edit"]["rank_at_1e-6_relative"])
    for name,score in base_report["results"].items():
        print(name,score["cases"],score["head_only"]["rows"],{k:round(score[k]["macro_f1_7"],5) for k in
              ["native_head_only","head_only","direct_write","shuffled_control"]},score["vs_head"],flush=True)
    if not z.skip_verify:
        enc=Encoder()
        with torch.no_grad():
            mat=enc.model.layers[23].mlp.down_proj.weight
            assert h(mat.detach().cpu().numpy())==h(reference)
            mat.copy_(torch.from_numpy(changed))
            assert h(mat.detach().cpu().numpy())==h(changed)
        item=gs["fit"][0]
        pre=read(item,z.cap,z.mode)
        physical=enc.run(item,z.cap,z.mode)
        predicted=pre["e"].astype(float)+pre["k"].astype(float)@diff.astype(float).T
        error=physical["e"]-predicted
        observed_y=(norm(physical["e"])@w+b).argmax(1)
        predicted_y=(norm(predicted)@w+b).argmax(1)
        base_report["physical_native_pytorch_replay"]={"case":item["id"],"rows":pre["row"].tolist(),
                   "tokens":int(pre["tok"].sum()),"fresh_forward_after_copy_parameter":True,
                   "tensor_sha256_after_runtime_write":h(mat.detach().cpu().numpy()),
                   "maximum_hidden_error":float(np.max(np.abs(error))),
                   "relative_hidden_l2_error":float(np.linalg.norm(error)/np.linalg.norm(physical["e"])),
                   "fresh_argmax":observed_y.tolist(),"algebraic_argmax":predicted_y.tolist(),
                   "agreement":float(np.mean(observed_y==predicted_y))}
        save(root/"report.json",base_report)
        print("physical",base_report["physical_native_pytorch_replay"],flush=True)


if __name__=="__main__":
    main()
