"""A second physical weight write: solve a joint linear predictor and factor it into backbone + head."""
import argparse
import csv
import json
import os

import numpy as np
import scipy.linalg as lin
import torch
from safetensors.numpy import save_file

from extract import Encoder, groups, path_for, read
from prepare import LABELS, ROOT, SPLIT, save
from study import KEY_PERM, SOURCE, compare, cw, data, h, load_source, metrics, ridge, summary


def position(d):
    ix=d["ix"].astype(float)
    n=d["ncase"].astype(float)
    p=(ix+.5)/n
    centers=np.linspace(0,1,12)
    knots=np.exp(-.5*((p[:,None]-centers)/.065)**2)
    left=np.exp(-ix[:,None]/np.array([1.,3.,9.]))
    right=np.exp(-(n-ix-1)[:,None]/np.array([1.,3.,9.]))
    return np.column_stack([knots,left,right,np.log1p(d["tok"])])


def scale(x):
    # Scale globally so keys retain the exact linear map required for materialization.
    return max(float(np.sqrt(np.mean(np.sum((x-x.mean(axis=0))**2,axis=1)))),1e-8)


def transition(train_ids):
    c=np.ones((7,7))*.5
    with (SPLIT/"train.csv").open(encoding="utf-8-sig",newline="") as fh:
        r=csv.DictReader(fh);last_c=None;last_y=None
        for row in r:
            cid=row["Index"]
            if cid not in train_ids or not row["Text"].strip():continue
            role=LABELS.index(row["Label"])
            if cid==last_c:c[last_y,role]+=1
            last_y=role;last_c=cid
    return np.log(c/c.sum(1,keepdims=True))


def decode(d,s,trans,weight):
    if weight==0:return s.argmax(1)
    p=np.zeros(len(s),int)
    for cid in np.unique(d["case"]):
        idx=np.flatnonzero(d["case"]==cid);idx=idx[np.argsort(d["ix"][idx])]
        cuts=np.flatnonzero(np.diff(d["ix"][idx])>1)+1
        for seq in np.split(idx,cuts):
            vals=s[seq[0]].copy();back=[]
            for row in seq[1:]:
                options=vals[:,None]+weight*trans
                back.append(options.argmax(0))
                vals=options.max(0)+s[row]
            state=int(vals.argmax());p[seq[-1]]=state
            for row,bt in zip(seq[-2::-1],reversed(back)):
                state=int(bt[state]);p[row]=state
    return p


def join(*ds):
    return {k:np.concatenate([d[k] for d in ds]) for k in ds[0]}


def learn(fit,sel,mode,trans,weights):
    x=fit["e"];kx=fit["k"]
    sx=scale(x);sk=scale(kx);po=position(fit);sp=scale(po)
    t=np.eye(7)[fit["y"]]
    best=None;detailed=[]
    for cpos in (weights if mode=="position" else [0.]):
        for ck in (0.,.5,1.5,4.):
            matrix=np.column_stack([x/sx,ck*kx/sk,cpos*po/sp])
            tr=matrix.shape[1]
            for reg in (.03,.3,1.5):
                for powr in (0.,.65):
                    w,b=ridge(matrix,t,cw(fit["y"],powr),reg,True)
                    we=w[:768]/sx;wk=w[768:1792]*ck/sk;wp=w[1792:]*cpos/sp
                    added=sel["k"]@wk
                    s=sel["e"]@we+position(sel)@wp+b+added
                    for tw in ((0.,.05,.15,.35) if mode=="position" else [0.]):
                        p=decode(sel,s,trans,tw)
                        score=metrics(sel["y"],p)["macro_f1_7"]
                        config=(score,cpos,ck,reg,powr,tw)
                        detailed.append(list(config))
                        if best is None or score>best[0][0]:
                            best=(config,we,wk,wp,b)
    return best,detailed


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",default="quick");parser.add_argument("--mode",default="context")
    parser.add_argument("--cap",type=int,default=384);parser.add_argument("--common",action="store_true")
    parser.add_argument("--verify",action="store_true")
    z=parser.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","1")));torch.set_num_interop_threads(1)
    gs=groups(z.stage)
    sets={k:data(v,z.cap,z.mode) for k,v in gs.items() if k in ["fit","edit","select","confirm"]}
    if "natural_select" in gs and all(path_for(v,z.cap,z.mode).is_file() for v in gs["natural_select"]):
        own={v["id"] for v in gs["select"]}
        sets["select"]=data(gs["select"]+[v for v in gs["natural_select"] if v["id"] not in own],z.cap,z.mode)
    if z.common:
        sets["common"]=data(groups("common")["common"],z.cap,z.mode)
    fit=join(sets["fit"],sets["edit"]);sel=sets["select"]
    trans=transition(set(fit["case"]))
    folder=ROOT/f"joint-{z.stage}-{z.mode}-{z.cap}";folder.mkdir(exist_ok=True)
    original=load_source()
    rep={"roles":LABELS,"target":SOURCE,"stage":z.stage,"mode":z.mode,"cap":z.cap,
         "method":"joint weighted ridge solves on identical fit+edit cases for all control conditions; conditional class score from last MLP keys is factored into a real pretrained down projection",
         "partition":"official train fit+edit group union, select groups for regularization, decoder, and release selection; frozen confirmation neither fitted nor selected",
         "subsets":{k:{"cases":len(np.unique(d['case'])),"rows":len(d['y'])} for k,d in sets.items()},
         "experiments":{}}
    for mode in ["text","position"]:
        top,all_trials=learn(fit,sel,mode,trans,[.5,1.5,4.])
        # Independently choose the best nonzero-key write even if the release selection chooses no edit.
        fkeys=[v for v in all_trials if v[2]>0]
        ncfg=max(fkeys,key=lambda v:v[0])
        hcfg=max([v for v in all_trials if v[2]==0],key=lambda v:v[0])
        def refit(cfg):
            _,cp,ck,reg,powr,tw=cfg
            e,k,po=fit["e"],fit["k"],position(fit)
            se,sk,sp=scale(e),scale(k),scale(po)
            mat=np.column_stack([e/se,ck*k/sk,cp*po/sp])
            w,b=ridge(mat,np.eye(7)[fit["y"]],cw(fit["y"],powr),reg,True)
            return w[:768]/se,w[768:1792]*ck/sk,w[1792:]*cp/sp,b,tw
        we,wk,wp,b,tw=refit(ncfg)
        wh,_,ph,bh,th=refit(hcfg)
        lift=lin.pinv(we,rtol=1e-6,check_finite=False)
        delta=(wk@lift).T.astype(np.float32)
        modified=(original+delta).astype(np.float32);actual=modified-original
        error=float(np.linalg.norm(actual.T@we-wk)/max(np.linalg.norm(wk),1e-12))
        singular=lin.svdvals(lin.qr(wk,mode="economic")[1]@lin.qr(lift.T,mode="economic")[1].T)
        rec={"selection_prefers_zero":top[0][2]==0,"best_configs":{"best_any":list(top[0]),"independent_head_only":hcfg,"best_internal":ncfg},
             "config_order":["selection macro F1","case position scale","MLP key scale","ridge penalty","class weighting exponent","transition weight"],
             "all_selection_scores":all_trials,"tensor":{"name":SOURCE,"source":summary(original),"updated":summary(modified),
             "changed_entries":int(np.count_nonzero(actual)),"delta":summary(actual),
             "relative_change":float(np.linalg.norm(actual)/np.linalg.norm(original)),
             "rank_above_relative_1e_6":int(np.sum(singular>singular[0]*1e-6)),
             "relative_logit_factorization_error":error},"results":{}}
        for n,d in sets.items():
            base=d["e"]@we+position(d)@wp+b
            exact=base+d["k"]@wk
            head_matched=decode(d,base,trans,tw)
            pe=decode(d,exact,trans,tw)
            heads_opt=decode(d,d["e"]@wh+position(d)@ph+bh,trans,th)
            shuffle=decode(d,base+d["k"]@wk[KEY_PERM],trans,tw)
            rec["results"][n]={key:metrics(d["y"],v) for key,v in [("optimized_zero",heads_opt),
                     ("matched_zero",head_matched),("direct",pe),("shuffled",shuffle)]}
            rec["results"][n]["vs_optimized_zero"]=compare(d["y"],heads_opt,pe)
            np.savez_compressed(folder/f"{mode}-{n}.npz",case=d["case"],csv_row=d["row"],sentence_index=d["ix"],
                                char_start=d["start"],char_end=d["end"],gold=d["y"],direct=pe,
                                matched_zero=head_matched,optimized_zero=heads_opt,shuffled=shuffle,emissions=exact.astype(np.float32))
        ckpt={SOURCE:modified,"classifier.weight":np.ascontiguousarray(we.T,dtype=np.float32),
              "classifier.bias":np.asarray(b,dtype=np.float32),"position.weight":np.ascontiguousarray(wp.T,dtype=np.float32),
              "transition.log_probabilities":np.asarray(trans,dtype=np.float32),"transition.gain":np.array([tw],np.float32)}
        save_file(ckpt,str(folder/f"{mode}.safetensors"),metadata={"base_revision":"545ac9d9c7aece444f014ada5a1e9818132b4d35","pooling":"unscaled last pre-final-norm residual sentence mean"})
        rec["checkpoint_tensors"]={key:summary(v) for key,v in ckpt.items() if key!=SOURCE}
        if z.verify:
            if mode=="text":enc=Encoder()
            with torch.no_grad():enc.model.layers[23].mlp.down_proj.weight.copy_(torch.from_numpy(modified))
            item=gs["fit"][0];d=data([item],z.cap,z.mode);raw=enc.run(item,z.cap,z.mode)
            predicted=(d["e"]+d["k"]@actual.astype(np.float64).T)@we+position(d)@wp+b
            real=raw["e"].astype(float)@we+position(d)@wp+b
            rec["fresh_forward"]={"case":item["id"],"csv_rows":d['row'].tolist(),"max_abs_logit_difference":float(np.max(np.abs(predicted-real))),
                                 "same_argmax":bool(np.array_equal(predicted.argmax(1),real.argmax(1)))}
        rep["experiments"][mode]=rec
        save(folder/"report.json",rep)
        for n,res in rec["results"].items():
            if n in ["select","confirm","common"]:
                print(mode,n,[round(res[k]["macro_f1_7"],4) for k in ["optimized_zero","matched_zero","direct","shuffled"]],ncfg,flush=True)


if __name__=="__main__":main()
