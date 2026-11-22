"""Write the existing vocabulary prediction matrix or its preceding native nonlinear projection."""
import argparse
import os
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file
import scipy.linalg as la
from scipy.special import expit
import torch
from torch import nn
from torch.nn import functional as F

from extract import groups
from prepare import ROOT, LABELS, save
from study import cw, data, metrics, compare, ridge, summary, h
from joint import join

# Registered before any label comparison. Each role has one native vocabulary coordinate.
WORDS = [" none"," facts"," issue"," applicant"," respondent"," reasoning"," decision"]
ROWS = [7182,5307,2843,25291,45932,10115,3062]
MIX = np.random.default_rng(1122561).permutation(768)


def gelu(x):
    return .5*x*(1+np.tanh(.7978845608028654*(x+.044715*x**3)))


def deriv(x):
    t=np.tanh(.7978845608028654*(x+.044715*x**3))
    return .5+.5*t+.5*x*(1-t*t)*.7978845608028654*(1+.134145*x*x)


def layer(z,w):
    return gelu(z@w.T)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stage",default="quick");p.add_argument("--cap",type=int,default=384)
    p.add_argument("--mode",default="context");p.add_argument("--common",action="store_true")
    opts=p.parse_args();torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","1")));torch.set_num_interop_threads(1)
    with safe_open(str(ROOT/"model/model.safetensors"),framework="numpy") as ck:
        one=ck.get_tensor("head.linear1.weight").astype(float)
        base=ck.get_tensor("head.linear2.weight")
        C=ck.get_tensor("cls.weight")[ROWS].copy()
    two=base.astype(float);C=C.astype(float)
    g=groups(opts.stage)
    ds={n:data(g[n],opts.cap,opts.mode) for n in ["fit","edit","select","confirm"]}
    if opts.common:
        ds["common"]=data(groups("common")["common"],opts.cap,opts.mode)
    ds["fit_edit"]=join(ds["fit"],ds["edit"])
    hold=[n for n in ds if n not in ["fit","edit"]]
    keys={n:layer(ds[n]["native"],one) for n in ds}
    hidden={n:layer(x,two) for n,x in keys.items()}
    scores={n:v@C.T for n,v in hidden.items()}
    train=ds["fit_edit"]
    y=train["y"];target=np.eye(7)[y]
    kt=keys["fit_edit"];ht=hidden["fit_edit"];orig=scores["fit_edit"]
    ys=ds["select"]["y"]
    # Baseline labels from published tokens; label-specific mean removal from fit only.
    intercept=-orig.mean(0)
    headopt=None;internal=None;grid=[]
    for powr in [0.,.7]:
        wei=cw(y,powr)
        for reg in [.03,.3,1.5]:
            # Directly replace precisely the seven original vocabulary rows and retain an external 7-bias.
            hw,b=ridge(ht,target,wei,reg,True)
            val=metrics(ys,(hidden["select"]@hw+b).argmax(1))["macro_f1_7"]
            if headopt is None or val>headopt[0]:headopt=(val,hw,b,powr,reg)
            # Decoder fixed at the seven original rows. Solve an additive write through a mean native GELU Jacobian.
            native=orig+intercept
            slope=deriv(kt@two.T).mean(0)
            jac=C*slope[None,:]
            lift=la.pinv(jac.T,rtol=1e-6,check_finite=False)
            for margin in [1.,4.]:
                goal=(target-1/7)*margin-(native-native.mean(1,keepdims=True))
                a,_=ridge(kt,goal,wei,reg,False)
                for gain in [.5,1.,1.5]:
                    # Avoid multiplying large dense matrices for each trial. GELU is still evaluated exactly per sentence.
                    hs=gelu(keys["select"]@two.T+gain*(keys["select"]@a)@lift)
                    ss=hs@C.T+intercept
                    m=metrics(ys,ss.argmax(1))
                    grid.append([powr,reg,margin,gain,m["macro_f1_7"]])
                    if internal is None or m["macro_f1_7"]>internal[0]:
                        internal=(m["macro_f1_7"],a,lift,powr,reg,margin,gain)
    val,hw,b,powr,reg=headopt
    _,a,lift,ip,ir,margin,gain=internal
    delta=(gain*(a@lift).T).astype(np.float32)
    changed=(base+delta).astype(np.float32)
    random=(base+delta[:,MIX]).astype(np.float32)
    singular=la.svdvals(la.qr(a,mode="economic")[1]@la.qr(lift.T,mode="economic")[1].T)*gain
    folder=ROOT/f"mlm-{opts.stage}-{opts.mode}-{opts.cap}";folder.mkdir(exist_ok=True)
    patch={"head.linear2.weight":changed,"seven_output_rows.original":np.ascontiguousarray(C,dtype=np.float32),
           "seven_output_rows.replacement":np.ascontiguousarray(hw.T,dtype=np.float32),
           "seven_output_rows.ids":np.array(ROWS,dtype=np.int64),
           "decoder_baseline.bias":np.asarray(intercept,dtype=np.float32),"decoder_fitted.bias":np.asarray(b,dtype=np.float32)}
    save_file(patch,str(folder/"writes.safetensors"),metadata={"revision":"545ac9d9c7aece444f014ada5a1e9818132b4d35",
                 "internal":"head.linear2.weight is the complete edited matrix; the separately fitted cls row values are an independent comparator"})
    actual=changed-base
    rec={"role_order":LABELS,"stage":opts.stage,"seven_original_decoder_tokens":list(zip(WORDS,ROWS)),
         "method":"Pool native final post-token-RMSNorm hidden vectors over the sentence, apply published two GELU projections, then use seven chosen original output vocabulary rows. This custom sentence readout does not predict a masked token.",
         "case_protocol":"fit and edit case groups combined for all supervised controls. Selection, confirmation, and optional shared validation are disjoint.",
         "decoder_row_edit":{"parameter":"cls.weight","full_shape":[49152,768],"old_seven_rows":summary(C.astype(np.float32)),
                             "new_seven_rows":summary(hw.T.astype(np.float32)),"changed_entries":int(np.count_nonzero(hw.T.astype(np.float32)!=C.astype(np.float32))),
                             "bias_shape":[7],"selected":[powr,reg]},
         "internal_projection_write":{"parameter":"head.linear2.weight","source":summary(base),"updated":summary(changed),
              "changed_entries":int(np.count_nonzero(actual)),"delta":summary(actual),"rank_relative_1e-6":int(np.sum(singular>singular[0]*1e-6)),
              "relative_frobenius":float(np.linalg.norm(actual)/np.linalg.norm(base)),"selected":[ip,ir,margin,gain],"selection_grid":grid},
         "scores":{}}
    for name in hold:
        x=keys[name]
        raw=scores[name];y=ds[name]["y"]
        targ=layer(x,changed.astype(float))@C.T+intercept
        null=layer(x,random.astype(float))@C.T+intercept
        hs=hidden[name]@hw+b
        frozen=raw+intercept
        rec["scores"][name]={k:metrics(y,v.argmax(1)) for k,v in [("original_verbalizer",raw),("calibrated_verbalizer",frozen),
                       ("decoder_row_replacement",hs),("preceding_projection_write",targ),("shuffled_projection_write",null)]}
        rec["scores"][name]["changes_vs_calibrated"]=compare(y,frozen.argmax(1),targ.argmax(1))
        np.savez_compressed(folder/f"{name}.npz",csv_row=ds[name]["row"],case=ds[name]["case"],sentence_index=ds[name]["ix"],
                  char_start=ds[name]["start"],char_end=ds[name]["end"],gold=y,decoder_logits=hs.astype(np.float32),projection_logits=targ.astype(np.float32))
        print(name,[(key,round(q["macro_f1_7"],4)) for key,q in rec["scores"][name].items() if "macro_f1_7" in q],flush=True)
    # Physically install both original checkpoint projections, then install the selected real edit.
    first=nn.Linear(768,768,bias=False);second=nn.Linear(768,768,bias=False)
    with torch.no_grad():
        first.weight.copy_(torch.from_numpy(one.astype(np.float32)));second.weight.copy_(torch.from_numpy(base))
        assert h(second.weight.numpy())==h(base)
        second.weight.copy_(torch.from_numpy(changed))
        sample=ds["confirm"]["native"][:8].astype(np.float32)
        tested=(F.gelu(second(F.gelu(first(torch.from_numpy(sample)),approximate="tanh")),approximate="tanh")
                @torch.from_numpy(C.T.astype(np.float32))+torch.from_numpy(intercept.astype(np.float32))).numpy()
    reference=layer(keys["confirm"][:len(sample)],changed.astype(float))@C.T+intercept
    rec["physical_runtime_check"]={"installed_parameter_sha256":h(second.weight.detach().numpy()),
         "rows":ds["confirm"]["row"][:8].tolist(),"max_abs_logit_difference":float(np.max(abs(tested-reference))),
         "same_argmax":bool(np.array_equal(tested.argmax(1),reference.argmax(1)))}
    save(folder/"report.json",rec)
if __name__=="__main__":main()
