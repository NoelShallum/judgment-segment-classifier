"""Fit independent analytic native classifier writes and frozen role-sequence controls."""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import time

import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file, save_file
from scipy.linalg import cho_factor, cho_solve

from encode import HERE, VIEWS
from native import read, save_arrays, sha, write

WIDTH = 768
VOCAB = 30522
EXTRA = {"native": np.arange(WIDTH), "position": np.arange(WIDTH + 9),
         "lexical": np.r_[np.arange(WIDTH), np.arange(WIDTH + 9, WIDTH + 16)],
         "position_lexical": np.arange(WIDTH + 16)}


def loader():
    src = HERE / "viterbi.cpp"
    target = HERE / ("viterbi-" + sha(src)[:12] + ".so")
    if not target.exists():
        tmp = target.with_suffix(".part")
        subprocess.run(["g++","-std=c++17","-O3","-shared","-fPIC","-ffp-contract=off",str(src),"-o",str(tmp)], check=True)
        tmp.replace(target)
    fn = ctypes.CDLL(str(target)).role_viterbi
    ptr = np.ctypeslib.ndpointer
    fn.argtypes = [ptr(np.float32, flags="C_CONTIGUOUS"), ptr(np.float32, flags="C_CONTIGUOUS"),
                   ptr(np.int64, flags="C_CONTIGUOUS"), ctypes.c_size_t, ptr(np.int8, flags="C_CONTIGUOUS")]
    fn.restype = None

    def decode(scores, tr, ends, strength):
        if strength == 0.:
            return scores.argmax(1).astype(np.int8)
        out = np.empty(len(scores), np.int8)
        fn(np.ascontiguousarray(scores,np.float32), np.ascontiguousarray(strength*tr,np.float32),
           np.ascontiguousarray(ends,np.int64), len(ends)-1, out)
        return out

    return decode


def confusion(y, pred):
    return np.bincount(y.astype(np.int64)*7 + pred, minlength=49).reshape(7,7)


def measures(cm):
    hits = np.diagonal(cm,axis1=-2,axis2=-1)
    ratio = 2 * hits / np.maximum(cm.sum(-2) + cm.sum(-1),1)
    return ratio, ratio.mean(-1), hits.sum(-1) / cm.sum(axis=(-2,-1))


def metrics(cm):
    seven, f1, acc = measures(cm)
    return dict(macro_f1=float(f1), accuracy=float(acc), role_f1=seven.tolist(), labeled=cm.sum(1).tolist())


def span_counts(y, pred):
    def spans(sequence):
        a = np.flatnonzero(np.r_[True, sequence[1:] != sequence[:-1]])
        return set(zip(a,np.r_[a[1:],len(sequence)],sequence[a]))
    a, b = spans(y), spans(pred)
    return [len(a & b),len(b),len(a)]


def gather(directory, features, group, take=None, must_t4=True):
    cfg = read(directory / "protocol.json")
    keys = cfg["groups"][group][:take]
    bank = []
    for key in keys:
        facts = directory / "cases" / (key + ".npz")
        note = read(features / (key + ".json"))
        src = features / (key + ".npz")
        assert note["case_sha256"] == sha(facts) and note["features_sha256"] == sha(src)
        assert note["model"] == cfg["source_model"]["sha256"]
        assert note["cost"]["ordered"] == note["cost"]["shuffled"]
        if must_t4:
            backend = note["backend"]
            assert backend["cuda"] and len(backend["devices"]) == 1 and "T4" in backend["devices"][0]
        with np.load(facts,allow_pickle=False) as f, np.load(src,allow_pickle=False) as z:
            row = {name:f[name] for name in ("tokens","ends","rows","gold","incumbent")}
            row.update({view:z[view] for view in (*VIEWS,"source_scores")})
        assert all(row[k].shape == (len(row["gold"]),2,WIDTH) for k in VIEWS)
        assert len(row["tokens"]) == sum(np.diff(row["ends"]))
        row["id"] = key
        bank.append(row)
    return bank


def join(bank, key):
    return np.concatenate([x[key] for x in bank])


def offsets(bank):
    return np.r_[0,np.cumsum([len(row["gold"]) for row in bank])]


def counts(row):
    lengths = np.diff(row["ends"])
    labels = np.repeat(row["gold"],lengths)
    weights = np.repeat(1./lengths,lengths)
    return np.bincount(7*row["tokens"] + labels,weights=weights,minlength=VOCAB*7).reshape(VOCAB,7)


def lexicon(totals):
    totals = np.maximum(totals,0.)
    prior = (totals.sum(0)+1) / (totals.sum()+7)
    estimate = (totals + 12*prior) / (totals.sum(1,keepdims=True)+12)
    return np.clip(np.log(estimate / prior),-2.5,2.5).astype(np.float32)


def position(row):
    lengths = np.diff(row["ends"])
    n = len(lengths)
    i = np.arange(n,dtype=np.float32)
    p = (i+.5)/n
    log_size = np.full(n,np.log1p(n)/5,dtype=np.float32)
    return np.column_stack((p,p**2,np.sqrt(p),np.sqrt(1-p),1/(i+1),1/(n-i),
                            log_size,np.log1p(lengths)/5,lengths<=12)).astype(np.float32)


def auxiliary(bank, total, leave_case_out):
    data = []
    for row in bank:
        table = lexicon(total - counts(row)) if leave_case_out else lexicon(total)
        pooling = np.add.reduceat(table[row["tokens"]],row["ends"][:-1],axis=0) / np.diff(row["ends"])[:,None]
        data.append(np.column_stack((position(row),pooling)))
    return np.concatenate(data).astype(np.float32)


def transitions(bank):
    y = join(bank,"gold")
    prior = (np.bincount(y,minlength=7)+7)/(len(y)+49)
    pair = np.ones((7,7))*20*prior[None,:]
    for row in bank:
        g = row["gold"]
        pair += np.bincount(7*g[:-1].astype(np.int64)+g[1:],minlength=49).reshape(7,7)
    result = np.zeros((10,10),np.float32)
    result[3:,3:] = np.clip(np.log(pair / pair.sum(1,keepdims=True)/prior),-2.,2.)
    return result


def values(bank, view, factor, extra):
    x = join(bank,view)
    pooled = (1-factor)*x[:,0] + factor*x[:,1]
    return np.column_stack((pooled,extra))


def fit(d, f, output, checkpoint, take=None, must_t4=True):
    a = gather(d,f,"fit",take,must_t4)
    b = gather(d,f,"select",take,must_t4)
    y = join(a,"gold")
    ysel = join(b,"gold")
    ends = offsets(b)
    prior = (np.bincount(y,minlength=7)+1)/(len(y)+7)
    count = sum(counts(row) for row in a)
    train_aux = auxiliary(a,count,True)
    select_aux = auxiliary(b,count,False)
    target = np.eye(7,dtype=np.float32)[y]
    tr = transitions(a)
    decode = loader()
    saved = {"lexical.counts":count,"crf.transitions":tr}
    settings = {}
    for view in VIEWS:
        best = {}
        stages = []
        for fraction in (0.,.5,1.):
            x = values(a,view,fraction,train_aux)
            v = values(b,view,fraction,select_aux)
            mu = x.mean(0,dtype=np.float64)
            scale = 1/np.maximum(x.std(0,dtype=np.float64),.05)
            u = ((x-mu)*scale).astype(np.float32)
            for rarity in (0.,.35):
                mass = (prior[y]**-rarity).astype(np.float32)
                delta_mean = (mass @ u)/mass.sum()
                centered = u-delta_mean
                w = centered * mass[:,None]
                mean_target = (mass @ target)/mass.sum()
                gram = (w.T @ centered).astype(np.float64)
                rhs = (w.T @ (target - mean_target)).astype(np.float64)
                intercept_mean = mu + delta_mean/scale
                for method, dims in EXTRA.items():
                    for reg in (.03,.30):
                        selected = gram[np.ix_(dims,dims)]
                        selected.flat[::len(dims)+1] += reg*np.trace(selected)/len(dims)
                        raw = cho_solve(cho_factor(selected,check_finite=False),rhs[dims],check_finite=False)
                        coef = np.ascontiguousarray(raw * scale[dims,None],np.float32)
                        bias = (mean_target - intercept_mean[dims] @ coef).astype(np.float32)
                        logits = v[:,dims] @ coef + bias
                        for step in (0.,.08,.18,.36):
                            pred = decode(logits,tr,ends,step)
                            r = metrics(confusion(ysel,pred))
                            rule = dict(fraction_last=fraction, class_power=rarity, ridge=reg, transition=step,
                                        fit_feature_style=method)
                            score = r["macro_f1"],r["accuracy"]
                            stages.append(dict(**rule,macro_f1=score[0],accuracy=score[1]))
                            if method not in best or score > best[method][0]:
                                best[method] = (score,rule,coef,bias)
        settings[view] = {"grid":stages,"selected":{}}
        for name,(_,recipe,w,bias) in best.items():
            tag = view+"."+name
            saved[tag+".weight"] = w
            saved[tag+".bias"] = bias
            settings[view]["selected"][name] = recipe | {"select_macro_f1":best[name][0][0],"select_accuracy":best[name][0][1]}
        print("selected",view,settings[view]["selected"],flush=True)
    output.mkdir(parents=True,exist_ok=True)
    path = output / "analytic-heads.safetensors"
    save_file({key:np.ascontiguousarray(value) for key,value in saved.items()},str(path.with_suffix(".part")))
    path.with_suffix(".part").replace(path)
    reopened = load_file(str(path))
    for key,value in saved.items():
        np.testing.assert_array_equal(reopened[key],value)
    physical = patch(checkpoint,output,saved["ordered.native.weight"].T,saved["ordered.native.bias"])
    write(output / "freeze.json",dict(
        settings=settings,base_checkpoint_sha256=sha(checkpoint), plan_sha256=sha(d/"protocol.json"),
        weights_sha256=sha(path), fit_cases=[row["id"] for row in a], select_cases=[row["id"] for row in b],
        fit_rows=len(y),select_rows=len(ysel),parent_classifier=physical,gradient_steps=0,
        serialized_weights_roundtrip=True,
        selection="Each context and feature style has an independently optimized 7-way classifier with identical grids. All common-case outcomes are computed after this freeze."))


def patch(source, output, weight, bias):
    requested = {"classifier.weight":np.ascontiguousarray(weight,np.float32),"classifier.bias":np.ascontiguousarray(bias,np.float32)}
    assert requested["classifier.weight"].shape == (7,WIDTH) and requested["classifier.bias"].shape == (7,)
    dest = output / "ordered-native-head.safetensors"
    save_file(requested,str(dest))
    with source.open("rb") as f:
        header_size = struct.unpack("<Q",f.read(8))[0]
        header = json.loads(f.read(header_size))
    clone = output.parent / "ordered-native-full.safetensors"
    tmp = clone.with_suffix(".part")
    shutil.copyfile(source,tmp)
    details = {}
    with safe_open(str(source),framework="np") as orig, tmp.open("r+b") as copy:
        for key,values in requested.items():
            entry = header[key]
            assert entry["dtype"] == "F32" and tuple(entry["shape"]) == values.shape
            a,b = entry["data_offsets"]
            assert b-a == values.nbytes
            copy.seek(8+header_size+a)
            copy.write(values.tobytes())
            old = orig.get_tensor(key)
            details[key] = dict(shape=list(values.shape), changed_values=int(np.sum(old != values)),
                change_norm=float(np.linalg.norm(values-old)),frozen_dtype="float32")
    with safe_open(str(tmp),framework="np") as out:
        for key, value in requested.items():
            np.testing.assert_array_equal(out.get_tensor(key),value)
    tmp.replace(clone)
    return dict(tensors=details,patch_sha256=sha(dest),full_checkpoint_sha256=sha(clone),
                edit_rule="The source file was cloned byte-for-byte. Only the recorded safetensors data ranges for classifier.weight and classifier.bias were overwritten.")


def score_split(output, bank, frozen, params, decode):
    y = join(bank,"gold")
    ends = offsets(bank)
    extra = auxiliary(bank,params["lexical.counts"],False)
    p = {"source_cls":join(bank,"source_scores").argmax(1).astype(np.int8),"current_20pct":join(bank,"incumbent")}
    assert len(p["current_20pct"]) == len(y)
    for view in VIEWS:
        for mode, chosen in frozen["settings"][view]["selected"].items():
            name = view+"."+mode
            x = values(bank,view,chosen["fraction_last"],extra)
            pred = x[:,EXTRA[mode]] @ params[name+".weight"] + params[name+".bias"]
            p[name] = decode(pred,params["crf.transitions"],ends,chosen["transition"])
    by_case = {name:np.stack([confusion(y[a:b],z[a:b]) for a,b in zip(ends,ends[1:])]) for name,z in p.items()}
    span = {name:np.stack([span_counts(y[a:b],z[a:b]) for a,b in zip(ends,ends[1:])]) for name,z in p.items()}
    rows = {}
    for key in p:
        h,n,g = span[key].sum(0)
        rows[key] = metrics(by_case[key].sum(0)) | dict(exact_span_f1=float(2*h/(n+g)),exact_span_counts=[int(h),int(n),int(g)])
    pairlist = [(f"ordered.{mode}",f"{view}.{mode}") for mode in EXTRA for view in ("isolated","shuffled")]
    pairlist += [("ordered."+kind,"current_20pct") for kind in EXTRA]
    pairlist += [("ordered."+kind,"ordered.native") for kind in EXTRA if kind!="native"]
    draw = np.random.default_rng(26622).integers(len(bank),size=(2500,len(bank)))
    paired = {}
    for left,right in pairlist:
        a,b = (measures(by_case[k][draw].sum(1))[1] for k in (left,right))
        x,z = (span[k][draw].sum(1) for k in (left,right))
        g = 2*x[:,0]/(x[:,1]+x[:,2]) - 2*z[:,0]/(z[:,1]+z[:,2])
        paired[left+" vs "+right] = dict(macro_f1_ci95=np.quantile(a-b,[.025,.975]).tolist(),
                                         exact_span_f1_ci95=np.quantile(g,[.025,.975]).tolist())
    path = output / f"shared{len(bank)}-predictions.npz"
    save_arrays(path, keys=[row["id"] for row in bank], offsets=ends, csv_rows=join(bank,"rows"), gold=y, **p)
    return dict(judgments=len(bank), rows=len(y), scores=rows, paired=paired, prediction_sha256=sha(path))


def evaluate(d,f,output,take=None,must_t4=True):
    frozen = read(output / "freeze.json")
    assert frozen["serialized_weights_roundtrip"]
    assert frozen["plan_sha256"] == sha(d/"protocol.json")
    path = output / "analytic-heads.safetensors"
    assert frozen["weights_sha256"] == sha(path)
    params = load_file(str(path))
    b = gather(d,f,"shared",take,must_t4)
    decode = loader()
    for selection in (b[:16],b) if len(b)>16 else (b,):
        result = score_split(output,selection,frozen,params,decode)
        write(output / f"shared{len(selection)}.json",result | {"freeze_sha256":sha(output/"freeze.json"),
              "official_test_evaluated":False,"criterion":"All instances in the exact same case cohort at the original sentence cuts."})
        print("shared",len(selection),result["rows"],
              {k:round(v["macro_f1"],5) for k,v in result["scores"].items()},flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase",choices=("fit","evaluate","all"))
    cli.add_argument("directory",type=Path)
    cli.add_argument("checkpoint",type=Path)
    cli.add_argument("features",type=Path)
    cli.add_argument("output",type=Path)
    cli.add_argument("--smoke",type=int)
    args = cli.parse_args()
    if args.phase in ("fit","all"):
        fit(args.directory,args.features,args.output,args.checkpoint,args.smoke,args.smoke is None)
    if args.phase in ("evaluate","all"):
        evaluate(args.directory,args.features,args.output,args.smoke,args.smoke is None)


if __name__ == "__main__":
    main()
