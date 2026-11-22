"""Write two native classifier matrices jointly from cached hierarchy and packed BERT states."""

import argparse
from itertools import product
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from safetensors.numpy import load_file, save_file
from scipy.linalg import cho_factor, cho_solve


def evidence(data, features, hier, group):
    from fit import gather, join
    from native import read, sha

    banks = gather(data,features,group)
    manifest = read(hier/"receipts.json")
    lock = read(hier/"hybrid-protocol.json")
    context = []
    for a in banks:
        key = a["id"]
        item = manifest["cases"][key]
        src = hier/"states"/(key+".npz")
        assert item["source_case_sha256"] == sha(data/"cases"/(key+".npz"))
        assert item["states_file_sha256"] == sha(src)
        assert item["donor_sha256"] == lock["native_hierarchy_sha256"]
        assert item["protocol_sha256"] == sha(hier/"hybrid-protocol.json")
        with np.load(src,allow_pickle=False) as f:
            np.testing.assert_array_equal(f["rows"],a["rows"])
            h = f["context"]
        assert h.shape == (len(a["gold"]),200) and np.isfinite(h).all() and h.dtype == np.float32
        context.append(h)
    assert len(banks) == len(lock["groups"][group]) and len(join(banks,"gold")) == lock["rows"][group]
    return banks,np.concatenate(context)


def weighted_ridges(x, y, policy, tune=None):
    center = x.mean(0,dtype=np.float64)
    scale = 1/np.maximum(x.std(0,dtype=np.float64),.05)
    a = ((x-center)*scale).astype(np.float32)
    target = np.eye(7,dtype=np.float32)[y]
    prior = (np.bincount(y,minlength=7)+1)/(len(y)+7)
    balance = policy["inverse_frequency"] if tune is None else [tune["class_power"]]
    shrinkage = policy["ridge"] if tune is None else [tune["ridge"]]
    for power in balance:
        mass = (prior[y]**-power).astype(np.float32)
        mu = (mass @ a)/mass.sum()
        mean_target = (mass @ target)/mass.sum()
        x_centered = a-mu
        mean_x = center + mu/scale
        key = x_centered * mass[:,None]
        gram = (key.T @ x_centered).astype(np.float64)
        rhs = (key.T @ (target-mean_target)).astype(np.float64)
        for ridge in shrinkage:
            normal = gram + ridge * np.eye(len(gram)) * gram.trace()/len(gram)
            coeff = np.ascontiguousarray(
                cho_solve(cho_factor(normal,check_finite=False),rhs,check_finite=False) * scale[:,None], np.float32)
            bias = (mean_target - mean_x@coeff).astype(np.float32)
            yield power,ridge,coeff,bias


def features_for(bank, hierarchy, name, fraction):
    from fit import join

    if name == "hierarchy":
        return hierarchy
    b = join(bank,name)
    return np.column_stack(((1-fraction)*b[:,0]+fraction*b[:,1],hierarchy))


def choose(fit_banks, fit_h, select_banks, select_h, view, policy, tr, decode, tune=None):
    from fit import confusion, join, metrics, offsets

    y = join(fit_banks,"gold")
    target = join(select_banks,"gold")
    ends = offsets(select_banks)
    variance = [None] if view == "hierarchy" else policy["last_layer_fraction"]
    if tune is not None:
        variance = [tune["fraction_last"]]
    gains = policy["transition_strength"] if tune is None else [tune["transition"]]
    trial = []
    winner = None
    for fraction in variance:
        x = features_for(fit_banks,fit_h,view,fraction)
        v = features_for(select_banks,select_h,view,fraction)
        for power,ridge,w,bias in weighted_ridges(x,y,policy,tune):
            logits = v@w+bias
            for tau in gains:
                pred = decode(logits,tr,ends,tau)
                score = metrics(confusion(target,pred))
                rank = score["macro_f1"],score["accuracy"]
                rule = dict(fraction_last=fraction,class_power=power,ridge=ridge,transition=tau,
                            select_macro_f1=rank[0],select_accuracy=rank[1])
                trial.append(rule)
                if winner is None or rank > winner[0]:
                    winner = rank,rule,w,bias
    return winner[1],winner[2],winner[3],trial


def packed_scores(bank, frozen, params, view, style):
    from fit import EXTRA, auxiliary, values

    p = frozen["settings"][view]["selected"][style]
    fn = view+"."+style
    d = auxiliary(bank,params["lexical.counts"],False)
    h = values(bank,view,p["fraction_last"],d)
    return h[:,EXTRA[style]]@params[fn+".weight"]+params[fn+".bias"],p["transition"]


def prepare(train,select,policy,tr,decode,packed,packed_params):
    from fit import confusion, join, metrics, offsets

    out,arrays = {},{"crf.transitions":tr}
    for key,view in (("joint_ordered","ordered"),("joint_shuffled","shuffled"),
                     ("shuffled_same_settings","shuffled"),("hierarchy_only","hierarchy")):
        fixed = out["joint_ordered"]["selected"] if key == "shuffled_same_settings" else None
        rule,w,bias,trial = choose(*train,*select,view,policy,tr,decode,fixed)
        out[key] = dict(view=view,selected=rule,grid=trial)
        arrays[key+".weight"] = w
        arrays[key+".bias"] = bias
        print("joint selector",key,rule,flush=True)
    b,v = select
    p,_ = packed_scores(b,packed,packed_params,"ordered","native")
    h = v@arrays["hierarchy_only.weight"]+arrays["hierarchy_only.bias"]
    y = join(b,"gold")
    e = offsets(b)
    best = None
    possibilities = []
    for gain,tau in product(policy["blend_weights_for_bert"],policy["transition_strength"]):
        pred = decode((1-gain)*h+gain*p,tr,e,tau)
        m = metrics(confusion(y,pred))
        stamp = dict(bert_weight=gain,transition=tau,select_macro_f1=m["macro_f1"],select_accuracy=m["accuracy"])
        possibilities.append(stamp)
        key = m["macro_f1"],m["accuracy"]
        if best is None or key > best[0]:
            best = key,stamp
    out["logit_blend"] = dict(selected=best[1],grid=possibilities)
    return out,arrays


def native_patches(destination, arrays, original):
    from native import sha

    authored = {}
    for kind in ("joint_ordered","joint_shuffled","shuffled_same_settings"):
        w,bias = arrays[kind+".weight"],arrays[kind+".bias"]
        assert w.shape == (968,7) and bias.shape == (7,)
        files = (destination/(kind+"-bert.safetensors"),destination/(kind+"-hierarchy.safetensors"))
        we, bi = original["weight"].copy(), original["bias"].copy()
        we[3:] = w[768:].T
        bi[3:] = 0.
        tensors = [
            {"classifier.weight":np.ascontiguousarray(w[:768].T),"classifier.bias":bias},
            {"emitter.hidden2tag.weight":we, "emitter.hidden2tag.bias":bi},
        ]
        for file,s in zip(files,tensors):
            save_file(s,str(file.with_suffix(".part")))
            file.with_suffix(".part").replace(file)
        authored[kind] = {file.name:dict(sha256=sha(file), shapes={k:list(v.shape) for k,v in value.items()})
                          for file,value in zip(files,tensors)}
    return authored


def run_predictions(bank,h,spec,params,orig,packed,packed_params,decode,destination):
    from fit import join, offsets

    ends = offsets(bank)
    tr = params["crf.transitions"]
    result = dict(
        source_hierarchy=decode(h@orig["weight"][3:].T+orig["bias"][3:],orig["transitions"],ends,1.),
        source_bert_cls=join(bank,"source_scores").argmax(1).astype(np.int8),
        current_20pct=join(bank,"incumbent"))
    emissions = {}
    split_errors = {}
    for name in ("joint_ordered","joint_shuffled","shuffled_same_settings","hierarchy_only"):
        group = spec[name]
        rule = group["selected"]
        v = features_for(bank,h,group["view"],rule["fraction_last"])
        f = v@params[name+".weight"]+params[name+".bias"]
        if name != "hierarchy_only":
            a = load_file(str(destination/(name+"-bert.safetensors")))
            b = load_file(str(destination/(name+"-hierarchy.safetensors")))
            one = v[:,:768]@a["classifier.weight"].T + a["classifier.bias"]
            other = h@b["emitter.hidden2tag.weight"][3:].T+b["emitter.hidden2tag.bias"][3:]
            np.testing.assert_allclose(one+other,f,rtol=0,atol=8e-6)
            fused = decode(f,tr,ends,rule["transition"])
            physical = decode(one+other,tr,ends,rule["transition"])
            split_errors[name] = dict(max_abs_logit_difference=float(np.max(np.abs(one+other-f))),
                                      label_differences=int(np.sum(fused!=physical)))
            # Evaluate the two-patch inference path rather than the concatenated approximation.
            f = one+other
        emissions[name] = f
        result[name] = decode(f,tr,ends,rule["transition"])
    for view in ("ordered","shuffled"):
        for style in ("native","position_lexical"):
            f,t = packed_scores(bank,packed,packed_params,view,style)
            name = "packed_"+view+"."+style
            emissions[name] = f
            result[name] = decode(f,tr,ends,t)
    blend = spec["logit_blend"]["selected"]
    r = blend["bert_weight"]
    mix = r*emissions["packed_ordered.native"] + (1-r)*emissions["hierarchy_only"]
    result["logit_blend"] = decode(mix,tr,ends,blend["transition"])
    emissions["logit_blend"] = mix
    return result,emissions,split_errors


def evaluate(bank,h,spec,params,orig,packed,packed_params,decode,destination,reference,confidence_cuts,roles):
    from fit import confusion, join, measures, metrics, offsets, span_counts
    from native import read, save_arrays, sha, write

    all_preds,emissions,errors = run_predictions(bank,h,spec,params,orig,packed,packed_params,decode,destination)
    archived = reference/"shared48-predictions.npz"
    with np.load(archived,allow_pickle=False) as base:
        np.testing.assert_array_equal(base["keys"],[x["id"] for x in bank])
        np.testing.assert_array_equal(base["csv_rows"],join(bank,"rows"))
        np.testing.assert_array_equal(base["current_20pct"],all_preds["current_20pct"])
        for view in ("ordered","shuffled"):
            for style in ("native","position_lexical"):
                np.testing.assert_array_equal(base[view+"."+style],all_preds["packed_"+view+"."+style])
    for n in (16,48):
        subset = bank[:n]
        ends = offsets(subset)
        y = join(subset,"gold")
        preds = {name:p[:len(y)] for name,p in all_preds.items()}
        cm = {name:np.stack([confusion(y[a:b],p[a:b]) for a,b in zip(ends,ends[1:])]) for name,p in preds.items()}
        spans = {name:np.stack([span_counts(y[a:b],p[a:b]) for a,b in zip(ends,ends[1:])]) for name,p in preds.items()}
        current = preds["current_20pct"]
        score = {}
        for name,p in preds.items():
            a,b,g = spans[name].sum(0)
            score[name] = metrics(cm[name].sum(0)) | dict(
                exact_span_f1=float(2*a/(b+g)),corrected=int(np.sum((p==y)&(current!=y))),
                harmed=int(np.sum((p!=y)&(current==y))))
        draw = np.random.default_rng(723427).integers(n,size=(2500,n))
        primary = "joint_ordered"
        comparison = {}
        for control in ("joint_shuffled","shuffled_same_settings","hierarchy_only","packed_ordered.native","logit_blend","current_20pct"):
            left_measure,right_measure = (measures(cm[x][draw].sum(1)) for x in (primary,control))
            left,right = left_measure[1],right_measure[1]
            a,b = (spans[x][draw].sum(1) for x in (primary,control))
            span_gap = 2*a[:,0]/(a[:,1]+a[:,2]) - 2*b[:,0]/(b[:,1]+b[:,2])
            role_ci = np.quantile(left_measure[0]-right_measure[0],[.025,.975],axis=0).T
            comparison[control] = dict(macro_f1_ci95=np.quantile(left-right,[.025,.975]).tolist(),
                                       span_f1_ci95=np.quantile(span_gap,[.025,.975]).tolist(),
                                       per_role_f1_ci95=dict(zip(roles,role_ci.tolist())))
        native = np.sort(join(subset,"source_scores"),axis=1)
        bands = np.searchsorted(confidence_cuts,native[:,-1]-native[:,-2],side="right")
        diagnostics = []
        for k in range(len(confidence_cuts)+1):
            selected = bands == k
            if np.any(selected):
                diagnostics.append(dict(band=k,sentences=int(selected.sum()),scores={
                    method:metrics(confusion(y[selected],preds[method][selected]))
                    for method in ("source_bert_cls","joint_ordered","current_20pct")}))
        path = destination/f"hybrid-shared{n}-predictions.npz"
        save_arrays(path,keys=[i["id"] for i in subset],csv_rows=join(subset,"rows"),offsets=ends,
                    gold=y,**preds,**{"emission."+k:v[:len(y)] for k,v in emissions.items()})
        write(destination/f"hybrid-shared{n}.json",dict(
            judgments=n,rows=len(y),scores=score,paired_ordered_minus=comparison,physical_readout_checks=errors,
            source_margin_diagnostics=dict(train_only_quartiles=confidence_cuts,bands=diagnostics,
                                           used_for_gating=False),
            weights_freeze_sha256=sha(destination/"freeze.json"),prediction_sha256=sha(path),official_test_evaluated=False))
        print("joint replay",n,len(y),{k:round(v["macro_f1"],5) for k,v in score.items()},flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("packed_state",type=Path)
    p.add_argument("hierarchy_inputs",type=Path)
    args = p.parse_args()
    d = args.packed_state/"data"
    f = args.packed_state/"features"
    reference = args.packed_state/"run-corrected"
    sys.path.insert(0,str(d))
    from fit import join,loader
    from native import read, sha, write

    hierarchy = args.hierarchy_inputs
    policy = read(hierarchy/"hybrid-protocol.json")
    assert policy["packed_protocol_sha256"] == sha(d/"protocol.json")
    packed = read(reference/"freeze.json")
    assert packed["serialized_weights_roundtrip"]
    assert packed["plan_sha256"] == policy["packed_protocol_sha256"]
    assert packed["weights_sha256"] == sha(reference/"analytic-heads.safetensors")
    packed_params = load_file(str(reference/"analytic-heads.safetensors"))
    assert read(hierarchy/"receipts.json")["author_head_sha256"] == sha(hierarchy/"author-head.npz")
    with np.load(hierarchy/"author-head.npz",allow_pickle=False) as file:
        original = {k:file[k] for k in file.files}
    out = args.packed_state/"hybrid-run"
    out.mkdir(exist_ok=True)
    decode = loader()
    physical = out/"joint-heads.safetensors"
    if not (out/"freeze.json").exists():
        fit_sample = evidence(d,f,hierarchy,"fit")
        select = evidence(d,f,hierarchy,"select")
        spec,params = prepare(fit_sample,select,policy,packed_params["crf.transitions"],decode,packed,packed_params)
        save_file({key:np.ascontiguousarray(value) for key,value in params.items()},str(physical.with_suffix(".part")))
        physical.with_suffix(".part").replace(physical)
        reopened = load_file(str(physical))
        for key,value in params.items():
            np.testing.assert_array_equal(reopened[key],value)
        patches = native_patches(out,params,original)
        native = np.sort(join(fit_sample[0],"source_scores"),axis=1)
        confidence_cuts = np.quantile(native[:,-1]-native[:,-2],[.25,.5,.75]).tolist()
        write(out/"freeze.json",dict(settings=spec,weights_sha256=sha(physical),native_patches=patches,
            protocol_sha256=sha(hierarchy/"hybrid-protocol.json"),packed_freeze_sha256=sha(reference/"freeze.json"),
            fit_cases=policy["groups"]["fit"],select_cases=policy["groups"]["select"],gradient_steps=0,
            train_source_margin_quartiles=confidence_cuts,
            confidence_scope="Reference head only. Fit-case quartile boundaries for descriptive validation stratification. Never used in a gate."))
        del fit_sample,select
    checked = read(out/"freeze.json")
    assert checked["weights_sha256"] == sha(physical)
    assert checked["protocol_sha256"] == sha(hierarchy/"hybrid-protocol.json")
    assert checked["packed_freeze_sha256"] == sha(reference/"freeze.json")
    for pack in checked["native_patches"].values():
        for filename,m in pack.items():
            assert sha(out/filename) == m["sha256"]
    bank,h = evidence(d,f,hierarchy,"shared")
    evaluate(bank,h,checked["settings"],load_file(str(physical)),original,packed,packed_params,decode,out,reference,
             checked["train_source_margin_quartiles"],policy["role_order"])
    zip_path = args.packed_state/"packed-hybrid-results.zip"
    stage = zip_path.with_suffix(".part")
    with ZipFile(stage,"w",compression=ZIP_DEFLATED,compresslevel=2) as pack:
        for path in sorted(out.iterdir()):
            pack.write(path,path.name)
        for path in (hierarchy/"hybrid-protocol.json",hierarchy/"receipts.json",reference.parent/"device.json"):
            pack.write(path,path.name)
    stage.replace(zip_path)
    print(zip_path,sha(zip_path),flush=True)


if __name__=="__main__":
    main()
