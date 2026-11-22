"""Open the separately reserved 72-judgment gate after the complete training-only scaling freeze."""
import argparse
import csv
import json
import numpy as np
from safetensors.numpy import load_file,save_file
from native_memory import HERE,ROOT,DONOR,BASE,TO_ID,HierModel,confusion_docs,metric,dump,file_hash
from retention_replay import load_arm
from scale_program import FOLDER,NAMES,SIZES,compiler,save_np

def parts(name):
    return json.loads((FOLDER/name).read_text())

def run():
    plan=parts("plan.json");frozen=parts("freeze.json")
    assert frozen["source_plan"]==file_hash(FOLDER/"plan.json") and frozen["weights_sha256"]==file_hash(FOLDER/"final-heads.safetensors")
    candidate=frozen["chosen"];dest=FOLDER/"frozen-native-patch.safetensors"
    if not dest.exists():
        final=load_file(str(FOLDER/"final-heads.safetensors"))[candidate]
        m=HierModel(DONOR);emit=m.weight.copy();bias=m.bias.copy()
        emit[3:]=final[:-1].T;bias[3:]=final[-1]
        patch={"emitter.hidden2tag.weight":emit,"emitter.hidden2tag.bias":bias}
        if candidate=="retention":patch.update(load_file(str(HERE/"uniform-retention.safetensors")))
        tmp=dest.with_suffix(".part");save_file(patch,str(tmp));tmp.replace(dest)
        dump(FOLDER/"frozen-native.json",dict(chosen=candidate,frozen_on_train_before_case_outcomes=file_hash(FOLDER/"freeze.json"),patch_sha256=file_hash(dest),tensor_shapes={k:list(v.shape) for k,v in patch.items()},gradient_steps=0))
    if (FOLDER/"reserved72.json").exists():return
    from scale_full_balance import run as full_equal_role
    if not (FOLDER/"equal-role-4984.safetensors").exists():full_equal_role()
    from scale_lda import run as discriminant
    if not (FOLDER/"lda-4984.safetensors").exists():discriminant()
    from scale_quadratic import Q,program as freeze_quadratic
    if not (Q/"freeze.json").exists():freeze_quadratic(4792)
    ids=plan["validation"];assert len(ids)==72
    with (ROOT/"references/official/val.csv").open(newline="",encoding="utf-8-sig") as fp:
        rows={key:([],[]) for key in ids}
        for r in csv.DictReader(fp):
            text=r["Text"].strip()
            if r["Index"] in rows and text:rows[r["Index"]][0].append(text);rows[r["Index"]][1].append(TO_ID[r["Label"]])
    t=HierModel(DONOR);c=load_arm("uniform-retention.safetensors",plan["retention_patch"])
    viterbi=compiler();ys=[];ends=[0];ops=[[],[]]
    for k in ids:
        texts,gt=rows[k];w=t.words(texts);ys.extend(gt);ends.append(len(ys))
        ops[0].append(t.contextualize(w));ops[1].append(c.contextualize(w))
    ys=np.asarray(ys);states=[np.concatenate(x) for x in ops];heads={}
    old_head=load_file(str(HERE/"native-base-240-71.safetensors"))
    old=np.vstack((old_head["emitter.hidden2tag.weight"][3:].T,old_head["emitter.hidden2tag.bias"][3:])).astype(np.float32)
    orig=np.vstack((t.weight[3:].T,t.bias[3:])).astype(np.float32)
    for arm in NAMES:
        heads[(arm,"unedited_readout")]=orig;heads[(arm,"legacy_240")]=old
    for n in (*SIZES,4984):
        path=FOLDER/(f"heads-{n}.safetensors" if n!=4984 else "final-heads.safetensors")
        hh=load_file(str(path))
        for name,H in hh.items():heads[(name,str(n))]=H
        equal=load_file(str(FOLDER/f"equal-role-{n}.safetensors"))
        for name,H in equal.items():heads[(name,str(n)+"_equal_role_mass")]=H
        lda=load_file(str(FOLDER/f"lda-{n}.safetensors"))
        for name,H in lda.items():heads[(name,str(n)+"_lda")]=H
    preds={};scores={};tbl={}
    for (arm,size),H in heads.items():
        pred=viterbi(states[NAMES.index(arm)]@H[:-1]+H[-1],ends,t.transitions)
        key=arm+"-"+size;preds[key]=pred;tbl[key]=confusion_docs(ys,pred,ends);scores[key]=metric(tbl[key].sum(0))
    target=candidate+"-4984";comps={}
    for k in (candidate+"-legacy_240",candidate+"-240",candidate+"-960",candidate+"-2400","donor-unedited_readout"):
        a,b=preds[target],preds[k];changed=a!=b;rng=np.random.default_rng(11990);ds=[]
        for j in rng.integers(len(ids),size=(4000,len(ids))):
            x,z=metric(tbl[target][j].sum(0)),metric(tbl[k][j].sum(0));ds.append((x["macro_f1"]-z["macro_f1"],x["accuracy"]-z["accuracy"]))
        comps[k]={"changed":int(changed.sum()),"corrected":int((changed&(ys==a)).sum()),"harmed":int((changed&(ys==b)).sum()),"paired95_F1_accuracy":np.quantile(ds,[.025,.975],axis=0).T.tolist()}
    guard=(comps[candidate+"-legacy_240"]["paired95_F1_accuracy"][0][0]>0 and
           scores[target]["accuracy"]>=scores[candidate+"-legacy_240"]["accuracy"] and scores[target]["macro_f1"]>scores[candidate+"-240"]["macro_f1"])
    save_np(FOLDER/"reserved72-predictions.npz",keys=ids,ends=ends,gold=ys,**preds)
    dump(FOLDER/"reserved72.json",dict(rows=len(ys),cases=ids,models=scores,chosen=target,paired=comps,gate_passed=bool(guard),prospective_plan=file_hash(FOLDER/"plan.json"),frozen_model=file_hash(dest),test_evaluated=False))
    print(target,scores[target]["macro_f1"],scores[target]["accuracy"],"gate",guard,flush=True)
if __name__=="__main__":
    run()
    from scale_quadratic import Q,assess as quantify_quadratic
    if (Q/"freeze.json").exists() and not (Q/"reserved48.json").exists():quantify_quadratic()
