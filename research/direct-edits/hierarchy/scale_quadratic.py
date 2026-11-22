"""Direct covariance-based square-and-sum head beside a frozen native linear readout."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file,save_file
from native_memory import HERE,ROOT,DONOR,BASE,HierModel,TO_ID,metric,confusion,confusion_docs,dump,file_hash
from scale_program import FOLDER,NAMES,SIZES,empty,compiler,save_np
from retention_replay import load_arm

Q=FOLDER/"quadratic"
def reserve():
    Q.mkdir(exist_ok=True);path=Q/"plan.json"
    if path.exists():return json.loads(path.read_text())
    base=json.loads((FOLDER/"plan.json").read_text());blocked=set(base["validation"])
    blocked.update(json.loads((HERE/"uniform-retention-prospective.json").read_text())["ids"])
    blocked.update(json.loads((HERE.parent/"shared-confirmation.json").read_text())["extended_validation_judgment_ids"])
    for k in ("select","confirm","holdout"):
        with np.load(ROOT/f"cache/sentence-recurrent-inverse/{k}.npz") as f:blocked.update(f["keys"].tolist())
    src=ROOT/"references/official/val.csv"
    with src.open(encoding="utf-8-sig",newline="") as fp:all_ids={r["Index"] for r in csv.DictReader(fp)}
    pool=sorted(all_ids-blocked,key=lambda k:hashlib.sha256(("role-native-qda-20261122|"+k).encode()).digest())
    policy=dict(parent_plan=file_hash(FOLDER/"plan.json"),algorithm="Per-class, pooled-whitened covariance contrasts; native projection, squared activations, per-role weighted sums and a fixed linear readout.",
        rank=[8,24],class_covariance_shrink_to_pooled=[.65,.85],residual_gain=[0.,.20,.55,1.25],pooled_spherical_shrink=.12,pooled_scale="1/sqrt(rank) times low-rank Gaussian log-likelihood correction",
        primary="Pick rank, covariance shrink and gain on the 192 training-calibration judgments for each state space. Choose the space with best macro-F1 then accuracy at 4792, before combining all 4984 training judgments. Advance only if the training-side macro-F1 difference relative to the same 4792 linear readout is positive, and on a distinct 48 validation judgments the lower paired 95% macro-F1 bound exceeds zero relative to matched linear, with no point accuracy loss and higher point macro-F1 than a gain- and rank-matched role-permuted covariance comparator.",
        covariance_control="Same seven fitted covariance response columns, reassigned to roles by the fixed derangement [1, 4, 6, 0, 2, 3, 5]. Linear component unchanged.",role_permutation=[1,4,6,0,2,3,5],cases=pool[:48],validation_source=file_hash(src),gradient_steps=0,boundary="New explicit quadratic decoder tensors after unchanged 200-dimensional native hidden states. The legacy source has a linear decoder. No test scoring is authorized for this family.")
    dump(path,policy);return policy

def reduction(G,n,shrink,rank):
    means=G[:,:-1,-1]/n[:,None];mu=np.average(means,axis=0,weights=n)
    Cs=np.array([G[k,:-1,:-1]/n[k]-np.outer(means[k],means[k]) for k in range(7)])
    pooled=np.average(Cs,axis=0,weights=n);p=.12
    eig,vec=np.linalg.eigh((1-p)*pooled+p*np.eye(200)*np.trace(pooled)/200)
    white=(vec*(1/np.sqrt(np.maximum(eig,1e-9))))@vec.T
    filters=[];bias=[];coeff=[];offsets=[]
    for i in range(7):
        e,U=np.linalg.eigh((1-shrink)*(white@Cs[i]@white)+shrink*np.eye(200))
        a=1.-1./e;ix=np.argsort(np.abs(a))[-rank:];W=white@U[:,ix]
        filters.append(W.T);bias.append(-means[i]@W);coeff.append(.5*a[ix]/np.sqrt(rank));offsets.append(-.5*np.log(e[ix]).sum()/np.sqrt(rank))
    return {"quadratic_projection.weight":np.asarray(filters,np.float32),"quadratic_projection.bias":np.asarray(bias,np.float32),
            "quadratic_role.weight":np.asarray(coeff,np.float32),"quadratic_role.bias":np.asarray(offsets,np.float32)}

def response(H,w):
    z=np.empty((len(H),7),np.float32)
    for i in range(7):
        v=H@w["quadratic_projection.weight"][i].T+w["quadratic_projection.bias"][i]
        z[:,i]=(v*v)@w["quadratic_role.weight"][i]+w["quadratic_role.bias"][i]
    return z-z.mean(1,keepdims=True)

def program(n=4792):
    cfg=reserve();parent=json.loads((FOLDER/f"train-side-{n}.json").read_text())
    stat=empty()
    for p in sorted((FOLDER/"shards").glob("*.npz")):
        lo,hi=map(int,p.stem.split("-")); 
        if hi>n:continue
        with np.load(p) as f:
            for k in stat:stat[k]+=f[k]
    assert int(stat["count"].sum())==parent["training_rows"]
    cal=np.load(FOLDER/"train-calibration.npz");viterbi=compiler();m=HierModel(DONOR);heads=load_file(str(FOLDER/f"heads-{n}.safetensors"));spec={}
    for i,name in enumerate(NAMES):
        H=cal[name];y=cal["y"];ends=cal["ends"];W=heads[name]
        z=H@W[:-1]+W[-1];grid=[]
        baseline=metric(confusion(y,viterbi(z,ends,m.transitions)))
        for sh in cfg["class_covariance_shrink_to_pooled"]:
            for r in cfg["rank"]:
                patch=reduction(stat["G"][i],stat["count"],sh,r);corr=response(H,patch)
                for gain in cfg["residual_gain"]:
                    pred=viterbi(z+gain*corr,ends,m.transitions);s=metric(confusion(y,pred))
                    grid.append((s["macro_f1"],s["accuracy"],-gain,-r,sh,s))
        winner=max(grid,key=lambda q:q[:4]);spec[name]=dict(shrink=winner[4],rank=-winner[3],gain=-winner[2],metrics=winner[-1],matched_linear=baseline)
    chosen=max(spec,key=lambda k:(spec[k]["metrics"]["macro_f1"],spec[k]["metrics"]["accuracy"]))
    dump(Q/f"train-side-{n}.json",dict(fitted_judgments=n,fitted_rows=parent["training_rows"],arms=spec,chosen=chosen,source_plan=file_hash(Q/"plan.json")))
    print(n,{k:(z["metrics"]["macro_f1"],z["matched_linear"]["macro_f1"],z["gain"]) for k,z in spec.items()},flush=True)
    if n!=4792:return
    with np.load(FOLDER/"cal-stats.npz") as f:
        for k in stat:stat[k]+=f[k]
    settings=spec[chosen];patch=reduction(stat["G"][NAMES.index(chosen)],stat["count"],settings["shrink"],settings["rank"])
    patch["quadratic_role.weight"]*=settings["gain"];patch["quadratic_role.bias"]*=settings["gain"]
    patch["linear_head"]=load_file(str(FOLDER/"final-heads.safetensors"))[chosen]
    path=Q/"head.safetensors";tmp=path.with_suffix(".part");save_file(patch,str(tmp));tmp.replace(path)
    dump(Q/"freeze.json",dict(chosen=chosen,settings=settings,source_plan=file_hash(Q/"plan.json"),fit_rows=int(stat["count"].sum()),
        weights_sha256=file_hash(path),new_tensor_shapes={k:list(v.shape) for k,v in patch.items() if k!="linear_head"},extra_parameters=sum(v.size for k,v in patch.items() if k!="linear_head"),
        train_side_advance=settings["metrics"]["macro_f1"]>settings["matched_linear"]["macro_f1"],gradient_steps=0))

def assess():
    cfg=reserve();f=json.loads((Q/"freeze.json").read_text())
    if not f["train_side_advance"]:return
    src=ROOT/"references/official/val.csv";assert file_hash(src)==cfg["validation_source"]
    w=load_file(str(Q/"head.safetensors"));assert file_hash(Q/"head.safetensors")==f["weights_sha256"];c=f["chosen"]
    cohort=cfg["cases"];rows={k:([],[]) for k in cohort}
    with src.open(newline="",encoding="utf-8-sig") as fp:
        for r in csv.DictReader(fp):
            if r["Index"] in rows and r["Text"].strip():rows[r["Index"]][0].append(r["Text"].strip());rows[r["Index"]][1].append(TO_ID[r["Label"]])
    m=HierModel(DONOR) if c=="donor" else load_arm("uniform-retention.safetensors",json.loads((FOLDER/"plan.json").read_text())["retention_patch"])
    x=[];y=[];ends=[0]
    for k in cohort:
        t,g=rows[k];x.append(m.contextualize(m.words(t)));y.extend(g);ends.append(len(y))
    X=np.concatenate(x);y=np.asarray(y);viterbi=compiler();W=w["linear_head"];z=X@W[:-1]+W[-1]
    correction=response(X,w);a=viterbi(z+correction,ends,m.transitions);b=viterbi(z,ends,m.transitions)
    shuffled=viterbi(z+correction[:,cfg["role_permutation"]],ends,m.transitions)
    ca=confusion_docs(y,a,ends);sa=metric(ca.sum(0));comparisons={}
    for label,pred in (("linear",b),("role_shuffled",shuffled)):
        cb=confusion_docs(y,pred,ends);sb=metric(cb.sum(0));rng=np.random.default_rng(927855);boot=[]
        for ix in rng.integers(len(cohort),size=(4000,len(cohort))):
            left,right=metric(ca[ix].sum(0)),metric(cb[ix].sum(0));boot.append([left["macro_f1"]-right["macro_f1"],left["accuracy"]-right["accuracy"]])
        margin=np.quantile(boot,[.025,.975],axis=0).T
        comparisons[label]=dict(scores=sb,paired95=margin.tolist(),changes=int((a!=pred).sum()),
            corrected=int(((a!=pred)&(a==y)).sum()),harmed=int(((a!=pred)&(pred==y)).sum()))
    lin=comparisons["linear"];sh=comparisons["role_shuffled"]
    passed=lin["paired95"][0][0]>0 and sa["accuracy"]>=lin["scores"]["accuracy"] and sa["macro_f1"]>sh["scores"]["macro_f1"]
    save_np(Q/"reserved48-predictions.npz",keys=cohort,ends=ends,gold=y,quadratic=a,linear=b,role_shuffled=shuffled)
    dump(Q/"reserved48.json",dict(cases=cohort,rows=len(y),quadratic=sa,controls=comparisons,passed=bool(passed),weights_sha256=f["weights_sha256"],official_test_evaluated=False))
if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("action",choices=("reserve","fit","assess"));a.add_argument("--n",type=int,default=4792);o=a.parse_args()
    if o.action=="reserve":reserve()
    elif o.action=="fit":program(o.n)
    else:assess()
