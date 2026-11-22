"""Restartable train-corpus sufficient statistics and one-shot native readout writes."""
import argparse
import csv
import ctypes
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import time
import numpy as np
from scipy.special import softmax
from safetensors.numpy import load_file,save_file
import torch
from native_memory import HERE,ROOT,DONOR,BASE,HierModel,TO_ID,native,metric,confusion,confusion_docs,dump,file_hash,decode
from retention_replay import load_arm

FOLDER=HERE/"scale"
SIZES=(240,960,2400,4792)
SHARD=48
SALT="attoflow-hierarchy-readout-scaling-v1"
DB=FOLDER/"training.sqlite3"
SOURCE=ROOT/"references/official/train.csv"
NAMES=("donor","retention")
D=201
def salted(tag,k):
    return hashlib.sha256((SALT+"|"+tag+"|"+str(k)).encode()).digest()

def initialize():
    if not DB.exists():
        tmp=DB.with_suffix(".part")
        if tmp.exists():tmp.unlink()
        conn=sqlite3.connect(tmp)
        conn.execute("PRAGMA journal_mode=OFF");conn.execute("PRAGMA synchronous=OFF")
        conn.execute("CREATE TABLE sentence (doc INTEGER, seq INTEGER, text TEXT NOT NULL, role INTEGER NOT NULL, PRIMARY KEY(doc,seq)) WITHOUT ROWID")
        buf=[];counts=np.zeros(7,np.int64);docs=set()
        with SOURCE.open(newline="",encoding="utf-8-sig") as fp:
            for num,row in enumerate(csv.DictReader(fp)):
                t=row['Text'].strip()
                if not t:continue
                k=int(row["Index"]);r=TO_ID[row["Label"]];buf.append((k,num,t,r));counts[r]+=1;docs.add(k)
                if len(buf)>=25000:
                    conn.executemany("INSERT INTO sentence VALUES (?,?,?,?)",buf);buf.clear()
        conn.executemany("INSERT INTO sentence VALUES (?,?,?,?)",buf);conn.commit();conn.close()
        known=json.loads((HERE/"counts.json").read_text());assert counts.tolist()==known["class_support"] and len(docs)==known["docs"]==4984
        tmp.replace(DB)
    plan=FOLDER/"plan.json"
    if not plan.exists():
        conn=sqlite3.connect(DB);ids=[v[0] for v in conn.execute("SELECT DISTINCT doc FROM sentence")];conn.close()
        cal=sorted(ids,key=lambda k:salted("calibration",k))[:192];fit=sorted(set(ids)-set(cal),key=lambda k:salted("fit",k))
        val=ROOT/"references/official/val.csv"
        with val.open(newline="",encoding="utf-8-sig") as fp:all_val={r["Index"] for r in csv.DictReader(fp)}
        blocked=set(json.loads((HERE/"uniform-retention-prospective.json").read_text())["ids"])
        blocked.update(json.loads((HERE.parent/"shared-confirmation.json").read_text())["extended_validation_judgment_ids"])
        for name in ("select","confirm","holdout"):
            with np.load(ROOT/f"cache/sentence-recurrent-inverse/{name}.npz") as f:blocked.update(f["keys"].tolist())
        reserved=sorted(all_val-blocked,key=lambda k:salted("locked-validation",k))[:72]
        assert len(fit)==4792 and len(cal)==192 and len(reserved)==72
        dump(plan,dict(source=file_hash(SOURCE),donor=file_hash(BASE),retention_patch=file_hash(HERE/"uniform-retention.safetensors"),calibration=cal,fit=fit,
             fitted_prefixes=list(SIZES),final_refit=4984,validation=reserved,val_csv_sha256=file_hash(val),calibration_policy="Choose class power/ridge/gain on the 192 train-side judgments excluded from each partial solve; the externally published donor was previously trained on this split. Then refit the chosen full-scale head on all 4,984. The 72 validation IDs have not been used by this scaling family; unrelated historical validation exposure may exist.",
             grid=dict(class_power=[0.,.28,.6],ridge=[.03,.3],gain=[.5,1.25,2.5]),selection_key="macro-F1 then accuracy on train-side calibration",primary_contender="Of donor-state and retention-state heads at 4792, select on the fixed train calibration then lock the 4984 direct refit; all smaller scales serve as controls.",
             gate="On 72 untouched-by-family validation judgments, require positive lower 95% judgment-bootstrap macro-F1 gap vs the matching frozen 240-case head (native baseline matched for presence or absence of the 200 gate entries). Require point macro-F1 > freshly refit 240-example-scale arm, no point accuracy loss vs frozen 240. Do not score official test without separate authorization.",
             gradient_steps=0,cache_source="known-sentence-pair-edit per-doc states where verified from matching inputs; recurrent-inverse/fit cached native word vectors; saved new word vectors pinned to parent model SHA."))
    cfg=json.loads(plan.read_text())
    assert cfg["source"]==file_hash(SOURCE) and cfg["donor"]==file_hash(BASE) and cfg["retention_patch"]==file_hash(HERE/"uniform-retention.safetensors")
    return cfg

def compiler():
    source=FOLDER/"viterbi.cpp";so=FOLDER/("viterbi-"+file_hash(source)[:12]+".so")
    if not so.exists():
        tmp=so.with_suffix(".part");subprocess.run(["g++","-std=c++17","-O3","-shared","-fPIC","-ffp-contract=off",str(source),"-o",str(tmp)],check=True);tmp.replace(so)
    f=ctypes.CDLL(str(so)).role_viterbi;ptr=np.ctypeslib.ndpointer;f.argtypes=[ptr(np.float32,flags="C_CONTIGUOUS"),ptr(np.float32,flags="C_CONTIGUOUS"),ptr(np.int64,flags="C_CONTIGUOUS"),ctypes.c_size_t,ptr(np.int8,flags="C_CONTIGUOUS")];f.restype=None
    def run(x,ends,tr):
        scores=np.ascontiguousarray(x,dtype=np.float32);ends=np.ascontiguousarray(ends,dtype=np.int64);res=np.empty(len(scores),np.int8)
        f(scores,np.ascontiguousarray(tr,np.float32),ends,len(ends)-1,res);return res
    return run

def empty():
    return dict(count=np.zeros(7,np.int64),G=np.zeros((2,7,D,D)),B=np.zeros((2,7,D,7)))

def update(store,views,gt,prior):
    store["count"]+=np.bincount(gt,minlength=7)
    eye=np.eye(7)
    for k,H in enumerate(views):
        X=np.column_stack((H,np.ones(len(H))))
        old=X@prior;R=(eye[gt]-softmax(old/2.2,axis=1))*4.4
        for c in np.unique(gt):
            g=gt==c;xx=X[g];store["G"][k,c]+=xx.T@xx;store["B"][k,c]+=xx.T@R[g]

def save_np(path,**kwargs):
    tmp=path.with_suffix(".part")
    with tmp.open("wb") as fp:np.savez(fp,**kwargs)
    tmp.replace(path)

class Corpus:
    def __init__(self,plan):
        self.db=sqlite3.connect(f"file:{DB}?mode=ro",uri=True)
        self.base=HierModel(DONOR);self.other=load_arm("uniform-retention.safetensors",plan["retention_patch"])
        self.prior=np.row_stack([self.base.weight[3:].T,self.base.bias[3:]]).astype(np.float64)
        self.path=FOLDER/"words";self.path.mkdir(exist_ok=True)
        with np.load(ROOT/"cache/sentence-recurrent-inverse/fit.npz") as f:
            assert f["backbone"].item()==plan['donor'];x=f["words"];ends=f["ends"]
            self.words={int(k):x[a:b] for k,a,b in zip(f["keys"],ends,ends[1:])}
        self.stats={"lane_cache":0,"verified_word_cache":0,"new_word_encode":0,"prior_state_reused":0,"prior_state_mismatch":0}

    def extract(self,k):
        raw=self.db.execute("SELECT text,role FROM sentence WHERE doc=? ORDER BY seq",(int(k),)).fetchall()
        tx,gt=zip(*raw);gt=np.asarray(gt,np.int64)
        wordpath=self.path/f"{k}.npy"
        if k in self.words:
            a=self.words[k];self.stats["verified_word_cache"]+=1
        elif wordpath.exists():
            a=np.load(wordpath,allow_pickle=False);self.stats["lane_cache"]+=1
        else:
            a=self.base.words(tx).numpy();tmp=wordpath.with_suffix(".part")
            with tmp.open("wb") as fp:np.save(fp,a,allow_pickle=False)
            tmp.replace(wordpath);self.stats["new_word_encode"]+=1
        assert a.shape==(len(gt),200) and np.isfinite(a).all();w=torch.from_numpy(a)
        oldpath=ROOT/f"cache/known-sentence-pair-edit/{k}.npy"
        actual=self.base.contextualize(w)
        if oldpath.exists():
            old=np.load(oldpath,allow_pickle=False)
            # Validate each cache row against recomputed current donor on the exact CSV inputs.
            if old.shape==actual.shape and np.allclose(actual,old,atol=5e-6,rtol=0):
                actual=old;self.stats["prior_state_reused"]+=1
            else:self.stats["prior_state_mismatch"]+=1
        return gt,(actual,self.other.contextualize(w))

def solution(G,B,counts,power,reg,gain,prior):
    freq=counts/counts.sum();weight=freq**-power
    gram=np.einsum("r,rij->ij",weight,G);rhs=np.einsum("r,rij->ij",weight,B)
    d=reg*gram.trace()/len(gram);pen=np.full(len(gram),d);pen[-1]=.001*d
    return (prior+gain*np.linalg.solve(gram+np.diag(pen),rhs)).astype(np.float32)

def view_score(states,y,ends,head,m,viterbi):
    z=states@head[:-1]+head[-1];pred=viterbi(z,ends,m.transitions)
    return metric(confusion(y,pred)),pred

def numeric_checkpoint(corpus,plan,cal,stats,n):
    dest=FOLDER/f"train-side-{n}.json"
    if dest.exists():return
    viterbi=compiler();y=cal["y"];ends=cal["ends"];out={};w={}
    for i,arm in enumerate(NAMES):
        opts={};heads={};X=cal[arm]
        for p in (.0,.28,.6):
            for reg in (.03,.3):
                for gain in (.5,1.25,2.5):
                    tag=f"{p},{reg},{gain}"
                    h=solution(stats["G"][i],stats["B"][i],stats["count"],p,reg,gain,corpus.prior)
                    val,_=view_score(X,y,ends,h,corpus.base,viterbi);opts[tag]=val;heads[tag]=h
        best=max(opts,key=lambda k:(opts[k]["macro_f1"],opts[k]["accuracy"]))
        out[arm]=dict(settings=best,metrics=opts[best],grid={k:dict(macro_f1=d["macro_f1"],accuracy=d["accuracy"]) for k,d in opts.items()});w[arm]=heads[best]
        z=X[:ends[3]]@w[arm][:-1]+w[arm][-1]
        np.testing.assert_array_equal(viterbi(z,ends[:4],corpus.base.transitions),decode(z,ends[:4],corpus.base.transitions))
    path=FOLDER/f"heads-{n}.safetensors";tmp=path.with_suffix(".part");save_file(w,str(tmp));tmp.replace(path)
    dump(dest,dict(training_judgments=n,training_rows=int(stats["count"].sum()),train_calibration_judgments=len(ends)-1,train_calibration_rows=len(y),arms=out,weight_sha256=file_hash(path),source_plan=file_hash(FOLDER/"plan.json"),gradient_steps=0));print("size",n,{k:(a["metrics"]["macro_f1"],a["metrics"]["accuracy"],a["settings"]) for k,a in out.items()},flush=True)

def run():
    plan=initialize();corpus=Corpus(plan);proof=file_hash(FOLDER/"plan.json")
    shard_dir=FOLDER/"shards";shard_dir.mkdir(exist_ok=True)
    calfile=FOLDER/"train-calibration.npz"
    if not calfile.exists() or not (FOLDER/"cal-stats.npz").exists():
        mats=[[],[]];ys=[];ends=[0];stat=empty()
        for i,k in enumerate(plan["calibration"]):
            gt,views=corpus.extract(k);ys.extend(gt);ends.append(len(ys))
            if (i+1)%24==0:
                dump(FOLDER/"progress.json",dict(stage="train-side calibration",processed=i+1,total=192,rows=len(ys),source_plan=proof));print("calibration",i+1,len(ys),flush=True)
            for a,p in zip(mats,views):a.append(p)
            update(stat,views,gt,corpus.prior)
        save_np(FOLDER/"cal-stats.npz",**stat)
        save_np(calfile,y=ys,ends=ends,donor=np.concatenate(mats[0]),retention=np.concatenate(mats[1]),keys=plan["calibration"],source_plan=proof)
    cal=np.load(calfile);assert cal["source_plan"].item()==proof
    cumulative=empty();ids=plan["fit"];progress=0
    # Split at every sample-size boundary so the same documents form nested, reproducible model comparisons.
    edges=sorted(set([0,*range(SHARD,len(ids),SHARD),*SIZES]))
    for a,b in zip(edges,edges[1:]):
        name=shard_dir/f"{a:04d}-{b:04d}.npz"
        if not name.exists():
            chunk=empty()
            for k in ids[a:b]:
                gt,views=corpus.extract(k);update(chunk,views,gt,corpus.prior)
            save_np(name,source_plan=proof,keys=ids[a:b],**chunk)
        with np.load(name) as block:
            assert block["source_plan"].item()==proof and block["keys"].tolist()==ids[a:b]
            for k in cumulative:cumulative[k]+=block[k]
        progress=b
        dump(FOLDER/"progress.json",dict(fit_judgments=progress,target=len(ids),fit_rows=int(cumulative["count"].sum()),calibration=len(plan["calibration"]),cache_usage_this_process=corpus.stats,source_plan=proof))
        if b%240==0 or b in SIZES:print("sufficient statistics",progress,int(cumulative["count"].sum()),flush=True)
        if b in SIZES:numeric_checkpoint(corpus,plan,cal,cumulative,b)
    with np.load(FOLDER/"cal-stats.npz") as block:
        for k in cumulative:cumulative[k]+=block[k]
    assert int(cumulative["count"].sum())==1122376
    tuned=json.loads((FOLDER/f"train-side-{len(ids)}.json").read_text());arms=tuned["arms"]
    chosen=max(arms,key=lambda k:(arms[k]["metrics"]["macro_f1"],arms[k]["metrics"]["accuracy"]))
    frozen={};weights={}
    for i,arm in enumerate(NAMES):
        setting=arms[arm]["settings"];p,reg,gain=map(float,setting.split(","))
        h=solution(cumulative["G"][i],cumulative["B"][i],cumulative["count"],p,reg,gain,corpus.prior)
        weights[arm]=h;frozen[arm]=dict(settings=setting)
    tmp=FOLDER/"final-heads.part";dest=FOLDER/"final-heads.safetensors";save_file(weights,str(tmp));tmp.replace(dest)
    dump(FOLDER/"freeze.json",dict(source_plan=proof,trained_judgments=4984,training_rows=int(cumulative["count"].sum()),support=cumulative["count"].tolist(),chosen=chosen,arms=frozen,weights_sha256=file_hash(dest),gradient_steps=0));print("frozen",chosen,dest,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("task",choices=("initialize","run"));args=p.parse_args()
    if args.task=="initialize":initialize()
    else:run()
