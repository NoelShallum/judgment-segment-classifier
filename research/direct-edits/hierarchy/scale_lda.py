"""Secondary corpus-scale generative native readout from the already recorded per-role Gram matrices."""
import json
import numpy as np
from safetensors.numpy import save_file
from native_memory import DONOR,HierModel,dump,file_hash
from scale_program import FOLDER,NAMES,SIZES,compiler,empty,view_score

def head(G,n,h0,shrink,logprior,blend):
    means=G[:,:-1,-1]/n[:,None]
    pooled=sum(G[k,:-1,:-1]-n[k]*np.outer(means[k],means[k]) for k in range(7))/(n.sum()-7)
    # Soft shrinkage toward a spherical covariance controls the inverse in less-used native directions.
    s=(1-shrink)*pooled+shrink*np.eye(pooled.shape[0])*pooled.trace()/len(pooled)
    w=np.linalg.solve(s,means.T)
    b=-.5*(means.T*w).sum(0)+logprior*np.log(n/n.sum())
    lda=np.vstack((w,b));lda-=lda.mean(1,keepdims=True)
    centered=h0-h0.mean(1,keepdims=True)
    # Match original native emission variance in the actual training distribution before interpolating with its CRF.
    total=G.sum(0);v=total[:-1,:-1]/n.sum()-np.outer(total[:-1,-1],total[:-1,-1])/(n.sum()**2)
    old_power=np.trace(centered[:-1].T@v@centered[:-1]);new_power=np.trace(w.T@v@w)-7*(w.mean(1)@v@w.mean(1))
    lda*=np.sqrt(old_power/max(new_power,1.e-10))
    return ((1-blend)*h0+blend*lda).astype(np.float32)

def run():
    preset=json.loads((FOLDER/"lda-control-plan.json").read_text());viterbi=compiler();m=HierModel(DONOR)
    h0=np.vstack([m.weight[3:].T,m.bias[3:]]).astype(float)
    cal=np.load(FOLDER/"train-calibration.npz");d=empty()
    def score(n,selected=None):
        models={};report={};settings={}
        for i,name in enumerate(NAMES):
            options=[selected[name]] if selected else [(r,p,b) for r in preset["shrinkage"] for p in preset["log_prior_weight"] for b in preset["native_mixture"]]
            grid=[]
            for r,p,b in options:
                h=head(d["G"][i],d["count"],h0,r,p,b)
                if selected:models[name]=h;continue
                val,_=view_score(cal[name],cal["y"],cal["ends"],h,m,viterbi);grid.append((val['macro_f1'],val['accuracy'],r,p,b,h,val))
            if not selected:
                best=max(grid,key=lambda q:q[:2]);models[name]=best[-2];settings[name]=best[2:5];report[name]=dict(metrics=best[-1],settings=best[2:5])
        tmp=FOLDER/"lda-tmp";dst=FOLDER/f"lda-{n}.safetensors";save_file(models,str(tmp));tmp.replace(dst)
        if report:dump(FOLDER/f"lda-{n}.json",dict(rows=int(d["count"].sum()),train_calibration_only=True,arms=report))
        return settings
    last={};n=0
    for p in sorted((FOLDER/"shards").glob("*.npz")):
        with np.load(p) as f:
            for k in d:d[k]+=f[k]
        n=int(p.stem.split("-")[1])
        if n in SIZES:last=score(n)
    if n==4792:
        with np.load(FOLDER/"cal-stats.npz") as f:
            for k in d:d[k]+=f[k]
        score(4984,last)
if __name__=="__main__":run()
