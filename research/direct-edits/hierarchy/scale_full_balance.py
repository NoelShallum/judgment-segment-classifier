"""Secondary full inverse-frequency control for each stored corpus-size prefix."""
from pathlib import Path
import argparse
import json
import numpy as np
from safetensors.numpy import save_file
from native_memory import DONOR,HierModel,metric,confusion,dump,file_hash
from scale_program import FOLDER,NAMES,SIZES,compiler,empty,solution,view_score

def reserve():
    p=FOLDER/"equal-role-control-plan.json"
    if p.exists():return json.loads(p.read_text())
    plan=dict(primary_plan=file_hash(FOLDER/"plan.json"),power=1.0,meaning="Each of the seven classes has equal total mass in its weighted Gram and cross-products.",prefixes=[*SIZES,4984],lambda_settings=[.03,.3],gain_settings=[.5,1.25,2.5],purpose="Secondary matched comparison, excluded from primary-model selection. Grid candidates assessed only on separate train-side calibration. Full-size settings reuse those selected at 4792.",registration="Before any corpus-scale calibration or external validation results.")
    dump(p,plan);return plan

def run():
    plan=reserve();viterbi=compiler();m=HierModel(DONOR)
    h0=np.vstack((m.weight[3:].T,m.bias[3:])).astype(float)
    cal=np.load(FOLDER/"train-calibration.npz");stat=empty()
    def fit(n,settings=None):
        scores={};params={}
        for r,name in enumerate(NAMES):
            grid={}
            selected=settings[name] if settings else None
            choices=[selected] if selected else [(a,b) for a in plan["lambda_settings"] for b in plan["gain_settings"]]
            for lam,gain in choices:
                h=solution(stat["G"][r],stat["B"][r],stat["count"],1.,lam,gain,h0)
                if settings:
                    params[name]=h;continue
                s,_=view_score(cal[name],cal["y"],cal["ends"],h,m,viterbi);grid[(lam,gain)]=(s,h)
            if not settings:
                key=max(grid,key=lambda k:(grid[k][0]["macro_f1"],grid[k][0]["accuracy"]));params[name]=grid[key][1];scores[name]=dict(settings=list(key),metrics=grid[key][0])
        if scores:dump(FOLDER/f"equal-role-{n}.json",dict(rows=int(stat["count"].sum()),arms=scores,train_calibration_only=True))
        tmp=FOLDER/"equal-role-tmp";dest=FOLDER/f"equal-role-{n}.safetensors";save_file(params,str(tmp));tmp.replace(dest)
        return {k:v["settings"] for k,v in scores.items()}
    sizes=set(SIZES);last={}
    for p in sorted((FOLDER/"shards").glob("*.npz")):
        with np.load(p) as d:
            for k in stat:stat[k]+=d[k]
        n=int(p.stem.split("-")[1])
        if n in sizes:last=fit(n)
    if n==4792:
        with np.load(FOLDER/"cal-stats.npz") as d:
            for k in stat:stat[k]+=d[k]
        fit(4984,last)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("reserve","run"));q=p.parse_args()
    if q.mode=="reserve":reserve()
    else:run()
