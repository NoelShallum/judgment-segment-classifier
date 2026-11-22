"""Run the frozen projection on the two additional paragraph tests and case-disjoint sentence set."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from seven_paths import ROOT,folder_for,LABELS
from seven_paragraphs import dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument("model",type=Path);ap.add_argument("libraries",type=Path);ap.add_argument("--threads",type=int,default=3);ap.add_argument("--semantic",action="store_true");args=ap.parse_args()
    pilot=ROOT/("gemma-semantic" if args.semantic else "gemma-projection");extra=ROOT/("gemma-semantic-extra" if args.semantic else "gemma-extra")
    basename="instruction-semantic" if args.semantic else "instruction-projection"
    group_order=("pilot","paragraphs","homogeneous","confirmation") if args.semantic else ("paragraphs","homogeneous","confirmation")
    extra.mkdir(exist_ok=True);(extra/"texts").mkdir(exist_ok=True)
    for n in (("prefix.txt","suffix.txt","choices.txt") if args.semantic else ("prefix.txt","suffix.txt")):(extra/n).write_bytes((pilot/n).read_bytes())
    prefix=(extra/"prefix.txt").read_text();suffix=(extra/"suffix.txt").read_text();all_rows=[]
    for group in group_order:
        for r in json.loads((folder_for(group)/"approved.json").read_text()):
            key=hashlib.sha256((prefix+"\0"+r["text"]+"\0"+suffix).encode()).hexdigest()
            (extra/"texts"/(key+".txt")).write_text(r["text"])
            all_rows.append({"id":r["id"],"hash":key,"group":group})
    dump(extra/"rows.json",all_rows);(extra/"manifest.txt").write_text("\n".join(dict.fromkeys(r["hash"] for r in all_rows))+"\n")
    expected={s for s in (pilot/"manifest.txt").read_text().splitlines() if s}
    while not all((pilot/"vectors"/(h+".f32")).is_file() for h in expected):
        time.sleep(15)
    env={**os.environ,"OPENBLAS_NUM_THREADS":"1","ROLE_CPU_THREADS":"1"}
    subprocess.run([sys.executable,str(ROOT/"score_projection.py"),str(args.model)]+(["--semantic"] if args.semantic else []),env=env,check=True)
    subprocess.run([sys.executable,str(ROOT/"matched_metrics.py"),"--corpus","pilot"],env=env,check=True)
    if args.semantic:
        frozen=json.loads((pilot/"frozen.json").read_text())["selected_actual_edit"]
        if frozen["accuracy"]<=10/56 or frozen["macro_f1"]<=0.1171:
            print("Semantic variant declined by its prespecified development criteria",flush=True)
            return
    runner=subprocess.Popen([str(ROOT/"runtime"/("project-semantic" if args.semantic else "project-hidden")),str(args.model),str(args.libraries),str(extra),str(args.threads),"2048"],stdout=(extra/"runner.log").open("w"),stderr=subprocess.STDOUT)
    import numpy as np
    from safetensors.numpy import load_file
    record=json.loads((pilot/"frozen.json").read_text());heads={}
    for n in ("unchanged","edited"):
        h=load_file(str(ROOT/"checkpoints"/basename/(n+".safetensors")))
        heads[n]=(h["classifier.weight"],h["classifier.bias"])
    pending=set(group_order)
    while pending:
        for group in tuple(pending):
            chosen=[r for r in all_rows if r["group"]==group];dest=folder_for(group)
            if not all((extra/"vectors"/(r["hash"]+".f32")).is_file() for r in chosen):continue
            values=np.stack([np.fromfile(extra/"vectors"/(r["hash"]+".f32"),dtype="<f4") for r in chosen])
            raw=values[:,:-7];native=values[:,-7:].argmax(1);answers={}
            preds={n:(raw@w.T+b).argmax(1) for n,(w,b) in heads.items()};preds["native"]=native
            for n,pred in preds.items():
                answers[("semantic_" if args.semantic else "instruction_")+n]={"sha256":record["spec"]["sha256"] if n=="native" else record["sha256"][n],
                    "predictions":{r["id"]:LABELS[i] for r,i in zip(chosen,pred)}}
            dump(dest/(basename+".json"),answers)
            subprocess.run([sys.executable,str(ROOT/"matched_metrics.py"),"--corpus",group],check=True,env=env)
            print("Completed frozen instruction evaluation",group,flush=True);pending.remove(group)
        if pending and runner.poll() is not None and runner.returncode!=0:raise RuntimeError("Local decoder exited before completing all inputs")
        if pending:time.sleep(10)
    runner.wait()
    if runner.returncode:raise RuntimeError("Local decoder failed")
if __name__=="__main__":main()
