"""Freeze and evaluate a written seven-row replacement for a local instruction decoder's LM head."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from safetensors.numpy import save_file
from associative_edit import correction
from gguf_choice_rows import read_rows
from prepare_projection import TARGET as LETTER_TARGET
from seven_paths import PILOT as DEST
from program_trial import metric,save_json
from weight_edit import LABELS,ROOT,file_hash

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("gguf",type=Path);ap.add_argument("--freeze",action="store_true");ap.add_argument("--semantic",action="store_true");args=ap.parse_args()
    TARGET=ROOT/"gemma-semantic" if args.semantic else LETTER_TARGET
    basename="instruction-semantic" if args.semantic else "instruction-projection"
    dest=ROOT/"checkpoints"/basename
    label_prefix="semantic_" if args.semantic else "instruction_"
    spec=json.loads((ROOT/"gemma_projection_spec.json").read_text())
    if args.semantic:spec={**spec,**json.loads((TARGET/"design.json").read_text())}
    if file_hash(args.gguf)!=spec["sha256"]:raise ValueError("Parent checksum failed")
    rows=json.loads((TARGET/"rows.json").read_text())
    tokens=list(map(int,(TARGET/"token-ids.txt").read_text().split()))
    W,name=read_rows(args.gguf,tokens)
    def collect(group):
        rs=[r for r in rows if r["group"]==group]
        values=np.stack([np.fromfile(TARGET/"vectors"/(r["hash"]+".f32"),dtype="<f4") for r in rs])
        assert values.shape==(len(rs),W.shape[1]+7)
        x,native=values[:,:-7].copy(),values[:,-7:].copy()
        logits=x@W.T
        if not np.isfinite(logits).all():raise ValueError("Invalid hidden vector")
        agree=int(np.sum(logits.argmax(1)==native.argmax(1)))
        err=float(np.max(abs(30*np.tanh(logits/30)-native)))
        # Q8 internal matrix-vector multiplication also quantizes activations.
        # Compare class ranks for every row and report its discrepancies and magnitude.
        return rs,x,native,{"rows":len(rs),"same_native_argmax":agree,"max_abs_softcapped_logit_difference":err}
    record=TARGET/"frozen.json"
    if not record.exists():
        ns,null,_,ncheck=collect("neutral");ks,keys,_,kcheck=collect("keys");sr,sel,_,scheck=collect("select")
        y=np.array([LABELS.index(r["label"]) for r in sr]);ky=np.array([LABELS.index(r["label"]) for r in ks])
        null_logits=(null@W.T).mean(0);null_logits-=null_logits.mean()
        cand=[];options=[]
        for alpha in (0,.25,.5,1):
            bias=(-alpha*null_logits).astype(np.float32)
            anchored=keys@W.T+bias
            top=np.sort(anchored,axis=1)[:,-2:]
            margin=.5*float(np.median(top[:,1]-top[:,0]))
            for damping in (0,.5,1):
                ww,bb=correction(keys,ky,W,bias,margin,damping) if damping else (W,bias)
                pred=(sel@ww.T+bb).argmax(1);m=metric(y,pred)
                change=float(np.linalg.norm(ww-W)+np.linalg.norm(bb))
                score=(m["accuracy"],m["macro_f1"],-change)
                cand.append({"bias_coefficient":alpha,"damping":damping,"margin":margin,"accuracy":m["accuracy"],"macro_f1":m["macro_f1"],"change_norm":change})
                options.append((score,ww,bb))
        best=max(range(len(cand)),key=lambda j:options[j][0])
        best_edit=max((j for j in range(1,len(cand)) if cand[j]["change_norm"]>0),key=lambda j:options[j][0])
        dest.mkdir(parents=True,exist_ok=True)
        for key,j in (("unchanged",0),("edited",best_edit)):
            _,ww,bb=options[j]
            save_file({"classifier.weight":np.ascontiguousarray(ww),"classifier.bias":np.ascontiguousarray(bb)},str(dest/(key+".safetensors")),
                metadata={"source_sha256":spec["sha256"],"type":"AttoFlow Gemma 4 seven-class readout"})
        frozen={"spec":spec,"hidden_width":W.shape[1],"source_tensor":name,"token_ids":tokens,
            "selected_including_parent":cand[best],"selected_actual_edit":cand[best_edit],"development_candidates":cand,
            "projection_parity":{"neutral":ncheck,"keys":kcheck,"select":scheck},
            "prompt_sha256":hashlib.sha256((TARGET/"prefix.txt").read_bytes()+b"\0"+(TARGET/"suffix.txt").read_bytes()).hexdigest(),
            "sha256":{k:file_hash(dest/(k+".safetensors")) for k in ("unedited","edited") if (dest/(k+".safetensors")).exists()}}
        frozen["sha256"]["unchanged"]=file_hash(dest/"unchanged.safetensors")
        if record.exists() and frozen!=json.loads(record.read_text()):raise ValueError("Attempt to change already frozen edit")
        save_json(record,frozen)
        print("Frozen",cand[best_edit],flush=True)
    if args.freeze:return
    from safetensors.numpy import load_file
    frozen=json.loads(record.read_text());cr,cx,native,parity=collect("confirm")
    has_matched=any(r["group"]=="matched" for r in rows)
    if has_matched:mr,mx,mnative,mparity=collect("matched")
    else:mparity=None
    cy=np.array([LABELS.index(r["label"]) for r in cr])
    c={};matched={}
    for k in ("unchanged","edited"):
        f=dest/(k+".safetensors");assert file_hash(f)==frozen["sha256"][k]
        m=load_file(str(f));w,b=m["classifier.weight"],m["classifier.bias"]
        c[k]=metric(cy,(cx@w.T+b).argmax(1))
        if not has_matched:continue
        pred=(mx@w.T+b).argmax(1)
        matched[label_prefix+k]={"sha256":frozen["sha256"][k],"predictions":{r["id"]:LABELS[j] for r,j in zip(mr,pred)}}
    if has_matched:matched[label_prefix+"native"]={"sha256":frozen["spec"]["sha256"],"predictions":{r["id"]:LABELS[j] for r,j in zip(mr,mnative.argmax(1))}}
    c["native"]=metric(cy,native.argmax(1))
    if has_matched:save_json(DEST/(basename+".json"),matched)
    save_json(ROOT/"results"/(basename+"-validation.json"),{"frozen":frozen,"confirmation":c,"confirmation_parity":parity,"matched_parity":mparity})
    print("Confirmation",[(k,round(v["accuracy"],4)) for k,v in c.items()],flush=True)
if __name__=="__main__":main()
