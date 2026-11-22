"""Write a low-rank additive head correction for explicitly specified authored behaviors."""
import argparse
import json
import numpy as np
from scipy.linalg import solve

from encoder import components, write_config
from program_trial import fixture_rows, get_features, metric, official_sample, paired_document_interval, save_json
from role_fixtures import ANCHORS,CHECKS
from weight_edit import LABELS, ROOT, file_hash, patched_checkpoint, tensor


SETTINGS=tuple((margin, damping) for margin in (1.5,3.0,5.0) for damping in (.01,.1,.5))


def correction(keys,desired,w,b,margin,damping):
    # A specified output should exceed the present strongest rival by 'margin'.
    # Correcting both rows preserves the mean logit of each authored example.
    logits=keys@w.T+b
    masked=logits.copy();masked[np.arange(len(keys)),desired]=-np.inf
    rival=masked.argmax(1)
    needed=np.maximum(0,margin+logits[np.arange(len(keys)),rival]-logits[np.arange(len(keys)),desired])
    residual=np.zeros_like(logits,dtype=np.float64)
    residual[np.arange(len(keys)),desired]=needed/2
    residual[np.arange(len(keys)),rival]=-needed/2
    # Subtract the common key component so the edit acts on contrasts between
    # anchors. The intercept restores the original feature-space origin.
    center=keys.mean(axis=0,dtype=np.float64)
    centered=keys.astype(np.float64)-center
    gram=centered@centered.T
    regularization=damping*float(gram.trace()/len(keys))
    gram.flat[::len(keys)+1]+=regularization
    directions=(centered.T@solve(gram,residual,assume_a="pos")).T
    intercept=-(directions@center)
    return (w+directions).astype(np.float32),(b+intercept).astype(np.float32)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--parent",choices=["direct","context"],default="direct")
    parser.add_argument("--backend",choices=["intel","torch"],default="torch")
    args=parser.parse_args()
    from mixed_encoder_trial import CONTEXT,get_vectors
    base=CONTEXT if args.parent=="context" else ROOT/"checkpoints/selected-direct/model.safetensors"
    w,b=tensor(base,"classifier.weight"),tensor(base,"classifier.bias")
    rows,weights,stats=official_sample("select",160)
    crows,cweights,cstats=official_sample("confirm",160)
    groups={"authored-anchors":fixture_rows(ANCHORS),"authored-checks":fixture_rows(CHECKS),"select-160":rows,"confirm-160":crows}
    if args.parent=="context":
        vectors=get_vectors(CONTEXT,groups,args.backend)
    else:
        model,tok,device=components()
        vectors={name:get_features(model,tok,device,items,name)[0] for name,items in groups.items()}
    targets={name:np.array([r[3] for r in items]) for name,items in groups.items()}
    keys=np.concatenate([vectors["authored-anchors"],vectors["authored-checks"]])
    desired=np.concatenate([targets["authored-anchors"],targets["authored-checks"]])
    x,y=vectors["select-160"],targets["select-160"]
    parent_score=metric(y,(x@w.T+b).argmax(1),weights)
    priority="accuracy" if args.parent=="context" else "macro_f1"
    best=(parent_score[priority],0,0,w,b)
    candidates=[]
    for margin,damping in SETTINGS:
        ww,bb=correction(keys,desired,w,b,margin,damping)
        pred=(x@ww.T+bb).argmax(1)
        m=metric(y,pred,weights)
        candidates.append({"margin":margin,"damping":damping,"selection":m,
                           "authored_key_accuracy":float(np.mean((keys@ww.T+bb).argmax(1)==desired))})
        if m["macro_f1"]>=parent_score["macro_f1"] and m[priority]>best[0]+1e-12:
            best=(m[priority],margin,damping,ww,bb)
    result,margin,damping,ww,bb=best
    dest=ROOT/"checkpoints"/("associative-"+args.parent)
    changes=patched_checkpoint(base,dest/"model.safetensors",{"classifier.weight":ww,"classifier.bias":bb})
    write_config(dest)
    details={"method":"Analytical low-rank key-to-logit correction to the frozen parent head",
             "selection_priority":priority,"parent":args.parent,"parent_sha256":file_hash(base),"target_sha256":file_hash(dest/"model.safetensors"),
             "role_keys":len(keys),"selected_margin":margin,"selected_damping":damping,
             "changes":changes,"labels":LABELS,"gradient_steps":0,"teacher_outputs_used":False}
    save_json(dest/"edit.json",details)
    cx,cy=vectors["confirm-160"],targets["confirm-160"]
    before=(cx@w.T+b).argmax(1);after=(cx@ww.T+bb).argmax(1)
    report={"method":details,"selection_stats":stats,"confirmation_stats":cstats,"candidates":candidates,
            "selection_original":parent_score,"selected_development_score":result,"confirmation_parent":metric(cy,before,cweights),
            "confirmation":metric(cy,after,cweights),"paired_gain_interval":paired_document_interval(crows,cy,before,after,cweights),
            "design":"The 84 authored anchors and diagnostic cases are programming examples. Every desired logit margin is specified from their original authored role. Solve the regularized key system analytically and write two tensors. The baseline is eligible if all edits degrade validation. Official test text, public test predictions and Gemini answers are excluded from programming and selection."}
    save_json(ROOT/"results"/("associative-"+args.parent+"-validation.json"),report)
    print("Analytical edit",margin,damping,"confirmed",report["confirmation"]["macro_f1"],"selection",result,flush=True)


if __name__=="__main__":main()
