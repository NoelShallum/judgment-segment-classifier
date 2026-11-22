"""Measure a physically written lexical residual head on top of the compact associative BERT edit."""
import json
import numpy as np
from safetensors.numpy import save_file
from graft_rows import data
from matched_local import vectors
from program_trial import metric,official_sample,save_json
from role_cues import FEATURES,cue_features,programmed_rows
from seven_paths import folder_for
from weight_edit import LABELS,ORIGINAL,ROOT,file_hash,tensor
def main():
    parent=ROOT/"checkpoints/associative-direct/model.safetensors"
    w,b=tensor(parent,"classifier.weight"),tensor(parent,"classifier.bias");rows=programmed_rows()
    scores={};best=None;specs=[]
    for fold in ("select","confirm"):
        y,sw,x,_=data(fold);items,_,_=official_sample(fold,160)
        c=cue_features([r[2] for r in items]);original=x["a"]@w.T+b
        if fold=="select":
            for strength in (0,.5,1,2,4,8):
                m=metric(y,(original+c@(rows*strength)).argmax(1),sw)
                specs.append({"coefficient":strength,"accuracy":m["accuracy"],"macro_f1":m["macro_f1"]})
                p=specs[0]
                if m["macro_f1"]>=p["macro_f1"] and (best is None or (m["accuracy"],m["macro_f1"])>(best[0],best[1])):best=(m["accuracy"],m["macro_f1"],strength)
        strength=best[2]
        scores[fold]={"feature_coverage":int(np.sum(c.any(1))),"examples":len(y),"parent":metric(y,original.argmax(1),sw),
                      "written":metric(y,(original+c@(rows*strength)).argmax(1),sw)}
    path=ROOT/"checkpoints/cue-program";path.mkdir(exist_ok=True)
    artifact=path/"cue.safetensors";save_file({"cue.weight":rows*strength},str(artifact))
    report={"program":"Twenty-five authored discourse features with signed zero-sum seven-role rows. The scalar gain is chosen solely on weighted validation accuracy with a development macro-F1 floor at the parent. Inference adds this explicit cue layer to an already edited frozen BERT head. No gradients and no provider supervision.",
            "parent_sha256":file_hash(parent),"cue_sha256":file_hash(artifact),"rules":len(FEATURES),"coefficient":strength,"candidates":specs,"scores":scores,
            "exploratory":"Introduced after seeing aggregate results of earlier families. Claims of external replication require a fresh cohort."}
    save_json(ROOT/"results/cue-program-validation.json",report)
    for group in ("confirmation","paragraphs","homogeneous","pilot"):
        dest=folder_for(group);examples=json.loads((dest/"approved.json").read_text());texts=[r["text"] for r in examples]
        x=vectors(ORIGINAL,texts);c=cue_features(texts)
        pred=(x@w.T+b+c@(rows*strength)).argmax(1)
        save_json(dest/"cues.json",{"cue_head":{"sha256":report["cue_sha256"],"predictions":{r["id"]:LABELS[j] for r,j in zip(examples,pred)}}})
    print("Cue strength",strength,"confirmation",scores["confirm"]["written"]["accuracy"],scores["confirm"]["written"]["macro_f1"],flush=True)
if __name__=="__main__":main()
