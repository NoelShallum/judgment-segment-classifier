"""Install selected role rows from a compatible public head using direct tensor arithmetic."""
import json
from itertools import product
import numpy as np
from encoder import fingerprint,write_config
from mixed_encoder_trial import CONTEXT
from program_trial import metric,official_sample,save_json
from seven_paths import folder_for
from weight_edit import ORIGINAL,LABELS,ROOT,file_hash,patched_checkpoint,tensor
ROLE_GROUPS={"all":list(range(7)),"speaker":[3,4],"narrative":[0,1],"discourse":[2,5,6]}
PATHS={"a":ORIGINAL,"a_edit":ROOT/"checkpoints/associative-direct/model.safetensors","b":CONTEXT,"b_edit":ROOT/"checkpoints/associative-context/model.safetensors"}
def data(split):
    rows,weights,info=official_sample(split,160);text=[r[2] for r in rows];f=fingerprint(text)
    a=next((ROOT/"cache").glob("legalseg-current-sentence-fp32-v1-"+split+"-160-"+f[:16]+".npy"))
    b=ROOT/"cache"/("mixed-"+file_hash(CONTEXT)[:15]+"-"+f[:20]+"-intel.npy")
    return np.asarray([r[3] for r in rows]),weights,{"a":np.load(a),"b":np.load(b)},info
def main():
    sy,sw,sx,sinfo=data("select");cy,cw,cx,cinfo=data("confirm")
    heads={k:(tensor(p,"classifier.weight"),tensor(p,"classifier.bias")) for k,p in PATHS.items()}
    records=[];chosen=None;best_edited=None
    for name,(w,b) in heads.items():
        enc=name[0];x=sx[enc];parent=metric(sy,(x@w.T+b).argmax(1),sw)
        def add(ww,bb,description):
            nonlocal chosen,best_edited
            m=metric(sy,(x@ww.T+bb).argmax(1),sw)
            row={**description,"backbone":enc,"parent":name,"accuracy":m["accuracy"],"macro_f1":m["macro_f1"]}
            records.append(row)
            if m["macro_f1"]+1e-10<parent["macro_f1"]:return
            rank=(m["accuracy"],m["macro_f1"])
            if chosen is None or rank>chosen[0]:chosen=(rank,row,ww,bb)
            if description["coefficient"]>0 and (best_edited is None or rank>best_edited[0]):best_edited=(rank,row,ww,bb)
        add(w,b,{"donor":name,"role_group":"all","part":"control","coefficient":0})
        for donor,(dw,db) in heads.items():
            if donor[0]==enc:continue
            for (group,idx),part,alpha in product(ROLE_GROUPS.items(),("bias","rows","rows_and_bias"),(.25,.5,.75,1.)):
                nw=w.copy();nb=b.copy()
                if part!="bias":nw[idx]+=alpha*(dw[idx]-w[idx])
                if part!="rows":nb[idx]+=alpha*(db[idx]-b[idx])
                add(nw,nb,{"donor":donor,"role_group":group,"part":part,"coefficient":alpha})
    rank,spec,ww,bb=best_edited
    parent=PATHS[spec["parent"]];dest=ROOT/"checkpoints/graft-direct"
    changes=patched_checkpoint(parent,dest/"model.safetensors",{"classifier.weight":ww,"classifier.bias":bb});write_config(dest)
    headhash=file_hash(dest/"model.safetensors");enc=spec["backbone"]
    before=(cx[enc]@heads[spec["parent"]][0].T+heads[spec["parent"]][1]).argmax(1)
    after=(cx[enc]@ww.T+bb).argmax(1)
    report={"target_sha256":headhash,"parent_sha256":file_hash(parent),"source_encoder":enc,"changes":changes,"selected_actual_edit":spec,
        "top_including_unchanged":chosen[1],"number_of_variants":len(records),"selection_cases":sinfo,"confirmation_cases":cinfo,
        "confirmation_parent":metric(cy,before,cw),"confirmation":metric(cy,after,cw),"variants":records,
        "protocol":"Raw classifier row and bias grafts between existing compatible encoders. Search is restricted to authored role groups and four fixed fractions. Rank on weighted development accuracy subject to retaining same-parent development macro-F1. Report an untouched confirmation fold. There is no optimizer or gradient. The family was introduced after earlier aggregate pilot results and is exploratory."}
    save_json(dest/"edit.json",{k:report[k] for k in ("target_sha256","parent_sha256","source_encoder","changes","selected_actual_edit")});save_json(ROOT/"results/graft-validation.json",report)
    print(spec,"confirmation",report["confirmation"]["accuracy"],report["confirmation"]["macro_f1"],flush=True)
    for group in ("pilot","confirmation","paragraphs","homogeneous"):
        folder=folder_for(group);items=json.loads((folder/"approved.json").read_text());texts=[r["text"] for r in items]
        base=ORIGINAL if enc=="a" else CONTEXT
        f=ROOT/"cache"/("matched-windows-"+file_hash(base)[:16]+"-"+fingerprint(texts)[:20]+".npy")
        if not f.is_file():continue
        pred=(np.load(f)@ww.T+bb).argmax(1)
        save_json(folder/"graft.json",{"grafted_head":{"sha256":headhash,"predictions":{r["id"]:LABELS[i] for r,i in zip(items,pred)}}})
if __name__=="__main__":main()
