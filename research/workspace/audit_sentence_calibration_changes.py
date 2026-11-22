"""Measure which individual sentence decisions move under the affine edit."""
import json

import numpy as np
from safetensors.numpy import load_file

from sentence_emission_calibration import (
    DONOR,EDGE,ONE,OUT,ROOT,ZERO,HierModel,KnownSentences,document,file_hash,
    gold_cases,load_decoder,prepare_score,row_summary,save_json,torch,transition_scores
)
from hier_full import CACHE
from hier_model import LABELS


def main():
    torch.set_num_threads(2)
    baseline=load_file(str(OUT/"known-sentence-role-pair.safetensors"))
    patch=load_file(str(OUT/"known-sentence-emission-calibration-retention.safetensors"))
    vocab=json.loads((EDGE/"vocab.json").read_text())
    model=HierModel(DONOR)
    docs=[]
    blocks=[]
    for k,rr in gold_cases().items():
        d=document(k,rr)
        d["units"]=d["reference_units"]
        sentences=[line for _,line,_ in rr if line.strip()]
        if len(sentences)==len(rr):
            with np.load(CACHE/(k+".npz")) as f:raw=f["context"]
        else:
            raw=model.logits(model.contextualize(model.words(sentences)))
        docs.append(d)
        blocks.append((prepare_score(d,raw,{"exact":baseline}),transition_scores(d["text"],d["units"],vocab,baseline)))
        if len(blocks)%240==0:print("Prepared",len(blocks),flush=True)
    bank=KnownSentences(blocks,baseline,load_decoder()).bind(docs)
    before=bank.decode(ZERO,ONE).astype(np.int64)
    bank.logits=np.ascontiguousarray(bank.logits@patch["emission.weight"]+patch["emission.bias"],dtype=np.float64)
    after=bank.decode(ZERO,ONE).astype(np.int64)
    joint=np.bincount(49*bank.gold+7*before+after,minlength=343).reshape(7,7,7)
    expected=json.loads((ROOT/"results/sentence-emission-calibration.json").read_text())["splits"]["test"]["retention"]
    np.testing.assert_array_equal(joint.sum(2),expected["preceding"]["confusion"])
    np.testing.assert_array_equal(joint.sum(1),expected["candidate"]["confusion"])
    changes=[]
    fixed=broken=0
    for a,left in enumerate(LABELS):
        for b,right in enumerate(LABELS):
            if a==b:
                continue
            f=int(joint[b,a,b]);l=int(joint[a,a,b])
            fixed+=f;broken+=l
            changes.append({"previous":left,"new":right,"changed":int(joint[:,a,b].sum()),
                            "became_correct":f,"became_incorrect":l,"net_correct":f-l})
    audit={"rows":len(before),"cases":len(docs),"changed":int(np.sum(before!=after)),
           "unchanged":int(np.sum(before==after)),"fixed":fixed,"broken":broken,
           "gold_role_axis":LABELS,"counts_gold_previous_new":joint.tolist(),
           "edits":sorted(changes,key=lambda q:-abs(q["net_correct"]))}
    save_json(ROOT/"results/sentence-calibration-change-audit.json",audit)
    print({k:audit[k] for k in ("rows","changed","fixed","broken")},flush=True)
    print(audit["edits"][:5],flush=True)


if __name__=="__main__":
    main()
