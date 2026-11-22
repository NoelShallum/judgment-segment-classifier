"""Verify GPU target pooling and match the small and large referral scorers."""
import json

import numpy as np
import torch

import legal_bert_small_referral as small
import sentence_bert_referral_write as bert
from sentence_document_graph_write import ROOT,OUT,file_hash,load_file,paired,row_summary,save_json


def main():
    torch.set_num_threads(1)
    tok=small.BertTokenizerFast.from_pretrained(small.MODEL,local_files_only=True)
    rows,_=bert.load_views("confirm")
    sample=small.bert.pack_ids(tok,rows,bert.MODES[1])[::103]
    enc,_=small.engine("GPU")
    fast=small.infer(enc,tok,sample)
    base=small.BertModel(small.BertConfig.from_json_file(str(small.MODEL/"config.json"))).eval()
    base.load_state_dict(small.load_torch(small.MODEL/"encoder.safetensors"),strict=True)
    forward=small.Pooled(base).eval()
    ref=[]
    with torch.inference_mode():
        for tokens in sample:
            target=torch.zeros((1,len(tokens)),dtype=torch.long)
            start=max((k+1 for k,c in enumerate(tokens[:-1]) if c==tok.sep_token_id),default=1)
            target[0,min(start,len(tokens)-2):len(tokens)-1]=1
            ref.append(forward(torch.tensor([tokens]),torch.ones_like(target),target).numpy()[0])
    slow=np.stack(ref)
    np.testing.assert_allclose(fast,slow,atol=2e-4,rtol=3e-5)
    folder=ROOT/"results"
    support=folder/(small.TAG+"-frozen.json")
    sp=json.loads(support.read_text())
    sn="target-pooled Legal-BERT Small"
    s=sp["arms"][sn]
    a=OUT/(small.TAG+"-1.safetensors")
    assert s["weights_sha256"]==file_hash(a)
    sw=load_file(str(a))
    sx=np.column_stack((np.concatenate([r[4] for r in rows]),np.load(small.CACHE/"confirm-GPU.npy",allow_pickle=False)))
    sg,sm=s["settings"][-2:]
    small_labels=bert.apply(rows,bert.response(sx,sw),sg,sm)
    src,other,_=bert.encoded("confirm",(bert.ARMS[2],))
    assert all(np.array_equal(x[3],y[3]) and x[0]==y[0] for x,y in zip(rows,src))
    plan=bert.read(bert.TAG+"-frozen")
    k=bert.ARMS[2]
    dest=OUT/(bert.TAG+"-2.safetensors")
    assert file_hash(dest)==plan["checkpoints"][k]
    w=load_file(str(dest))
    variant,*_,gain,margin=plan["frozen_modes"][k]
    large_labels=bert.apply(rows,bert.response(bert.design(rows,other[k],variant,w),w),gain,margin)
    mats=[bert.aggregate(rows,p) for p in (small_labels,large_labels)]
    assert row_summary(mats[0].sum(0))==json.loads((folder/(small.TAG+".json")).read_text())["splits"]["confirm"]["models"][sn]["scores"]
    assert row_summary(mats[1].sum(0))==bert.read(bert.TAG)["splits"]["confirm"]["arms"][k]["scores"]
    record={"freeze_sha256":file_hash(support),"checked_pooling_inputs":len(sample),
        "max_abs_pooling_drift":float(np.max(np.abs(fast-slow))),
        "paired_small_vs_large":paired(*mats),"same_confirmation_cases":len(rows)}
    save_json(folder/(small.TAG+"-qa.json"),record)
    print(record,flush=True)


if __name__=="__main__":
    main()
