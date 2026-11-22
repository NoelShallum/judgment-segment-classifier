"""Run the locally available InLegalBERT reviewer on a wider, matched referral budget."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
import time

import numpy as np
from safetensors.numpy import load_file

import mixedbread_referral_overlay as overlay
import sentence_bert_referral_write as B
import sentence_referral_actual_replay as replay
import sentence_referral_utility_write as utility
import sentence_pairwise_repair as repair
from sentence_document_graph_write import OUT, file_hash, cm, paired, row_summary, save_json, torch

P=repair.P
TAG="sentence-wide-bert-benchmark"
SPLITS=("select","confirm","test","holdout")
RATE=.20
CHUNK=384


def gather(spec):
    docs,positions,split=spec
    torch.set_num_threads(1)
    model=B.HierModel(B.DONOR)
    cfg=B.load_file(str(B.PARENT))
    vocab=json.loads((B.EDGE/"vocab.json").read_text())
    return {d["id"]:B.case_view(d,v,model,cfg,vocab,q)
            for d,v,q in zip(docs,B.aligned_embeddings(model,docs,split),positions)}


def feature_bank(bank,split,union):
    docs=P.docs(split)
    np.testing.assert_array_equal(bank[0],[d["id"] for d in docs])
    offsets=bank[1]
    positions=[union[(union>=a)&(union<b)]-a for a,b in zip(offsets,offsets[1:])]
    with ProcessPoolExecutor(4) as workers:
        found={k:v for block in workers.map(gather,[(docs[j::4],positions[j::4],split) for j in range(4)])
               for k,v in block.items()}
    rows=[found[d["id"]] for d in docs]
    np.testing.assert_array_equal(np.concatenate([r[1] for r in rows]),bank[2])
    np.testing.assert_array_equal(np.concatenate([r[2] for r in rows]),bank[3])
    return rows


def encode(rows,split,device,digest):
    from openvino_worker import embed_ids,get_engine
    old,embeddings,proof=B.encoded(split,(overlay.BERT,))
    tok=B.load_views(split)[1]
    current=B.pack_ids(tok,rows,B.MODES[1])
    previous=B.pack_ids(tok,old,B.MODES[1])
    old_features=embeddings[overlay.BERT]
    known=np.full(len(current),-1,dtype=np.int64)
    k=j=0
    for a,b in zip(rows,old):
        assert a[0]==b[0]
        q,r=a[3],b[3]
        match={int(key):j+i for i,key in enumerate(r)}
        for i,pos in enumerate(q):
            if int(pos) in match:
                z=match[int(pos)]
                assert current[k+i]==previous[z]
                known[k+i]=z
        k+=len(q)
        j+=len(r)
    assert (k,j)==(len(current),len(previous))
    enc_meta=json.loads((B.BERT_ART/"source.json").read_text())
    assert file_hash(B.BERT_ART/"encoder.xml")==enc_meta["xml_sha256"]
    assert file_hash(B.BERT_ART/"encoder.bin")==enc_meta["encoder_bin_sha256"]
    missing=np.flatnonzero(known<0)
    merged=np.empty((len(current),768),np.float32)
    merged[known>=0]=old_features[known[known>=0]]
    # Fix the window order and original encoder provenance so a job can resume by block.
    sig=sha256(json.dumps([current,known.tolist(),enc_meta,digest,proof[overlay.BERT]["features_sha256"]]).encode()).hexdigest()
    where=P.ROOT/"cache"/TAG/(split+"-"+sig[:16]+"-"+device.lower())
    where.mkdir(exist_ok=True,parents=True)
    engine=None
    start=time.monotonic()
    print(split,"union",len(current),"reused",len(current)-len(missing),"new",len(missing),
          "cache",where,flush=True)
    for q in range(0,len(missing),CHUNK):
        idx=missing[q:q+CHUNK]
        chunk=where/f"{q:06d}.npy"
        if chunk.is_file():
            x=np.load(chunk,allow_pickle=False)
        else:
            if engine is None:
                engine,_=get_engine(device)
            x=embed_ids(engine,[current[i] for i in idx],max_length=B.LIMIT)
            stage=chunk.with_suffix(".part")
            with stage.open("wb") as f:
                np.save(f,x,allow_pickle=False)
            stage.replace(chunk)
        assert x.shape==(len(idx),768) and np.isfinite(x).all()
        merged[idx]=x
        print(split,"new windows available",q+len(idx),"/",len(missing),
              "seconds",round(time.monotonic()-start,1),flush=True)
    return merged,{"union":len(current),"reused":len(current)-len(missing),"new":len(missing),
                   "input_sha256":sig,"embedding_sha256":sha256(merged.tobytes()).hexdigest(),
                   "weights":enc_meta["source"],"device":device}


def evaluate(device,splits):
    overlay_file=repair.receipt(overlay.TAG)
    policy=json.loads(overlay_file.read_text())
    rule=policy["arms"]["BERT only"]["choice"]
    bert_file=repair.receipt(B.TAG)
    old=json.loads(bert_file.read_text())
    path=OUT/(B.TAG+"-2.safetensors")
    assert file_hash(path)==old["checkpoints"][overlay.BERT]
    w=load_file(str(path))
    dest=P.RESULT/(TAG+".json")
    info={"bert_freeze_sha256":file_hash(bert_file),"local_overlay_sha256":file_hash(overlay_file),
          "router_freeze_sha256":file_hash(repair.receipt(utility.TAG)),"rate":RATE}
    all_results={"method":info,"splits":{}}
    if dest.is_file():
        cached=json.loads(dest.read_text())
        assert cached["method"]==info
        all_results=cached
    for split in splits:
        bank,base,ranks,src=replay.risk(split)
        assert info["router_freeze_sha256"]==src
        positions={name:utility.route(bank[1],score,RATE) for name,score in ranks.items()}
        assert len(positions["confidence"])==len(positions["utility"])
        union=np.union1d(*positions.values())
        rows=feature_bank(bank,split,union)
        embeddings,receipt=encode(rows,split,device,src)
        kind=old["frozen_modes"][overlay.BERT][0]
        score=B.response(B.design(rows,embeddings,kind,w),w)
        prior=np.concatenate([r[5] for r in rows])
        reference=np.concatenate([r[2][r[3]] for r in rows])
        z=overlay.edit(base,union,prior,score,*rule[:2],reference if rule[2] else None)
        labels=replay.report(bank,base,z,ranks,RATE)
        out={"cases":len(bank[0]),"baseline":row_summary(cm(bank[2],base)),
             "embedding_provenance":receipt,"comparison":labels}
        all_results["splits"][split]=out
        save_json(dest,all_results)
        print("Matched",split,{k:r["labels"]["macro_f1"] for k,r in labels["routes"].items()},flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--device",choices=("CPU","GPU"),default="GPU")
    cli.add_argument("splits",nargs="*")
    args=cli.parse_args()
    evaluate(args.device,args.splits or SPLITS)
