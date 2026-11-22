"""Benchmark dynamic INT8 against the pinned FP32 InLegalBERT reviewer."""
import argparse
import json
import time

import numpy as np
import torch
from safetensors.numpy import load_file

import mixedbread_referral_overlay as overlay
import sentence_bert_referral_write as bert
import sentence_phase_readout_write as phase
import sentence_pairwise_consensus as consensus
import sentence_pairwise_repair as repair
from sentence_document_graph_write import OUT, cm, file_hash, paired, row_summary, save_json
from weight_edit import ORIGINAL

P=repair.P
TAG="sentence-bert-int8-review"
WORK=P.ROOT/"runtime"/TAG
ONNX=WORK/"encoder-fp32.onnx"
QUANT=WORK/"encoder-int8.onnx"
META=WORK/"provenance.json"


class Pooled(torch.nn.Module):
    def __init__(self,encoder):
        super().__init__()
        self.encoder=encoder

    def forward(self,input_ids,attention_mask):
        return self.encoder(input_ids=input_ids,attention_mask=attention_mask,return_dict=False)[1]


def prepare():
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType,quantize_dynamic
    from encoder import components

    WORK.mkdir(parents=True,exist_ok=True)
    source=file_hash(ORIGINAL)
    policy=bert.read(bert.TAG+"-frozen")
    assert policy["backbone_weights_sha256"]==source
    if META.is_file():
        old=json.loads(META.read_text())
        assert old["source_weights_sha256"]==source
        assert file_hash(ONNX)==old["fp32_sha256"] and file_hash(QUANT)==old["quantized_sha256"]
        return old
    if not ONNX.is_file():
        model,_,_=components(device="cpu")
        wrapper=Pooled(model.bert).eval()
        sample=(torch.arange(24,dtype=torch.long).reshape(2,12)+1,torch.ones(2,12,dtype=torch.long))
        temp=WORK/"encoder-fp32.part.onnx"
        with torch.inference_mode():
            torch.onnx.export(wrapper,sample,str(temp),input_names=["input_ids","attention_mask"],
                              output_names=["pooled"],opset_version=17,
                              dynamic_axes={"input_ids":{0:"batch",1:"sequence"},"attention_mask":{0:"batch",1:"sequence"},
                                            "pooled":{0:"batch"}},dynamo=False)
        temp.replace(ONNX)
        print("Exported FP32 graph",ONNX.stat().st_size,flush=True)
    stage=WORK/"encoder-int8.part.onnx"
    quantize_dynamic(str(ONNX),str(stage),per_channel=True,reduce_range=True,
                     weight_type=QuantType.QUInt8,op_types_to_quantize=["MatMul","Gemm"])
    stage.replace(QUANT)
    provenance={"source_weights_sha256":source,"fp32_sha256":file_hash(ONNX),"quantized_sha256":file_hash(QUANT),
                "fp32_bytes":ONNX.stat().st_size,"quantized_bytes":QUANT.stat().st_size,
                "onnxruntime":ort.__version__,"weight_type":"QUInt8","per_channel":True,"reduce_range":True}
    save_json(META,provenance)
    print("Wrote INT8 graph",provenance,flush=True)
    return provenance


def make_ort(path,threads):
    import onnxruntime as ort

    options=ort.SessionOptions()
    options.intra_op_num_threads=threads
    options.inter_op_num_threads=1
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    engine=ort.InferenceSession(str(path),sess_options=options,providers=["CPUExecutionProvider"])

    def score(values,mask):
        return engine.run(["pooled"],{"input_ids":values,"attention_mask":mask})[0]

    return score


def features(infer,windows):
    out=np.empty((len(windows),768),np.float32)
    bins=(16,32,64,96)
    groups={k:[] for k in bins}
    for i,row in enumerate(windows):
        groups[next(k for k in bins if len(row)<=k)].append(i)
    for n,group in groups.items():
        batch=min(24,1536//n)
        for a in range(0,len(group),batch):
            ix=group[a:a+batch]
            x=np.zeros((len(ix),n),np.int64)
            m=np.zeros_like(x)
            for j,k in enumerate(ix):
                size=len(windows[k])
                x[j,:size]=windows[k]
                m[j,:size]=1
            out[ix]=infer(x,m)
    assert np.isfinite(out).all()
    return out


def evaluate(splits,threads):
    prov=prepare()
    file=repair.receipt(bert.TAG)
    cfg=json.loads(file.read_text())
    overlay_file=repair.receipt(overlay.TAG)
    policy=json.loads(overlay_file.read_text())
    assert policy["sources"][bert.TAG]==file_hash(file)
    choice=policy["arms"]["BERT only"]["choice"]
    arm=overlay.BERT
    head=OUT/(bert.TAG+"-2.safetensors")
    assert file_hash(head)==cfg["checkpoints"][arm]
    frozen=load_file(str(head))
    mode=cfg["frozen_modes"][arm][0]
    parent,w,p_hash=consensus.inputs()
    assert policy["sources"][repair.TAG]==p_hash
    recipe=repair.load_recipe()
    int8=make_ort(QUANT,threads)
    fp32=make_ort(ONNX,threads)
    for run in (int8,fp32):
        features(run,[[101,2054,102]]*24)
    result={"graph":prov,"thread_count":threads,"bert_freeze_sha256":file_hash(file),
            "overlay_freeze_sha256":file_hash(overlay_file),"splits":{}}

    for split in splits:
        rows,bank,_=bert.encoded(split,(arm,))
        _,tk=bert.load_views(split)
        windows=bert.pack_ids(tk,rows,bert.MODES[1])
        data,_,base,_=phase.view(split,parent,w,recipe)
        np.testing.assert_array_equal(data[0],[r[0] for r in rows])
        np.testing.assert_array_equal(data[2],np.concatenate([r[1] for r in rows]))
        indices=np.concatenate([q+a for a,(_,_,_,q,_,_,_) in zip(data[1],rows)])
        prior=np.concatenate([r[5] for r in rows])
        old=np.concatenate([r[2][r[3]] for r in rows])

        def review(features):
            y=bert.response(bert.design(rows,features,mode,frozen),frozen)
            return overlay.edit(base,indices,prior,y,*choice[:2],old if choice[2] else None)

        cached=bank[arm]
        orig=review(cached)
        expected=P.dense.read(overlay.TAG)["splits"][split]["arms"]["BERT only"]["labels"]
        assert row_summary(cm(data[2],orig))==expected
        old_stats=P.dense.paired_confusion(data,orig)
        out=result["splits"][split]={"cases":len(rows),"referrals":len(indices),"fp32_reference":expected,"methods":{}}
        for name,engine in (("INT8 CPU",int8),("FP32 CPU",fp32)):
            t=time.monotonic()
            val=features(engine,windows)
            elapsed=time.monotonic()-t
            new=review(val)
            stats=P.dense.paired_confusion(data,new)
            delta=val-cached
            summary=row_summary(stats.sum(0))
            out["methods"][name]={"wall_seconds":elapsed,"windows_per_second":len(windows)/elapsed,
                "labels":summary,"paired_to_pinned":paired(stats,old_stats),
                "label_disagreements":int(np.count_nonzero(orig!=new)),
                "mean_abs_pooler_error":float(np.abs(delta).mean()),"peak_pooler_error":float(np.abs(delta).max())}
            print(split,name,len(windows),round(elapsed,2),summary["macro_f1"],
                  out["methods"][name]["label_disagreements"],flush=True)
        save_json(P.RESULT/(TAG+".json"),result)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("splits",nargs="*")
    parser.add_argument("--threads",type=int,default=4)
    args=parser.parse_args()
    evaluate(args.splits or ("confirm","select"),args.threads)
