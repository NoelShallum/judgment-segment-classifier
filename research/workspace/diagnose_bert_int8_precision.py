"""Measure how weight type and range affect pooled InLegalBERT features."""
import json
import time

import numpy as np
from onnxruntime.quantization import QuantType, quantize_dynamic

import bench_bert_int8_review as benchmark
import mixedbread_referral_overlay as overlay
import sentence_bert_referral_write as bert
from sentence_document_graph_write import file_hash, save_json


def main():
    source=benchmark.prepare()
    assert file_hash(benchmark.ONNX)==source["fp32_sha256"]
    rows,banks,_=bert.encoded("confirm",(overlay.BERT,))
    _,tokenizer=bert.load_views("confirm")
    windows=bert.pack_ids(tokenizer,rows,bert.MODES[1])
    index=np.sort(np.random.default_rng(976).choice(len(windows),32,replace=False))
    sample=[windows[i] for i in index]
    target=banks[overlay.BERT][index]
    trials=(("unsigned-reduced",QuantType.QUInt8,True),
            ("unsigned-full",QuantType.QUInt8,False),
            ("signed-reduced",QuantType.QInt8,True))
    record={"source":source,"cohort":"confirm","positions":index.tolist(),"results":{}}
    for name,kind,reduced in trials:
        graph=benchmark.QUANT if name=="unsigned-reduced" else benchmark.WORK/(name+".onnx")
        if name!="unsigned-reduced":
            stage=graph.with_suffix(".part.onnx")
            quantize_dynamic(str(benchmark.ONNX),str(stage),per_channel=True,reduce_range=reduced,
                             weight_type=kind,op_types_to_quantize=["MatMul","Gemm"])
            stage.replace(graph)
        infer=benchmark.make_ort(graph,2)
        start=time.monotonic()
        vec=benchmark.features(infer,sample)
        error=np.abs(vec-target)
        result={"mean_abs_pooler_error":float(error.mean()),"peak_pooler_error":float(error.max()),
                "graph_size_bytes":graph.stat().st_size,"graph_sha256":file_hash(graph),
                "seconds":time.monotonic()-start}
        print(name,json.dumps(result),flush=True)
        record["results"][name]=result
    save_json(benchmark.P.RESULT/"sentence-bert-int8-precision.json",record)


if __name__=="__main__":
    main()
