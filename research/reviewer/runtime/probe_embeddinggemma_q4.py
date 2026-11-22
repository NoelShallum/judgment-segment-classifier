"""Check the pinned EmbeddingGemma ONNX encoder on CPU before classification work."""
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parent
REVISION="5090578d9565bb06545b4552f76e6bc2c93e4a66"
ARTIFACT=ROOT/"references/embeddinggemma-q4"
OUTPUT=ROOT/"results/embeddinggemma-q4-cpu-probe.json"
PROMPT="task: classification | query: "


class Encoder:
    def __init__(self,cap=192,threads=2):
        source=ARTIFACT/"onnx/model_q4.onnx"
        tokenizer=Tokenizer.from_file(str(ARTIFACT/"tokenizer.json"))
        tokenizer.enable_padding(pad_id=0,pad_token="<pad>")
        tokenizer.enable_truncation(max_length=cap)
        opt=ort.SessionOptions()
        opt.intra_op_num_threads=threads
        opt.inter_op_num_threads=1
        self.runtime=ort.InferenceSession(str(source),sess_options=opt,providers=["CPUExecutionProvider"])
        self.tokenizer=tokenizer

    def encode(self,texts):
        rows=self.tokenizer.encode_batch([PROMPT+s for s in texts])
        inputs={"input_ids":np.asarray([r.ids for r in rows],dtype=np.int64),
                "attention_mask":np.asarray([r.attention_mask for r in rows],dtype=np.int64)}
        return self.runtime.run(["sentence_embedding"],inputs)[0]


def main():
    proof=json.loads((ARTIFACT/"provenance.json").read_text())
    assert proof["revision"]==REVISION
    for entry in proof["files"]:
        path=ARTIFACT/entry["file"]
        assert path.stat().st_size==entry["bytes"]
        with path.open("rb") as body:
            assert hashlib.file_digest(body,"sha256").hexdigest()==entry["sha256"]
    sentences=[
        "The petition is dismissed. No order as to costs.",
        "The appeal is therefore allowed and the impugned order is set aside.",
        "The claimant has argued that an officer acted outside his authority.",
        "The record shows that the parties executed a deed on 19 March.",
        "We must determine whether the petition was filed within time.",
        "The judgment contains the following submissions by learned counsel. " * 14,
    ]
    start=perf_counter()
    model=Encoder()
    initial=perf_counter()-start
    a=model.encode(sentences)
    b=np.concatenate([model.encode([s]) for s in sentences])
    assert a.shape==b.shape==(len(sentences),768)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    norm=np.linalg.norm(a,axis=1)
    assert np.allclose(norm,1.,atol=1e-5)
    cos=(a*b).sum(1)/(norm*np.linalg.norm(b,axis=1))
    assert cos.min()>.999
    timing={}
    for n in (1,len(sentences)):
        seconds=[]
        for _ in range(3):
            now=perf_counter()
            model.encode(sentences[:n])
            seconds.append(perf_counter()-now)
        timing[str(n)]=float(np.median(seconds))
    report={"provenance":proof,"output_dimensions":768,"runtime":ort.__version__,"threads":2,
            "probe_count":len(sentences),"batch_vs_individual_min_cosine":float(cos.min()),
            "batch_vs_individual_max_absolute_difference":float(abs(a-b).max()),
            "startup_seconds":initial,"median_seconds_by_batch_size":timing,
            "prompt":PROMPT,"probe_max_tokens":192,"task_metrics_available":False}
    OUTPUT.write_text(json.dumps(report,indent=2)+"\n")
    print(report["batch_vs_individual_min_cosine"],timing,flush=True)


if __name__=="__main__":
    main()
