"""Check and run the publisher's GTE ModernBERT token encoder on Intel graphics."""
from pathlib import Path
import numpy as np
import openvino as ov

class ModernBertOpenVINO:
    def __init__(self, root: Path, from_source=False, key="gte"):
        child=root/"runtime"/(key+"-source-fp32-gpu" if from_source else "gte-fp16-gpu")
        child.mkdir(parents=True,exist_ok=True)
        core=ov.Core()
        path=(root/"runtime"/(key+"-cpu-equivalent")/"encoder.xml" if from_source else root/"references/gte-modernbert/encoder-fp16.onnx")
        graph=core.read_model(path)
        settings={"INFERENCE_PRECISION_HINT":"f32" if from_source else "f16","PERFORMANCE_HINT":"LATENCY","NUM_STREAMS":"1","CACHE_DIR":str(child)}
        self.compiled=core.compile_model(graph,"GPU",settings)
    def __call__(self, pack):
        ids=pack["input_ids"].numpy()
        mask=pack["attention_mask"].numpy()
        return self.compiled({"input_ids":ids,"attention_mask":mask})[0]
