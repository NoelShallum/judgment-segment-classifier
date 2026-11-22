"""Check unit-feature parity before using the official FP16 ONNX graph."""
import time
import numpy as np
from program_trial import fixture_rows,save_json
from role_fixtures import ANCHORS
from seven_paths import ROOT

def compare(enc,key="gte"):
    texts=[r[2] for r in fixture_rows(ANCHORS)]
    docs=[texts[i:i+7] for i in range(0,len(texts),7)]
    docs.append(["Earlier proceedings provide useful background. " * 96,
                 "The respondent advances a different interpretation. " * 74,
                 "The court now states its conclusion. The appeal is allowed."])
    runtime=enc.runtime
    result={}
    for mode in ("isolated","bidirectional"):
        enc.runtime=None;t=time.monotonic()
        expected=np.concatenate(enc.process(docs,mode));cpu_time=time.monotonic()-t
        enc.runtime=runtime;t=time.monotonic()
        actual=np.concatenate(enc.process(docs,mode));gpu_time=time.monotonic()-t
        similarity=(actual*expected).sum(1)
        absolute=np.max(np.abs(actual-expected))
        result[mode]={"rows":len(actual),"min_cosine_similarity":float(similarity.min()),"max_difference":float(absolute),
                      "cpu_s":cpu_time,"gpu_s":gpu_time}
        print(mode,result[mode],flush=True)
        if similarity.min()<.999 or absolute>.02:
            raise ValueError("FP16 and source CPU unit representations failed the parity gate")
    save_json(ROOT/"results"/(key+"-openvino-parity.json"),result)
