"""Verify the fitted document calibrator through the normal CPU sentence interface."""
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np
from safetensors.numpy import save_file

from sentence_document_graph_write import OUT,ROOT,EDGE,document,gold_cases,load_file,file_hash,row_summary,save_json
from verify_sentence_local_readout_portable import replay

DEST = OUT/"sentence-document-calibrated-portable.safetensors"
SOURCE = OUT/"sentence-document-readout-portable.safetensors"
PATCH = OUT/"sentence-readout-fusion-document-only.safetensors"


def shard(args):
    return replay(*args,checkpoint=DEST)


def main():
    folder = ROOT/"results"
    frozen = folder/"sentence-readout-fusion-frozen.json"
    design = json.loads(frozen.read_text())
    result = json.loads((folder/"sentence-readout-fusion.json").read_text())
    parent_receipt = json.loads((folder/"sentence-document-readout-portable-qa.json").read_text())
    assert parent_receipt["weights_sha256"] == file_hash(SOURCE)
    assert result["freeze_sha256"] == file_hash(frozen)
    assert design["fitted_weights_sha256"][PATCH.name] == file_hash(PATCH)
    w = load_file(str(PATCH))
    gain,margin,protect = design["choices"]["document only"][-3:]
    edit = {"document_calibration.center":w["center"],"document_calibration.scale":w["scale"],
            "document_calibration.matrix":w["coeff"],"document_calibration.bias":w["bias"],
            "document_calibration.settings":np.asarray([gain,margin,float(protect)])}
    contents = {**load_file(str(SOURCE)),**edit}
    save_file(contents,str(DEST))
    read = load_file(str(DEST))
    for k,v in contents.items():
        np.testing.assert_array_equal(read[k],v)
    docs = [document(k,v) for k,v in gold_cases().items()]
    live = {r["case"] for r in json.loads((EDGE/"qa.json").read_text())["cases"]}
    with ProcessPoolExecutor(4) as pool:
        parts = list(pool.map(shard,[(docs[i::4],live) for i in range(4)]))
    observed = row_summary(sum(c for c,_ in parts))
    checked = [v for _,r in parts for v in r]
    assert live == {v["case"] for v in checked}
    assert observed == result["splits"]["test"]["arms"]["document only"]["scores"]
    save_json(folder/"sentence-document-calibrated-portable-qa.json",
        {"scores":observed,"cases":len(docs),"live_cases":checked,
        "weights_sha256":file_hash(DEST),"parent_sha256":file_hash(SOURCE),
        "fit_weights_sha256":file_hash(PATCH),"freeze_sha256":file_hash(frozen),
        "tensors":{k:list(v.shape) for k,v in edit.items()}})
    print("Verified portable document calibration",observed["macro_f1"],observed["accuracy"],flush=True)


if __name__=="__main__":
    main()
