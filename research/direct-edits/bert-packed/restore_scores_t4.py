"""Recover the original selector matrices and replay cached features without fitting or GPU inference."""

from pathlib import Path
import shutil
import sys
from zipfile import ZIP_DEFLATED, ZipFile

state = Path("/content/attoflow-inlegalbert-packed")
data = state/"data"
deps = Path("/content/attoflow-inlegalbert-value/dependencies")
if not (deps/"transformers").exists():
    deps = state/"dependencies"
sys.path[:0] = [str(data),str(deps)]

import numpy as np
from safetensors.numpy import load_file, save_file
from fit import EXTRA, auxiliary, confusion, evaluate, gather, join, loader, metrics, offsets, values
from native import read, sha, write

source = state/"run"
cfg = read(source/"freeze.json")
assert sha(data/"protocol.json") == cfg["plan_sha256"]
bad = source/"analytic-heads.safetensors"
assert cfg["weights_sha256"] == sha(bad) and not cfg.get("serialized_weights_roundtrip")
incoming = load_file(str(bad))
params = {key:np.ascontiguousarray(v.ravel(order="C").reshape(v.shape,order="F"))
          if key.endswith(".weight") else np.ascontiguousarray(v) for key,v in incoming.items()}
initial_patch = source/"ordered-native-head.safetensors"
assert cfg["parent_classifier"]["patch_sha256"] == sha(initial_patch)
native = load_file(str(initial_patch))
np.testing.assert_array_equal(params["ordered.native.weight"].T,native["classifier.weight"])
np.testing.assert_array_equal(params["ordered.native.bias"],native["classifier.bias"])

select = gather(data,state/"features","select")
assert [q["id"] for q in select] == cfg["select_cases"]
y = join(select,"gold")
assert len(y) == cfg["select_rows"]
extra = auxiliary(select,params["lexical.counts"],False)
ends = offsets(select)
decode = loader()
proof = {}
for view, variants in cfg["settings"].items():
    for variant,chosen in variants["selected"].items():
        k = view+"."+variant
        x = values(select,view,chosen["fraction_last"],extra)[:,EXTRA[variant]]
        logits = x@params[k+".weight"]+params[k+".bias"]
        prediction = decode(logits,params["crf.transitions"],ends,chosen["transition"])
        measured = metrics(confusion(y,prediction))
        assert np.isclose(measured["macro_f1"],chosen["select_macro_f1"],rtol=0,atol=1e-12),(k,measured,chosen)
        assert np.isclose(measured["accuracy"],chosen["select_accuracy"],rtol=0,atol=1e-12),(k,measured,chosen)
        proof[k] = measured
print("Recovered the frozen selector metrics exactly for all",len(proof),"arms",flush=True)

out = state/"run-corrected"
out.mkdir(exist_ok=True)
matrix = out/"analytic-heads.safetensors"
save_file(params,str(matrix.with_suffix(".part")))
matrix.with_suffix(".part").replace(matrix)
reopened = load_file(str(matrix))
for key,v in params.items():
    np.testing.assert_array_equal(reopened[key],v)
shutil.copyfile(initial_patch,out/"ordered-native-head.safetensors")
cfg["weights_sha256"] = sha(matrix)
cfg["serialized_weights_roundtrip"] = True
cfg["serialization_correction"] = dict(
    original_run_freeze_sha256=sha(source/"freeze.json"), incorrect_archive_weights_sha256=sha(bad),
    explanation="The frozen classifier matrices came from SciPy in column-major layout. The NumPy safetensors writer serialized their physical bytes without normalization. Restored the row-major layout used by the original selector. The C-contiguous native head patch is already correct.",
    selection_metrics_replayed_exactly=proof, fits_rerun=0, gpu_inferences_rerun=0)
write(out/"freeze.json",cfg)
evaluate(data,state/"features",out)
dest = state/"packed-results-corrected.zip"
tmp = dest.with_suffix(".part")
with ZipFile(tmp,"w",compression=ZIP_DEFLATED,compresslevel=3) as z:
    for path in sorted(out.iterdir()):
        z.write(path,path.name)
    z.write(state/"device.json","device.json")
    z.write(data/"protocol.json","protocol.json")
    for group in read(data/"protocol.json")["groups"].values():
        for key in group:
            z.write(state/"features"/(key+".json"),"feature-receipts/"+key+".json")
tmp.replace(dest)
print(dest,sha(dest),flush=True)
