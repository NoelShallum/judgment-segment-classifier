"""Export pre-decoder logits without refitting. Run inside the staged Colab VM after the benchmark."""

import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

state = Path("/content/attoflow-inlegalbert-packed")
data = state/"data"
output = state/"run-corrected"
dependencies = Path("/content/attoflow-inlegalbert-value/dependencies")
if not (dependencies/"transformers").exists():
    dependencies = state/"dependencies"
sys.path[:0] = [str(data),str(dependencies)]

import numpy as np
from safetensors.numpy import load_file
from fit import EXTRA, VIEWS, auxiliary, gather, join, loader, offsets, values
from native import read, save_arrays, sha, write

freeze = read(output/"freeze.json")
assert freeze["serialized_weights_roundtrip"]
assert freeze["plan_sha256"] == sha(data/"protocol.json")
assert freeze["weights_sha256"] == sha(output/"analytic-heads.safetensors")
assert (output/"shared48.json").is_file()
params = load_file(str(output/"analytic-heads.safetensors"))
dest = state/"emission-scores-corrected"
dest.mkdir(exist_ok=True)
decode = loader()
exports = {}
for group in ("fit","select","shared"):
    bank = gather(data,state/"features",group)
    y = join(bank,"gold")
    ends = offsets(bank)
    aux = auxiliary(bank,params["lexical.counts"],group == "fit")
    pack = {}
    preds = {}
    if group == "shared":
        path = output/"shared48-predictions.npz"
        assert sha(path) == read(output/"shared48.json")["prediction_sha256"]
        stored = np.load(path,allow_pickle=False)
        np.testing.assert_array_equal(stored["keys"],[row["id"] for row in bank])
        np.testing.assert_array_equal(stored["csv_rows"],join(bank,"rows"))
        np.testing.assert_array_equal(stored["gold"],y)
    for view in VIEWS:
        for name, rule in freeze["settings"][view]["selected"].items():
            tag = view+"."+name
            # Copy exact dtype and row ordering from the frozen runner.
            h = values(bank,view,rule["fraction_last"],aux)[:,EXTRA[name]]
            emission = h @ params[tag+".weight"] + params[tag+".bias"]
            p = decode(emission,params["crf.transitions"],ends,rule["transition"])
            if group == "shared":
                np.testing.assert_array_equal(p,stored[tag])
            pack["emission."+tag] = emission
            preds["pred."+tag] = p
    if group == "shared":
        np.testing.assert_array_equal(stored["source_cls"],join(bank,"source_scores").argmax(1))
        stored.close()
    file = dest/(group+".npz")
    save_arrays(file, keys=[j["id"] for j in bank], csv_rows=join(bank,"rows"), offsets=ends, gold=y,
                source_cls_logits=join(bank,"source_scores"), incumbent=join(bank,"incumbent"), **pack,**preds)
    exports[group] = dict(rows=len(y),cases=len(bank),sha256=sha(file))
    print(group,len(bank),len(y),flush=True)
meta = dest/"manifest.json"
write(meta,dict(groups=exports,freeze_sha256=sha(output/"freeze.json"),
    role_order=read(data/"protocol.json")["roles"],
    score_type="Float32 native source logits and per-arm pre-decoder ridge outputs.",
    fit_caveat="Fit scores are in-sample for the learned head. They may only be used descriptively. A later gate needs case-disjoint cross-fitting.",
    select_caveat="Selection scores shaped the frozen settings. They are not an independent test for this model or a downstream gate.",
    shared_caveat="No fit or selection on these cases. The returned shared labels match the original benchmark for all arms."))
file = state/"packed-scores-corrected.zip"
temp = file.with_suffix(".part")
with ZipFile(temp,"w",compression=ZIP_DEFLATED,compresslevel=2) as archive:
    for entry in sorted(dest.iterdir()):
        archive.write(entry,entry.name)
temp.replace(file)
print(file,sha(file),flush=True)
