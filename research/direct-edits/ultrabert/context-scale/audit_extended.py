"""Compare exactly the same original UltraBERT targets under two token budgets."""
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

HERE=Path(__file__).resolve().parent


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def score(zip_path,prefix):
    with ZipFile(zip_path) as z:
        assert z.testzip() is None
        inspect=json.loads(z.read(prefix+"shared-report.json"))
        saved=np.load(io.BytesIO(z.read(prefix+"shared-predictions.npz")),allow_pickle=False)
        return inspect,{k:saved[k] for k in saved.files}


def cf(y,p,indices):
    return np.stack([np.bincount(7*y[ix].astype(int)+p[ix],minlength=49).reshape(7,7) for ix in indices])


def f1(c):
    tp=np.diagonal(c,axis1=-2,axis2=-1)
    return (2*tp/(c.sum(-1)+c.sum(-2)).clip(min=1)).mean(-1)


def main():
    previous=HERE/"attoflow-ultrabert-t4-results.zip"
    current=HERE/"extended-results.zip"
    first,x=score(previous,"model/")
    second,z=score(current,"")
    assert first["freeze_sha256"]==second["freeze_sha256"]
    assert first["cases"]==second["cases"]==16 and first["rows"]==second["rows"]==3558
    for k in ("case","row","ix","start","end","tok","gold","incumbent"):
        np.testing.assert_array_equal(x[k],z[k])
    inds=[np.flatnonzero(z["case"]==i) for i in dict.fromkeys(z["case"])]
    choices=np.random.default_rng(38115).integers(len(inds),size=(3000,len(inds)))
    methods={}
    for m in first["methods"]:
        a,b=cf(x["gold"],x[m],inds),cf(z["gold"],z[m],inds)
        assert np.isclose(f1(a.sum(0)),first["methods"][m]["macro_f1_7"])
        assert np.isclose(f1(b.sum(0)),second["methods"][m]["macro_f1_7"])
        sample=f1(b[choices].sum(1))-f1(a[choices].sum(1))
        methods[m]={"384_macro_f1":float(f1(a.sum(0))),"2048_macro_f1":float(f1(b.sum(0))),
                    "long_context_minus_short_paired95":np.quantile(sample,[.025,.975]).tolist(),
                    "labels_changed":int(np.count_nonzero(x[m]!=z[m]))}
    result={"native_backbone":"dmis-lab/UltraBERT-150M","unchanged_freeze_sha256":first["freeze_sha256"],"cases":16,"rows":3558,
            "identical_text_tokens":int(x["tok"].sum()),"same_human_labels_and_offsets":True,
            "source_384_archive":digest(previous),"source_2048_archive":digest(current),"methods":methods,
            "primary":"Ordered larger inputs. The same classifier and rank-two weight patch were used under both contexts."}
    (HERE/"extended-comparison.json").write_text(json.dumps(result,indent=2)+"\n")
    print(methods,flush=True)


if __name__=="__main__":
    main()
