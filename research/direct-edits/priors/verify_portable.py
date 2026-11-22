"""Exercise the portable entry point on real cases, a sentence and a paragraph."""
import importlib.metadata
import json
import sys
from pathlib import Path

from safetensors.numpy import load_file
import numpy as np
import torch

import experiment as work
from predict_document import Predictor


def main():
    torch.set_num_threads(1)
    here=work.HERE.parent/"portable-smoke"
    here.mkdir(exist_ok=True)
    files={"hier.safetensors":work.DONOR/"hier.safetensors","word2idx.json":work.DONOR/"word2idx.json",
           "priors.safetensors":work.HERE/"priors.safetensors","retention.safetensors":work.HERE.parent/"hierarchy/uniform-retention-plus-head.safetensors"}
    for k,v in files.items():
        if not (here/k).is_file():(here/k).symlink_to(v)
    spec={"strength":.125,"files":{k:work.file_hash(v) for k,v in files.items()}}
    (here/"model.json").write_text(json.dumps(spec))
    runtime=Predictor(here)
    f=np.load(work.HERE/"official-targeted-predictions.npz")
    ids=f["keys"][:4]
    expected={k:f["targeted"][a:b] for k,a,b in zip(f["keys"],f["offsets"],f["offsets"][1:]) if k in ids}
    cases=[]
    for k,txt,gold in work.cases("test",set(ids)):
        result,bias=runtime.predict(txt)
        assert result["labels"]==[work.LABELS[i] for i in expected[k]] and bias.shape==(10,)
        spans=result["spans"]
        assert spans[0]["start_unit"]==0 and spans[-1]["end_unit_exclusive"]==len(txt)
        assert all(a["end_unit_exclusive"]==b["start_unit"] for a,b in zip(spans,spans[1:]))
        cases.append({"id":k,"rows":len(txt),"label_disagreements":0})
    native_checks={}
    for method,source,label_field,offset_field in (
        ("source",work.HERE/"official-targeted-predictions.npz","donor","offsets"),
        ("retention",work.HERE.parent/"hierarchy/uniform-retention-test-score.npz","targeted","ends"),
        ("retention-prior",work.HERE/"composition-test-predictions.npz","gate and text bias","offsets")):
        if not source.is_file() and method=="retention-prior":
            source=work.HERE/"composition-val-predictions.npz"
        with np.load(source,allow_pickle=False) as sample:
            subset={str(k):sample[label_field][a:b] for k,a,b in zip(sample["keys"][:3],sample[offset_field],sample[offset_field][1:])}
        rt=Predictor(here,method)
        n=0
        for k,txt,_ in work.cases("val" if "composition-val" in source.name else "test",set(subset)):
            res,_=rt.predict(txt)
            assert res["labels"]==[work.LABELS[i] for i in subset[k]]
            n+=len(txt)
        assert n==sum(len(v) for v in subset.values())
        native_checks[method]={"ids":list(subset),"rows":n,"disagreements":0,"source":source.name}
    single=runtime.predict("The appeal is allowed.")[0]
    para=runtime.predict("Notice was served on 3 May. We therefore allow the appeal.")[0]
    assert list(single)==["label"] and single["label"] in work.LABELS and para["label"] in work.LABELS
    result={"cases":cases,"alternative_methods":native_checks,"single_string_api":single,"single_paragraph_api":para,"source":"Public LegalSeg replay and synthetic interface smoke examples.",
            "versions":{k:importlib.metadata.version(k) for k in ("numpy","safetensors","scikit-learn","torch")}}
    work.write(work.HERE/"portable-replay.json",result)
    print(result,flush=True)


if __name__=="__main__":
    main()
