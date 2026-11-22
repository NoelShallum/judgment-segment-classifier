"""Verify PPLX outputs and compare exact spans on the published sentence grid."""
from pathlib import Path
import csv
import hashlib
import io
import json
import sys
from zipfile import ZipFile

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from span_comparison import interval, measures, tally


def macro(c):
    hit=np.diagonal(c,axis1=-2,axis2=-1)
    return (2*hit/(c.sum(-1)+c.sum(-2)).clip(min=1)).mean(-1)


def main():
    with ZipFile(HERE/"results.zip") as arc:
        assert arc.testzip() is None
        report=json.loads(arc.read("report.json"))
        protocol=json.loads(arc.read("protocol.json"))
        assert report["protocol_sha256"]==hashlib.sha256(arc.read("protocol.json")).hexdigest()
        x=np.load(io.BytesIO(arc.read("shared-predictions.npz")),allow_pickle=False)
        groups=list(dict.fromkeys(x["cases"]))
        assert groups==protocol["shared"]
        assert report["rows"]==len(x["gold"])
        roles={z:i for i,z in enumerate(protocol["role_order"])}
        sizes=np.cumsum([0]+[protocol["rows"][i] for i in groups])
        val=HERE.parents[1]/"references/official/val.csv"
        with val.open("rb") as opened:
            assert protocol["source_csv"]["val"]==hashlib.file_digest(opened,"sha256").hexdigest()
        checked=set(x["row"])
        gold={}
        with val.open(encoding="utf-8-sig",newline="") as stream:
            for i,case in enumerate(csv.DictReader(stream)):
                if i in checked:
                    gold[i]=(roles[case["Label"]],case["Index"])
        assert all(gold[i]==(y,c) for i,y,c in zip(x["row"],x["gold"],x["cases"],strict=True))
        preds={k:x[k] for k in x.files if k not in {"row","gold","cases"}}
        cm={}
        spans={}
        y=x["gold"]
        for k,p in preds.items():
            spans[k]=np.stack([tally(y[a:b],p[a:b]) for a,b in zip(sizes,sizes[1:])])
            cm[k]=np.stack([np.bincount(y[a:b]*7+p[a:b],minlength=49).reshape(7,7) for a,b in zip(sizes,sizes[1:])])
        for mode,r in report["models"].items():
            for k,s in r["scores"].items():
                assert np.isclose(macro(cm[mode+" "+k].sum(0)),s["macro_f1"])
            key=mode+"-native-projection.safetensors"
            assert hashlib.sha256(arc.read(key)).hexdigest()==r["native_projection"]["replacement_sha256"]
        draws=np.random.default_rng(1521).integers(len(groups),size=(3000,len(groups)))
        before="isolated native projection write"
        after="contextual native projection write"
        diffs=macro(cm[after][draws].sum(axis=1))-macro(cm[before][draws].sum(axis=1))
        stats={k:{**measures(a),"paired_span_f1_delta_vs_incumbent":interval(a,spans["incumbent"])} for k,a in spans.items()}
        report["paired_mf1_contextual_minus_isolated_projection"]={"point":float(macro(cm[after].sum(0))-macro(cm[before].sum(0))),
                                                                   "ci95":np.quantile(diffs,[.025,.975]).tolist()}
        report["segmentation_on_supplied_boundaries"]=stats
    receipt=HERE/"learned-native-parity.json"
    if receipt.is_file():
        native=json.loads(receipt.read_text())
        for mode,p in native["native_replay"].items():
            assert p["checkpoint_sha256"]==report["models"][mode]["native_projection"]["replacement_sha256"]
        report["learned_native_parity"]=native
    (HERE/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print({mode:{k:round(v["macro_f1"],5) for k,v in z["scores"].items()} for mode,z in report["models"].items()})
    print(report["paired_mf1_contextual_minus_isolated_projection"],flush=True)


if __name__=="__main__":
    main()
