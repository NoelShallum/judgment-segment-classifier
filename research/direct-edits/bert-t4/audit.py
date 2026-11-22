"""Check returned T4 predictions, paired exact spans and the preregistered advancement threshold."""
from pathlib import Path
import hashlib
import io
import json
import sys
from zipfile import ZipFile

import numpy as np
from safetensors.numpy import load

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from span_comparison import interval, measures, tally


def audit(stage):
    path=HERE/(stage+"-results.zip")
    out=HERE/(stage+"-audit.json")
    verified={}
    with ZipFile(path) as arc:
        assert arc.testzip() is None
        freeze_raw=arc.read("freeze.json")
        freeze=json.loads(freeze_raw)
        protocol=arc.read("protocol.json")
        assert hashlib.sha256(protocol).hexdigest()==freeze["protocol_sha256"]
        for f,h in freeze["sha256"].items():
            assert hashlib.sha256(arc.read(f)).hexdigest()==h
        config=json.loads(protocol)
        attested=json.loads(arc.read("device.json"))
        assert len(attested["devices"])==1 and "T4" in attested["devices"][0]
        arms=list(freeze["edit_tensor_stats"])
        for key in arms:
            tensors=load(arc.read(key+".safetensors"))
            for t,proof in zip(freeze["tensor_names"],freeze["edit_tensor_stats"][key],strict=True):
                assert hashlib.sha256(tensors[t].tobytes()).hexdigest()==proof["edited_tensor_sha256"]
        groups=("shared",) if stage=="priority" else ("shared","confirm")
        for group in groups:
            report=json.loads(arc.read(group+"-report.json"))
            raw=np.load(io.BytesIO(arc.read(group+"-predictions.npz")),allow_pickle=False)
            expected=config["groups"][group]
            if group=="shared" and stage=="priority":
                expected=expected[:16]
            assert expected==report["judgments"]==raw["case_ids"].tolist()
            assert report["fit_freeze_sha256"]==hashlib.sha256(freeze_raw).hexdigest()
            gold,off=raw["gold"],raw["offsets"]
            spans={}
            counts={}
            for name,score in report["score"].items():
                pred=raw[name]
                assert pred.shape==gold.shape
                cm=np.bincount(gold.astype(np.int64)*7+pred,minlength=49).reshape(7,7)
                assert cm.tolist()==score["confusion"]
                counts[name]=np.stack([cm_case(gold[a:b],pred[a:b]) for a,b in zip(off,off[1:])])
                ts=np.stack([tally(gold[a:b],pred[a:b]) for a,b in zip(off,off[1:])])
                spans[name]=(measures(ts),ts)
            base=counts["current_actual"]
            rng=np.random.default_rng(136310)
            choices=rng.integers(len(expected),size=(3000,len(expected)))
            f_base=f1(base[choices].sum(1))
            names={}
            for name,c in counts.items():
                vals=f1(c[choices].sum(1))-f_base
                pair=np.quantile(vals,[.025,.975]).tolist()
                delta=report["score"][name]["macro_f1"]-report["score"]["current_actual"]["macro_f1"]
                names[name]={"macro_f1":report["score"][name]["macro_f1"],"accuracy":report["score"][name]["accuracy"],
                    "macro_f1_delta_vs_current":delta,"macro_f1_delta_ci95":pair,
                    "span_scores":spans[name][0],"span_f1_delta_ci95":interval(spans[name][1],spans["current_actual"][1]),
                    "meets_1_point_validation_gate":bool(group=="confirm" and delta>=.01 and pair[0]>0)}
            verified[group]={"n_cases":len(expected),"n_rows":len(gold),"native_replay":report["native_replay"],
                             "targeted_confusion_ci95":report["paired_targeted_vs"],"methods":names}
    result={"archive_sha256":digest(path),"stage":stage,"groups":verified,
            "source_tensor_changes":freeze["edit_tensor_stats"],"validity":"Fully verified T4 archive, matched native replay, all per-case confusion matrices recomputed.",
            "published_comparison":"These are official-validation judgments. The full-test result stays in the separate benchmark receipt."}
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(stage,{group:{name:round(s["macro_f1"],5) for name,s in v["methods"].items()} for group,v in verified.items()},flush=True)


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def cm_case(g,p):
    return np.bincount(g.astype(np.int64)*7+p,minlength=49).reshape(7,7)


def f1(conf):
    tp=np.diagonal(conf,axis1=-2,axis2=-1)
    return (2*tp/(conf.sum(-1)+conf.sum(-2)).clip(min=1)).mean(-1)


if __name__=="__main__":
    audit(sys.argv[1])
