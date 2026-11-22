"""Freeze a matched case protocol for the native PPLX contextual projection."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CORPUS = ROOT/"references"/"official"
LABELS = ("None","Facts","Issue","Arguments of Petitioner","Arguments of Respondent","Reasoning","Decision")
ROLE_IDS = {name: i for i,name in enumerate(LABELS)}


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def gather(split, use):
    found = defaultdict(list)
    with (CORPUS/(split+".csv")).open(newline="",encoding="utf-8-sig") as f:
        for i, r in enumerate(csv.DictReader(f)):
            if r["Index"] in use and r["Text"].strip():
                found[r["Index"]].append((i,r["Text"].strip(),ROLE_IDS[r["Label"]]))
    assert set(found)==set(use)
    return found


def make_package():
    seed = json.loads((HERE.parent/"bert-t4/protocol.json").read_text())
    use_fit = seed["groups"]["fit"][:32]
    use_select = seed["groups"]["fit"][101:113]
    shared = seed["groups"]["shared"][:16]
    assert len(set(use_fit+use_select+shared)) == 60
    ds = gather("train",use_fit+use_select) | gather("val",shared)
    assert {row[2] for case in use_fit for row in ds[case]} == set(range(7))
    sources = HERE.parent / "pooler/shared-cases"
    refs = {k:json.loads((sources / (k+".json")).read_text())["predictions"]["unchanged"] for k in shared}
    assert all(len(ds[k])==len(p) for k,p in refs.items())
    manifest = {"model":json.loads((HERE/"source.json").read_text()),"fit":use_fit,"select":use_select,"shared":shared,
                "source_csv":{s:sha(CORPUS/(s+".csv")) for s in ("train","val")},
                "role_order":LABELS,"rows":{k:len(v) for k,v in ds.items()}, "context_token_budget":1024,
                "candidate":"Write the first seven rows of contextual_projection.weight [1024,1024] using closed-form ridge, producing one role per original row from its unquantized chunk representation.",
                "comparison":"Fit the same raw and independently optimized frozen-projection readouts in contextual and isolated views. Compare an equal-norm shuffled target write and AttoFlow on exactly the shared16 rows.",
                "fold":"All case identities fixed before PPLX outcomes. Fit uses only official training cases; select uses other official training cases. Both may contain texts seen by the LegalSeg public donor. Shared official validation was inspected previously."}
    with zipfile.ZipFile(HERE/"worker.zip","w",zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        z.writestr("protocol.json", json.dumps(manifest,indent=2))
        z.writestr("cases.json",json.dumps(ds,ensure_ascii=False))
        z.writestr("incumbent.json",json.dumps(refs))
        for f in ("native_pplx.py","score_projection.py"):
            z.write(HERE/f, f)
    with zipfile.ZipFile(HERE/"worker.zip") as z:
        assert z.testzip() is None
    print(json.dumps({"rows":{k:sum(len(ds[case]) for case in v) for k,v in (
        ("fit",use_fit),("select",use_select),("shared",shared))}, "worker_sha256":sha(HERE/"worker.zip")}),flush=True)


if __name__=="__main__":
    make_package()
