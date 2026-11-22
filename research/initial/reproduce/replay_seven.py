"""Verify the exact submitted texts and every recorded label comparison from an authorized LegalSeg test CSV."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from seven_paths import LABELS,FOLDERS
from seven_paragraphs import ROLES
ORDINAL=re.compile(r"^\s*(?:\(?\d{1,3}[.)]\s*|[ivxlcdm]+[.)]\s+)",re.I)
MONEY=re.compile(r"\bRs\.?\s*[\d,]+(?:\.\d+)?/?",re.I)
DATE=re.compile(r"\b\d{1,2}[/.]\d{1,2}[/.]\d{2,4}\b|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}\b")
def digest(text):return hashlib.sha256(text.encode()).hexdigest()
def normal(text):return " ".join(text.split())
def main():
    root=Path(__file__).resolve().parent;ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-csv",type=Path,default=root/"references/official/test.csv");ap.add_argument("--evidence",type=Path,default=root.parent/"evidence")
    ap.add_argument("--materialize",action="store_true",help="Write reconstructed text cases next to this script for a new local forward pass.");args=ap.parse_args()
    e=args.evidence;spec=json.loads((e/"official-csv-checksums.json").read_text())["files"]["test.csv"]
    with args.test_csv.open("rb") as f:h=hashlib.file_digest(f,"sha256").hexdigest()
    if h!=spec["sha256"]:raise ValueError("Test CSV has an unexpected source revision")
    with args.test_csv.open(encoding="utf-8-sig",newline="") as f:source=list(csv.DictReader(f))
    for group in FOLDERS:
        mf=e/(group+"-input-manifest.json");rf=e/("gemini-seven-"+group+".json")
        if not mf.is_file() or not rf.is_file():continue
        items=json.loads(mf.read_text());report=json.loads(rf.read_text());idmap={r["id"]:r for r in report["rows"]}
        approved=[];records=[]
        for r in items:
            if group=="homogeneous":vv=[source[i] for i in r["source_rows"]];text=" ".join(normal(a["Text"]) for a in vv)
            else:
                index=int(r["id"].split("-")[-1]);vv=[source[index]];raw=vv[0]["Text"]
                if group=="paragraphs":
                    raw=normal(raw)
                    if digest(raw)!=r["original_sha256"]:raise ValueError("Original multi-statement source hash differs")
                    text=DATE.sub("[date]",MONEY.sub("[amount]",raw))
                else:text=normal(ORDINAL.sub("",raw.strip()))
            if digest(text)!=r["sha256"] or not all(v["Index"]==r["doc"] for v in vv):raise ValueError("Hash or case mismatch "+r["id"])
            p=idmap[r["id"]]
            if p["text_sha256"]!=digest(text) or p["gold"]!=r["gold"]:raise ValueError("Inconsistent score manifest")
            approved.append({**r,"text":text});records.append(p)
        results=next(iter(report["results"].values()));labels=list(LABELS)
        for name,val in results["metrics"].items():
            cm=np.zeros((7,7),int)
            for r in records:cm[labels.index(r["gold"]),labels.index(r[name])]+=1
            correct=int(cm.trace());den=cm.sum(1)+cm.sum(0)
            macro=np.divide(2*cm.diagonal(),den,out=np.zeros(7),where=den>0).mean()
            if correct!=val["correct"] or not np.isclose(macro,val["macro_f1"]):raise ValueError("Score differs for "+name)
            if name!="gemini":
                n=sum(r[name]==r["gemini"] for r in records)
                if n!=results["paired_with_gemini"][name]["seven_label_agreement_count"]:raise ValueError("Provider agreement differs")
        if args.materialize:
            loc=root/FOLDERS[group];loc.mkdir(exist_ok=True);(loc/"approved.json").write_text(json.dumps(approved))
        print(group,len(records),"matched input hashes and",len(results["metrics"]),"sets of predictions verified")
    span=e/"gemini-seven-spans.json"
    if span.exists():
        v=json.loads(span.read_text())
        clouds=[]
        for r in v["records"]:
            raw=" ".join(normal(source[i]["Text"]) for i in r["source_rows"])
            if digest(raw)!=r["raw_sha256"]:raise ValueError("Different source span passage")
            clouds.append({"doc":r["case"],"rows":[{"row":i,"text":normal(source[i]["Text"]),"gold":ROLES[source[i]["Label"]]} for i in r["source_rows"]]})
        if args.materialize:
            loc=root/"seven-spans";loc.mkdir(exist_ok=True);(loc/"cloud.json").write_text(json.dumps(clouds))
        print(v["cases"],"shared whole-passage input hashes verified")
if __name__=="__main__":main()
