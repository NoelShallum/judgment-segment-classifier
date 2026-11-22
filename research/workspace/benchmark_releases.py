"""Audit authors' prediction files; never selects or scores an AttoFlow model."""
from collections import defaultdict
import csv
import json
import string
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
LABELS=("None","Facts","Issue","Arguments of Petitioner","Arguments of Respondent","Reasoning","Decision")
def score(truth,pred):
    t=np.asarray(truth,dtype=int);p=np.asarray(pred,dtype=int)
    cm=np.bincount(7*t+p,minlength=49).reshape(7,7);den=cm.sum(0)+cm.sum(1)
    f1=np.divide(2*cm.diagonal(),den,out=np.zeros(7),where=den!=0)
    return {"examples":len(t),"macro_f1":float(f1.mean()),"accuracy":float(np.mean(t==p)),"per_role_f1":dict(zip(LABELS,f1.tolist())),"confusion":cm.tolist()}
def main():
    refs=ROOT/"references"
    by_doc=defaultdict(list);cleaned=defaultdict(list);truth=[]
    strip_punc=str.maketrans(string.punctuation," "*len(string.punctuation))
    with (refs/"official/test.csv").open(encoding="utf-8-sig",newline="") as f:
        for r in csv.DictReader(f):
            label=LABELS.index(r["Label"]);truth.append(label);by_doc[r["Index"]].append(label)
            # The upstream prepare_data.py splits on a single tab and drops
            # lines with an empty word sequence after ASCII punctuation removal.
            s=r["Text"]
            if "\t" not in s and s.strip().lower().translate(strip_punc).split():cleaned[r["Index"]].append(label)
    report={"official_test_rows":len(truth),"official_test_documents":len(by_doc),"audited_predictions":{}}
    for folder in ["baseline_sentence","baseline_document"]:
        for f in sorted((refs/folder).glob("*.csv")):
            t=[];p=[]
            with f.open(newline="") as stream:
                for r in csv.DictReader(stream):
                    t.append(int(r.get("true",r.get("True"))));p.append(int(r.get("pred",r.get("Predicted"))))
            result=score(t,p)
            result["released_true_labels_equal_official_in_row_order"]=(truth==t)
            report["audited_predictions"][f.name]=result
    file=refs/"baseline_document/predictions.txt"
    named={};unexpected=set()
    for row in file.read_text().splitlines():
        name,sequence=row.split("\t",1);doc=str(int(name.removeprefix("file_"))-1)
        if doc in named:raise ValueError("duplicate case in named predictions")
        raw=sequence.rstrip(",").split(",");unexpected.update(set(raw)-set(LABELS));named[doc]=raw
    t=[];p=[];miscount=0;missing=0;covered=0
    if not unexpected:
        for doc,labels in cleaned.items():
            if doc not in named:missing+=1;continue
            if len(named[doc])!=len(labels):miscount+=1;continue
            t.extend(labels);p.extend(LABELS.index(x) for x in named[doc]);covered+=1
    report["named_hierarchical"]={"alignment":"Author case files are numbered 6409..7120; CSV Index is 6408..7119. Subtract one from filename ID. Apply author's plain-text punctuation/empty-token filter, then accept exact per-case lengths only.","filtered_official_population":sum(map(len,cleaned.values())),"predicted_docs":len(named),"unrecognized_labels":sorted(unexpected),"matched_docs_of_equal_length":covered,"missing_official_docs":missing,"same_id_different_length":miscount,"score_on_exactly_aligned_docs":score(t,p) if t else None}
    out=ROOT/"results";out.mkdir(exist_ok=True)
    (out/"published-baselines.json").write_text(json.dumps(report,indent=2)+"\n")
    print([(k,v["examples"],v["macro_f1"],v["released_true_labels_equal_official_in_row_order"]) for k,v in report["audited_predictions"].items()])
    named=report["named_hierarchical"];print({k:v for k,v in named.items() if k!="score_on_exactly_aligned_docs"},named["score_on_exactly_aligned_docs"]["macro_f1"] if t else None)
if __name__=="__main__":main()
