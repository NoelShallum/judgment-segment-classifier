"""Evaluate frozen seven-way heads on an external, genuinely paragraph-annotated corpus."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from zipfile import ZipFile

import numpy as np
import requests

from encoder import features, fingerprint, components, TOK
from program_trial import save_json
from weight_edit import LABELS, ORIGINAL, ROOT, file_hash, tensor


GOLD = ("Fact", "Argument", "Analysis", "Verdict")
PROJECT = ("outside", "Fact", "outside", "Argument", "Argument", "Analysis", "Verdict")
DEST = ROOT / "references/paragraph"
API = "https://data.mendeley.com/public-api/datasets/97khn9nswp/files?folder_id=root&version=1"
DIGEST = {"PRR.zip": "afc69352aaaa9b7512a12456821f98f685208ec9052dd2011d80c3020abedfd1",
          "SRR.zip": "6e0a7a01c36bd9348b4dbc0e2e4fa9a0e277f396331706c209008eb2f259ecf2"}
INDEX = np.array([GOLD.index(n) if n in GOLD else 4 for n in PROJECT])


def acquire():
    DEST.mkdir(parents=True,exist_ok=True)
    if all((DEST/n).is_file() and file_hash(DEST/n)==h for n,h in DIGEST.items()):
        return
    response=requests.get(API,timeout=20);response.raise_for_status()
    for file in response.json():
        name=file["filename"]
        if name not in DIGEST:
            continue
        r=requests.get(file["content_details"]["download_url"],timeout=(15,90));r.raise_for_status()
        if hashlib.sha256(r.content).hexdigest()!=DIGEST[name]:
            raise ValueError("Paragraph source checksum mismatch")
        tmp=(DEST/name).with_suffix(".partial");tmp.write_bytes(r.content);tmp.replace(DEST/name)


def read(name):
    rows=[]
    with ZipFile(DEST/name) as archive:
        for file in sorted(p for p in archive.namelist() if p.endswith(".txt")):
            for index,line in enumerate(archive.read(file).decode("utf-8-sig").splitlines()):
                text,role=line.rsplit("\t",1)
                rows.append((Path(file).stem,index,text.strip(),role.strip()))
    return rows


def normalized(text):
    text=unicodedata.normalize("NFKC",text).casefold()
    words=re.findall(r"[^\W_]+",text)
    if len(words)<12:
        return None
    return hashlib.blake2b(" ".join(words).encode(),digest_size=16).digest()


def overlapping_cases(para, sentences):
    from import_official import FILES
    source=ROOT/"references/official/train.csv"
    if file_hash(source)!=FILES["train.csv"][1]:
        raise ValueError("Wrong training data revision")
    segments=defaultdict(set)
    for case,_,text,_ in para+sentences:
        h=normalized(text)
        if h:
            segments[h].add(case)
    known=set()
    with source.open(encoding="utf-8-sig",newline="") as f:
        for r in csv.DictReader(f):
            h=normalized(r["Text"])
            if h in segments:
                known.add(h)
    hits=Counter(doc for h in known for doc in segments[h])
    return hits


def by_case(rows, pred):
    cases=sorted({r[0] for r in rows})
    index={d:i for i,d in enumerate(cases)}
    docs=np.array([index[r[0]] for r in rows])
    gold=np.array([GOLD.index(r[3]) for r in rows])
    return np.bincount(docs*20+gold*5+pred,minlength=len(cases)*20).reshape(len(cases),4,5)


def metrics(cm):
    true_positive=cm[:,:4].diagonal(); support=cm.sum(1)
    denom=support+cm[:,:4].sum(0)
    f1=np.divide(2*true_positive,denom,out=np.zeros(4),where=denom>0)
    return {"four_role_macro_f1":float(f1.mean()),"accuracy":float(true_positive.sum()/cm.sum()),
            "per_role_f1":dict(zip(GOLD,f1.tolist())),"support":dict(zip(GOLD,support.astype(int).tolist())),
            "outside_predictions":int(cm[:,4].sum()),"confusion_rows":GOLD,"confusion_columns":(*GOLD,"outside"),
            "confusion":cm.astype(int).tolist()}


def compare(rows,predictions):
    matrices={name:by_case(rows,values) for name,values in predictions.items()}
    report={name:metrics(cm.sum(0)) for name,cm in matrices.items()}
    origin,edited=matrices["original"],matrices["direct"]
    rng=np.random.default_rng(320411);diff=[]
    for _ in range(1000):
        idx=rng.integers(len(origin),size=len(origin))
        diff.append(metrics(edited[idx].sum(0))["four_role_macro_f1"]-metrics(origin[idx].sum(0))["four_role_macro_f1"])
    return {"rows":len(rows),"judgments":len(origin),"models":report,
            "direct_minus_original_paired_case_95pct":np.quantile(diff,[.025,.975]).tolist()}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--backend",choices=["torch","intel"],default="torch")
    args=ap.parse_args()
    acquire()
    paragraphs=read("PRR.zip");sentences=read("SRR.zip")
    assert len(paragraphs)==1076 and len({r[0] for r in paragraphs})==60
    assert {r[3] for r in paragraphs}==set(GOLD)
    overlap=overlapping_cases(paragraphs,sentences)
    texts=[r[2] for r in paragraphs]
    if args.backend=="intel":
        from openvino_worker import embed,get_engine
        if not json.loads((ROOT/"results/openvino-gpu-fp32-parity.json").read_text())["comparable"]:
            raise ValueError("Verify GPU FP32 parity first")
        engine,tok=get_engine("GPU")
        feature_file=ROOT/"cache"/("external-prr-fp32-"+fingerprint(texts)[:24]+".npy")
        if feature_file.is_file():
            x=np.load(feature_file,allow_pickle=False)
        else:
            x=embed(engine,tok,texts)
            tmp=feature_file.with_suffix(".partial")
            with tmp.open("wb") as f:np.save(f,x,allow_pickle=False)
            tmp.replace(feature_file)
    else:
        model,tok,device=components()
        x=features(model,tok,device,texts,"external-prr-torch-fp32")
    lengths=[len(seq) for seq in tok(texts,add_special_tokens=False,return_attention_mask=False,
              return_token_type_ids=False,truncation=False,verbose=False)["input_ids"]]
    paths={"original":ORIGINAL,"direct":ROOT/"checkpoints/selected-direct/model.safetensors",
           "gold_discriminant":ROOT/"checkpoints/discriminant-256/model.safetensors",
           "gemini_reviewed":ROOT/"checkpoints/augmented-discriminant/model.safetensors"}
    predictions={}
    for name,p in paths.items():
        if not p.is_file():
            continue
        logits=x@tensor(p,"classifier.weight").T+tensor(p,"classifier.bias")
        predictions[name]=INDEX[logits.argmax(1)]
    result={"dataset":"Nikita D, Dipti Rana and Rupa Mehta, Annotated Judgments from the Supreme Court of India, version 1",
            "doi":"10.17632/97khn9nswp.1","dataset_url":"https://data.mendeley.com/datasets/97khn9nswp/1",
            "license":"CC BY 4.0","source_sha256":DIGEST,"prediction_projection":dict(zip(LABELS,PROJECT)),
            "protocol":"One prediction per original paragraph. Map the predicted LegalSeg role to the four reference roles; retain none and issue as outside and count them wrong. No provider calls, training or score-based mapping on this corpus.",
            "backend":args.backend,"median_content_tokens":float(np.median(lengths)),
            "paragraphs_truncated_to_last_510":sum(n>510 for n in lengths),
            "overlap_check":"Case flagged if any paragraph or source companion sentence of at least 12 normalized words matches a complete LegalSeg training sentence. This conservative check cannot prove case identities differ.",
            "cases_with_matching_spans":dict(overlap),"all":compare(paragraphs,predictions)}
    keep=np.array([r[0] not in overlap for r in paragraphs])
    if keep.any():
        result["no_detected_training_match"]=compare([r for r,k in zip(paragraphs,keep) if k],
                                                 {n:pr[keep] for n,pr in predictions.items()})
    save_json(ROOT/"results/prr-paragraph-transfer.json",result)
    print("Paragraphs",len(paragraphs),"cases with exact training overlaps",len(overlap),
          {n:round(v["four_role_macro_f1"],4) for n,v in result["all"]["models"].items()},flush=True)


if __name__=="__main__":
    main()
