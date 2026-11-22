from pathlib import Path
import csv
import hashlib
import json
import os
import string
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
NAMES = ("None","Facts","Issue","Arguments of Petitioner","Arguments of Respondent","Reasoning","Decision")
TO_ID = {s:i for i,s in enumerate(NAMES)}
PUNC = str.maketrans(string.punctuation," "*len(string.punctuation))

def dump(path,obj):
    tmp = path.with_suffix(path.suffix+".part")
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")
    os.replace(tmp,path)

def file_hash(path):
    with path.open("rb") as fp:
        return hashlib.file_digest(fp,"sha256").hexdigest()

def counts():
    src = ROOT/"references/official/train.csv"
    vocab = json.loads((ROOT/"references/hier/word2idx.json").read_text())
    vocab_count = max(vocab.values())+1
    stats = np.zeros((3,7,vocab_count),np.int64)
    buffers = [[[] for _ in range(7)] for _ in range(3)]
    n = np.zeros(7,np.int64)
    def push():
        for kind in range(3):
            for role in range(7):
                b = buffers[kind][role]
                if b:
                    stats[kind,role] += np.bincount(b,minlength=vocab_count)
                    b.clear()
    started = time.monotonic()
    docs = set()
    with src.open(newline="",encoding="utf-8-sig") as fp:
        for item in csv.DictReader(fp):
            text = item["Text"].strip()
            if not text:
                continue
            role = TO_ID[item["Label"]]
            n[role] += 1
            docs.add(item["Index"])
            ids = [vocab.get(w,1) for w in text.lower().translate(PUNC).split()] or [1]
            buffers[0][role].extend(set(ids))
            buffers[1][role].extend(set(ids[:5]))
            buffers[2][role].extend(set(ids[-5:]))
            if n.sum() % 12288 == 0:
                push()
    push()
    temp = HERE/"counts.part"
    with temp.open("wb") as fp:
        np.savez_compressed(fp,counts=stats,labels=n)
    temp.replace(HERE/"counts.npz")
    dump(HERE/"counts.json",dict(train_sha256=file_hash(src),docs=len(docs),nonblank_sentences=int(n.sum()),class_support=n.tolist(),vocab=vocab_count,seconds=time.monotonic()-started))
    print((HERE/"counts.json").read_text(),flush=True)

if __name__=="__main__":
    counts()
