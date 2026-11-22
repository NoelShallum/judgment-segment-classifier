"""Download public author-released checkpoints, benchmark outputs and auxiliary corpora."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parent
PUBLIC = [
    ("checkpoint", "models", "L-NLProc/LegalSeg_InLegalBERT", "44d7925e5ab422ecf04acd5dcbb0b62a6f8d8f53", ["InLegalBERT/InLegalBERT(i)model.safetensors"]),
    ("tokenizer", "models", "law-ai/InLegalBERT", "b5ecfed8ed6cf9d25a3cb8225a8c52f161f7401a", ["config.json", "vocab.txt", "special_tokens_map.json", "tokenizer_config.json"]),
    ("baseline_sentence", "datasets", "L-NLProc/LegalSeg_InLegalBERT_Predictions", "e9233636e75dfa221dd18d14c4ecec66942fda13", ["InLegalBert(i)_predictions.csv","InLegalBert(i-2, i-1, i)_predictions.csv"]),
    ("baseline_document", "datasets", "L-NLProc/LegalSeg_Hier_BiLSTM_CRF_Predictions", "929608e0cd47d37151017952d7858ce5bfd648b0", ["Hier_BiLSTM_predictions.csv","predictions.txt"]),
    ("marro", "datasets", "subinay612/MARRO_Rhetorical-Role-Labeling", "668bc2d187d934f8e3cb71c15b887459ded347f4", ["README.md","IN-train-set.zip","IN-test-set.zip","UK-train-set.zip","UK-test-set.zip"]),
    ("minilm", "models", "sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41", ["tokenizer.json","tokenizer_config.json"]),
]
OFFICIAL = ("official","datasets","L-NLProc/LegalSeg_CSV","31217aa845866bbd18a8ad567a785e6f5485435a",["train.csv","val.csv","test.csv"])
EXPERIMENTAL_PARENTS = [
    ("minilm","models","sentence-transformers/all-MiniLM-L6-v2","1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
     ["config.json","special_tokens_map.json","vocab.txt","tokenizer.json","tokenizer_config.json","model.safetensors"]),
    ("context-parent","models","L-NLProc/LegalSeg_InLegalBERT","44d7925e5ab422ecf04acd5dcbb0b62a6f8d8f53",
     ["InLegalBERT/InLegalBERT(i-2, i-1, i)model.safetensors"]),
]

def fetch(owner, namespace, repo, revision, filename, metadata, headers):
    dest = ROOT/"references"/owner/Path(filename).name
    dest.parent.mkdir(parents=True,exist_ok=True)
    url=f"https://huggingface.co/{'datasets/' if namespace=='datasets' else ''}{repo}/resolve/{revision}/{quote(filename)}"
    info=metadata[filename]
    digest=info.get("lfs",{}).get("sha256")
    if not digest and info.get("lfs",{}).get("oid"): digest=info["lfs"]["oid"]
    if dest.is_file() and dest.stat().st_size==info["size"]:
        actual=hashlib.file_digest(dest.open("rb"),"sha256").hexdigest()
        if not digest or actual==digest:
            print("Cached",owner,filename,flush=True)
            return dict(owner=owner,file=filename,path=str(dest.relative_to(ROOT)),url=url,bytes=info["size"],sha256=actual,revision=revision)
    tmp=dest.with_suffix(dest.suffix+".part")
    for attempt in range(3):
        try:
            with requests.get(url,headers=headers,stream=True,timeout=(15,90)) as response:
                if response.status_code in (401,403): raise PermissionError(f"{response.status_code} access to {repo} requires approved credentials")
                if response.status_code!=200:raise ValueError(f"{response.status_code} fetching {repo}/{filename}")
                h=hashlib.sha256()
                with tmp.open("wb") as out:
                    for block in response.iter_content(1024*1024):
                        h.update(block);out.write(block)
            if tmp.stat().st_size!=info["size"] or digest and h.hexdigest()!=digest:raise ValueError("Download integrity mismatch: "+filename)
            tmp.replace(dest)
            print("Downloaded",owner,filename,info["size"],flush=True)
            return dict(owner=owner,file=filename,path=str(dest.relative_to(ROOT)),url=url,bytes=info["size"],sha256=h.hexdigest(),revision=revision)
        except PermissionError:
            raise
        except (requests.RequestException, ValueError):
            if attempt==2:raise
            time.sleep(1+attempt)

def main():
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group()
    group.add_argument("--official",action="store_true")
    group.add_argument("--experimental-parents",action="store_true")
    args=parser.parse_args()
    headers={}
    if args.official:
        from huggingface_hub import get_token
        key=get_token()
        if key: headers["Authorization"]="Bearer "+key
    jobs=[]
    specs=[OFFICIAL] if args.official else EXPERIMENTAL_PARENTS if args.experimental_parents else PUBLIC
    for owner,ns,repo,rev,files in specs:
        response=requests.get(f"https://huggingface.co/api/{ns}/{repo}/revision/{rev}",params={"blobs":"true"},headers=headers,timeout=20)
        response.raise_for_status();filespecs={x["rfilename"]:x for x in response.json()["siblings"]}
        for f in files:jobs.append((owner,ns,repo,rev,f,filespecs,headers))
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        results=list(pool.map(lambda job: fetch(*job),jobs))
    manifest="official-manifest.json" if args.official else "experimental-manifest.json" if args.experimental_parents else "public-manifest.json"
    (ROOT/manifest).write_text(json.dumps(results,indent=2)+"\n")
    if args.experimental_parents:
        for name in ("minilm","context-parent"):
            items=[r for r in results if r["owner"]==name]
            value=items[0] if name=="context-parent" else items
            (ROOT/"references"/name/"verified.json").write_text(json.dumps(value,indent=2)+"\n")
if __name__=="__main__":main()
