"""Copy only complete user-supplied LegalSeg CSVs matching the pinned Hub objects."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import time
import argparse

ROOT=Path(__file__).resolve().parent
FILES={
 "train.csv":(217019314,"60028fc085e3cb018e8d49a8879ba796dd03a0838367891f3312f9e576921794"),
 "val.csv":(51327760,"8abb2d789325418fc5ab4186c0dc63798e02b8fd82f9e00c324185e6bd86f816"),
 "test.csv":(27791389,"5e1a2f9a03fd7581d4585ef97f55787d058afbed063b5e03b6e3611cbe45f9ee")}
def sha(p):
    with p.open("rb") as h:return hashlib.file_digest(h,"sha256").hexdigest()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--downloads",type=Path,default=Path.home()/"Downloads")
    parser.add_argument("--wait",type=int,default=0)
    a=parser.parse_args()
    out=ROOT/"references/official";out.mkdir(exist_ok=True,parents=True)
    done={};start=time.monotonic()
    while True:
        for name,(size,expected) in FILES.items():
            if name in done:continue
            final=out/name
            paths=[final,a.downloads/name,*sorted(a.downloads.glob(Path(name).stem+" (*).csv"))]
            for p in paths:
                if not p.is_file() or p.stat().st_size!=size or sha(p)!=expected:continue
                if p!=final:
                    tmp=out/(name+".part");shutil.copyfile(p,tmp)
                    assert sha(tmp)==expected;os.replace(tmp,final)
                done[name]={"file":name,"bytes":size,"sha256":expected,"source":"User-supplied copy of pinned LegalSeg_CSV"}
                print("Verified",name,size,flush=True);break
        if len(done)==len(FILES) or time.monotonic()-start>=a.wait:break
        time.sleep(6)
    (out/"verified-local.json").write_text(json.dumps({"dataset":"L-NLProc/LegalSeg_CSV","revision":"31217aa845866bbd18a8ad567a785e6f5485435a","files":done},indent=2)+"\n")
    if len(done)!=len(FILES):print("Still downloading:",sorted(set(FILES)-set(done)))
if __name__=="__main__":main()
