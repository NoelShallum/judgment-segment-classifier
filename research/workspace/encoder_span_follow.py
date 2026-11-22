"""Start the larger comparison once the current CPU-only MiniLM pass has finished."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--minilm-pid",type=int,required=True)
    ap.add_argument("--granite-pid",type=int,required=True)
    args=ap.parse_args()
    root=Path(__file__).resolve().parent
    def still_running(pid,model):
        f=Path(f"/proc/{pid}/cmdline")
        try:
            cmd=f.read_bytes().split(b"\0")
            return b"encoder_span_edit.py" in cmd and model.encode() in cmd
        except FileNotFoundError:
            return False
    def launch(script,model,log):
        with (root/log).open("w") as f:
            return subprocess.run([sys.executable,"-u",script,model],stdout=f,stderr=subprocess.STDOUT,cwd=root)
    granite_norm=False
    while still_running(args.minilm_pid,"minilm") or still_running(args.granite_pid,"granite"):
        if not granite_norm and (root/"results/encoder-span-granite.json").is_file():
            launch("encoder_norm_edit.py","granite","encoder-norm-granite.log")
            granite_norm=True
        if not still_running(args.minilm_pid,"minilm"):
            break
        time.sleep(20)
    if not (root/"results/encoder-span-minilm.json").is_file():
        print("MiniLM terminated before producing results. Starting GTE for independent comparison.",flush=True)
    result=launch("encoder_span_edit.py","gte","encoder-span-gte.log")
    if result.returncode!=0:
        raise SystemExit(result.returncode)
    result=launch("encoder_norm_edit.py","gte","encoder-norm-gte.log")
    raise SystemExit(result.returncode)
if __name__=="__main__":
    main()
