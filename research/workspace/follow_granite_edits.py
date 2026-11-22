"""Run follow-up edits once the active Granite comparison commits its report."""
from pathlib import Path
import subprocess
import sys
import time

def main():
    pid=int(sys.argv[1])
    root=Path(__file__).resolve().parent
    report=root/"results/encoder-span-granite.json"
    while not report.exists():
        path=Path(f"/proc/{pid}/cmdline")
        try:
            current=path.read_bytes().split(b"\0")
        except FileNotFoundError:
            current=[]
        if b"encoder_span_edit.py" not in current or b"granite" not in current:
            raise RuntimeError("Granite encoder pass ended without a completed report")
        time.sleep(20)
    for task,log in (("encoder_norm_edit.py","encoder-norm-granite.log"),
                     ("contextual_role_keys.py","contextual-keys-granite.log"),
                     ("sparse_span_memory.py","sparse-granite.log")):
        with (root/log).open("w") as f:
            completed=subprocess.run([sys.executable,"-u",task,"granite"],cwd=root,stdout=f,stderr=subprocess.STDOUT)
        print(task,completed.returncode,flush=True)
if __name__=="__main__":
    main()
