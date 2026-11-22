"""Prefer GPU only after the source export passes torch equivalence."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT=Path(__file__).resolve().parent

def launch(command, log):
    with (ROOT/log).open("w") as file:
        return subprocess.run(command,cwd=ROOT,stdout=file,stderr=subprocess.STDOUT).returncode

def main():
    run=["bash","runtime/with-intel-runtime.sh",sys.executable,"-u","encoder_span_edit.py","granite","--backend","openvino-source","--threads","1"]
    if launch(run+["--parity-only"],"granite-source-parity.log"):
        print("Granite CPU worker remains active after parity failure.",flush=True)
        return
    if (ROOT/"results/encoder-span-granite.json").is_file():
        return
    pid=int(sys.argv[1])
    path=Path(f"/proc/{pid}/cmdline")
    if not path.exists():return
    args=path.read_bytes().split(b"\0")
    if not (b"encoder_span_edit.py" in args and b"granite" in args):
        return
    # Preserve cached exact tensors and use the validated graph for remaining inputs.
    os.kill(pid,signal.SIGTERM)
    supervisor=int(sys.argv[2])
    cmd=Path(f"/proc/{supervisor}/cmdline")
    if cmd.exists() and b"follow_granite_edits.py" in cmd.read_bytes():
        os.kill(supervisor,signal.SIGTERM)
    base=ROOT/"cache/encoder-span-edit-v2"
    dest=base/"granite-gpu"
    if not dest.exists():dest.symlink_to(base/"granite",target_is_directory=True)
    result=launch(run,"encoder-span-granite-accelerated.log")
    if result:
        result=launch([sys.executable,"-u","encoder_span_edit.py","granite","--threads","2"],"encoder-span-granite-resumed-cpu.log")
    if result:return
    for target,label in (("encoder_norm_edit.py","encoder-norm-granite.log"),
                         ("contextual_role_keys.py","contextual-keys-granite.log"),
                         ("sparse_span_memory.py","sparse-granite.log")):
        print(target,launch([sys.executable,"-u",target,"granite"],label),flush=True)
    subprocess.run([sys.executable,"render_span_research.py"],cwd=ROOT,check=True)
if __name__=="__main__":
    main()
