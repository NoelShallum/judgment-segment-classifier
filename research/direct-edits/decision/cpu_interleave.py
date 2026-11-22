"""Give one CPU-only diagnostic the core used by this experiment's long extraction."""
import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

from fetch import ROOT

parser = argparse.ArgumentParser()
parser.add_argument("pid", type=int)
parser.add_argument("script", choices=["broader_lexical.py", "joint_content.py", "attention_write.py", "per_head.py"])
parser.add_argument("args", nargs="*")
args = parser.parse_args()
proc = Path("/proc") / str(args.pid)
paused = False
try:
    if proc.exists():
        command = (proc / "cmdline").read_bytes()
        if b"cpu_stream_content.py" not in command:
            raise RuntimeError("CPU extractor identity changed before interleaving")
        os.kill(args.pid, signal.SIGSTOP)
        paused = True
    cmd = [sys.executable, "-u", args.script, *args.args]
    subprocess.run(cmd, cwd=ROOT, check=True, timeout=1500)
finally:
    if paused:
        try:
            os.kill(args.pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
