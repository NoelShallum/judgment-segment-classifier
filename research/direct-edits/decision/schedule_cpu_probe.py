"""Temporarily give the single assigned CPU core to the targeted mechanism probe."""
import os,signal,subprocess,sys
from pathlib import Path
from fetch import ROOT
existing=408259
check=Path('/proc')/str(existing)/'cmdline'
paused=False
try:
 if check.exists() and b'run.py' in check.read_bytes() and b'int8' in check.read_bytes():
  os.kill(existing,signal.SIGSTOP);paused=True
 result=subprocess.run([sys.executable,'-u','cpu_query_probe.py'],cwd=ROOT,timeout=900)
 if result.returncode:raise RuntimeError('CPU attention probe failed')
finally:
 if paused:
  try:os.kill(existing,signal.SIGCONT)
  except ProcessLookupError:pass
