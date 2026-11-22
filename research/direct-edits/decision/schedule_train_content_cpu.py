"""Schedule one CPU-only training feature job alongside the ongoing precision ablation."""
import os,signal,sys,subprocess
from pathlib import Path
from fetch import ROOT
pid=408259
proc=Path('/proc')/str(pid)/'cmdline'
paused=False
try:
 if proc.exists() and b'run.py' in proc.read_bytes() and b'int8' in proc.read_bytes():
  os.kill(pid,signal.SIGSTOP);paused=True
 if '--stream' in sys.argv:
  args=[sys.executable,'-u','cpu_stream_content.py','--maximum','12','--wave-size','2']
 else:
  args=[sys.executable,'-u','content_tokens.py','--device','CPU','--extract-only']
  if len(sys.argv)>1:args+=['--limit-cases',sys.argv[1]]
  args+=['train']
 rc=subprocess.run(args,cwd=ROOT,timeout=3600)
 if rc.returncode:raise RuntimeError('CPU content job failed')
finally:
 if paused:
  try:os.kill(pid,signal.SIGCONT)
  except ProcessLookupError:pass
