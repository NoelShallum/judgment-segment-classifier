from pathlib import Path
import subprocess,sys,time
from fetch import ROOT
deadline=time.monotonic()+7200
while time.monotonic()<deadline:
 src=ROOT/'queue.log'
 if src.exists():
  text=src.read_text()
  if 'packed result 0' in text:break
  if 'packed result ' in text:raise RuntimeError('Prior GPU worker failed')
 time.sleep(8)
else:raise TimeoutError('waiting for packed comparisons')
from materialize import build
build('ov-GPU-f32','packed-ordered','whole16')
subprocess.run([sys.executable,'-u','ov_physical.py','--arm','packed-ordered','--tier','whole16'],cwd=ROOT,check=True)
from report import create
create()
