import os
import subprocess
import sys
import time
from pathlib import Path
from fetch import ROOT
native=ROOT/'weights/ov-GPU-f32/local/pilot/validation.json'
proc=Path('/proc/416411/stat')
limit=time.monotonic()+900
while time.monotonic()<limit:
 if native.is_file() and (not proc.exists() or proc.read_text().split()[2]=='Z'):break
 if not proc.exists() and not native.exists():raise RuntimeError('Original local-context worker exited before validation, check openvino-f32-pilot.log')
 time.sleep(4)
else:raise RuntimeError('Native local GPU job has not handed off')
full=ROOT/'weights/ov-GPU-f32/alone/pilot/physical-checkpoint.json'
if not full.exists():subprocess.run([sys.executable,'materialize.py'],cwd=ROOT,check=True)
for name,args in [('physical',['ov_physical.py']),('packed',['packed.py','full-study','--device','GPU','--hint','f32'])]:
 with (ROOT/f'{name}-worker.log').open('w') as log:
  result=subprocess.run([sys.executable,'-u']+args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  print(name,'result',result.returncode,flush=True)
  if name=='packed' and result.returncode:raise RuntimeError('Packed worker stopped; inspect its receipt')
