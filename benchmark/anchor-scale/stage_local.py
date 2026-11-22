import hashlib
from pathlib import Path
import subprocess
import sys

h=Path(__file__).resolve().parent;n=int(sys.argv[1]);assert n in (256,512)
p=h/f't4-anchor-{n}.zip'
s='attoflow-role-context-t4'
subprocess.run(['colab','upload','-s',s,str(p),f'/content/reviewer2/anchor-input-{n}.zip'],check=True)
cmd=['colab','exec','-s',s,'--timeout','60','--env',f'ANCHOR_CASES={n}','--env','ANCHOR_SHA='+hashlib.sha256(p.read_bytes()).hexdigest(),'--file',str(h/'stage_remote.py')]
for _ in range(3):
    result=subprocess.run(cmd,capture_output=True,text=True)
    if result.returncode==0:
        print(result.stdout.strip());break
    print(result.stderr.splitlines()[-1:] or result.stdout.splitlines()[-1:],flush=True)
else:raise SystemExit(1)
