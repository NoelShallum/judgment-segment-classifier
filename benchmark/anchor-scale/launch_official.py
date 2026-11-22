"""Enforce the registered confirmation rule before the selected anchor replay."""
import hashlib
import os
import subprocess
from pathlib import Path

r=Path('/content/reviewer2');p=r/'anchor_official.py'
with p.open('rb') as fd:assert hashlib.file_digest(fd,'sha256').hexdigest()==os.environ['ANCHOR_EVAL_SHA']
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).strip()
assert gpu in {'Tesla T4','NVIDIA T4','T4'}
h=r/'anchor-official';h.mkdir(exist_ok=True)
worker=h/'promote.py'
worker.write_text('''import json,subprocess,sys,time
from pathlib import Path
r=Path('/content/reviewer2');h=r/'anchor-official'
for n in (256,512):
    flag=r/f'anchor-scale-{n}/run.exit'
    while not flag.exists() or not flag.read_text().strip():time.sleep(4)
    assert flag.read_text().strip()=='0'
cmd=[sys.executable,'-u',str(r/'anchor_official.py')]
subprocess.run(cmd+['--freeze-only'],check=True)
locked=json.loads((h/'official-lock.json').read_text());n=locked['chosen_cases'];arm=locked['chosen_arm']
groups=json.loads((r/f'anchor-scale-{n}/head-fit/results.json').read_text())['groups']
positive=all(groups[g]['scores'][arm]['all']['macro_f1']>groups[g]['scores']['doc-semantic']['all']['macro_f1'] and groups[g]['scores'][arm]['all']['pair_errors']<groups[g]['scores']['doc-semantic']['all']['pair_errors'] for g in ('confirm','fresh'))
(h/'promotion.json').write_text(json.dumps(dict(chosen_cases=n,chosen_arm=arm,promoted=positive),indent=2))
print('Locked anchor confirmation outcome',n,arm,positive,flush=True)
if positive:
    if arm=='pre-review':
        pre=h/'pre-review.npz'
        while not pre.exists():time.sleep(4)
        cmd+=['--pre-review',str(pre)]
    subprocess.run(cmd,check=True)
''')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=':'.join(str(r/k) for k in ('scale','legal','context','deps')),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='2',USE_TF='0',USE_FLAX='0')
pid=h/'pid'
if not (h/'run.exit').exists() and (not pid.exists() or not Path('/proc',pid.read_text().strip()).exists()):
    with (h/'run.log').open('wb') as out:
        job=subprocess.Popen(['bash','-c',"set -eu\ntrap 'echo $? > run.exit' EXIT\npython -u promote.py"],cwd=h,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    pid.write_text(str(job.pid))
print('Queued independent anchor evaluation',gpu)
