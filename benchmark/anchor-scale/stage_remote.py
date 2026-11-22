"""Extract the checksummed CPU route and run both stages in the existing T4 session."""
import hashlib
import json
import os
import subprocess
from pathlib import Path,PurePosixPath
from zipfile import ZipFile

r=Path('/content/reviewer2');n=int(os.environ['ANCHOR_CASES']);assert n in (256,512)
h=r/f'anchor-scale-{n}';h.mkdir(exist_ok=True)
p=r/f'anchor-input-{n}.zip'
with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==os.environ['ANCHOR_SHA']
with ZipFile(p) as z:
    assert z.testzip() is None
    assert all(len(PurePosixPath(name).parts)==1 for name in z.namelist())
    z.extractall(h)
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).strip()
assert gpu in {'Tesla T4','T4','NVIDIA T4'}
env=dict(os.environ,PYTHONPATH=':'.join(map(str,[h,r/'scale',r/'legal',r/'context',r/'deps'])),USE_TF='0',USE_FLAX='0',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='1')
code="""import hashlib,json,sys
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
h=Path(__file__).resolve().parent;n=int(sys.argv[1])
files=[*sorted((h/'head-fit').glob('*')),h/'returned/runtime-receipt.json',h/'manifest.json',h/'protocol.json',h/'fit_anchor.py',h/'t4_review.py']
dest=h.parent/f'anchor-{n}-dev.zip'
with ZipFile(dest.with_suffix('.part'),'w',ZIP_DEFLATED) as z:
    for p in files:z.write(p,str(p.relative_to(h)))
    z.writestr('SHA256SUMS.json',json.dumps({str(p.relative_to(h)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))
dest.with_suffix('.part').replace(dest)
"""
(h/'return_results.py').write_text(code)
pid=h/'pid'
if not (h/'run.exit').exists() and (not pid.exists() or not Path('/proc',pid.read_text().strip()).exists()):
    check=r/'donor.safetensors'
    cmd=f"set -eu\ntrap 'echo $? > run.exit' EXIT\npython -u t4_review.py --checkpoint {check}\nCUDA_VISIBLE_DEVICES='' python -u fit_anchor.py --scale ../scale --reviewed returned --blocks {n//64}\npython -u return_results.py {n}"
    with (h/'run.log').open('wb') as f:
        proc=subprocess.Popen(['bash','-c',cmd],cwd=h,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    pid.write_text(str(proc.pid))
print('Anchored training prefix',n,'running or complete',gpu)
