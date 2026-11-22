from pathlib import Path
import hashlib
import subprocess
from zipfile import ZipFile

devices=subprocess.check_output(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],text=True).strip().splitlines()
assert len(devices)==1 and devices[0] in {"Tesla T4","NVIDIA T4","T4"}
archive=Path("/content/karpo-t4-full-worker.zip")
assert hashlib.sha256(archive.read_bytes()).hexdigest()=="03f971f92957781594ec0ed1ffa2bed79a748092ce44a4561322db4f3734be70"
state=Path("/content/attoflow-karpo")
state.mkdir(exist_ok=True)
with ZipFile(archive) as z:
    assert z.testzip() is None
    z.extractall(state)
entry=state/"run-full.sh"
entry.write_text(
    "set -euo pipefail\n"
    "export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 USE_TF=0\n"
    "bash t4.sh train\n"
    "PYTHONPATH=.karpo-t4-deps python -u program.py fit\n"
    "bash t4.sh validation\n"
    "bash t4.sh sensitivity\n"
    "PYTHONPATH=.karpo-t4-deps python -u program.py validation\n"
    "python -u archive.py\n"
)
archive_script=state/"archive.py"
archive_script.write_text("""
from hashlib import file_digest
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
w=Path('run')
assert (w/'validation.json').is_file()
selected=[p for p in w.rglob('*') if p.is_file() and p.suffix in ('.json','.safetensors','.npz')]
assert all((w/'fit'/arm/'readout.safetensors').is_file() for arm in ('edited','shuffled'))
with ZipFile('karpo-results.part','w',compression=ZIP_DEFLATED,compresslevel=3) as z:
    for f in sorted(set(selected)):z.write(f,str(f))
    for f in ('source.json','protocol.json','token-plan.json'):z.write(f,f)
Path('karpo-results.part').replace('karpo-results.zip')
with open('karpo-results.zip','rb') as stream: print('Karpo results',file_digest(stream,'sha256').hexdigest(),flush=True)
""")
with (state/"t4-full.log").open("ab") as log:
    p=subprocess.Popen(["bash",str(entry)],cwd=state,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(state/"full.pid").write_text(str(p.pid))
print("Karpo full T4 study",p.pid,flush=True)
