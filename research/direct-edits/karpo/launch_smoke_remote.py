from pathlib import Path
import hashlib
import subprocess
from zipfile import ZipFile

devices=subprocess.check_output(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],text=True).strip().splitlines()
assert len(devices)==1 and devices[0] in {"Tesla T4","NVIDIA T4","T4"}
archive=Path("/content/karpo-t4-smoke-worker.zip")
assert hashlib.sha256(archive.read_bytes()).hexdigest()=="3b22cd771495e8b5a13be8f1a4e2905b0fd7a50006f155a483aded91a70d355a"
state=Path("/content/attoflow-karpo")
state.mkdir(exist_ok=True)
with ZipFile(archive) as z:
    assert z.testzip() is None
    z.extractall(state)
with (state/"t4-smoke.log").open("ab") as log:
    p=subprocess.Popen(["bash",str(state/"t4.sh"),"smoke"],cwd=state,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(state/"smoke.pid").write_text(str(p.pid))
print("Karpo T4 smoke",p.pid)
