"""Start the joint head fit using only existing CPU and cached representations."""

from pathlib import Path
import hashlib
import os
import subprocess
import sys
from zipfile import ZipFile

state = Path("/content/attoflow-inlegalbert-packed")
assert (state/"run-corrected/shared48.json").is_file()
archive = Path("/content/attoflow-packed-hybrid.zip")
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
assert digest == os.environ["ATTOFLOW_PACKED_HYBRID_SHA256"]
pid = state/"hybrid.pid"
if pid.exists():
    command = Path("/proc")/pid.read_text().strip()/"cmdline"
    assert not command.is_file() or b"hybrid.py" not in command.read_bytes()
work = state/"hybrid-input"
work.mkdir(exist_ok=True)
with ZipFile(archive) as z:
    assert z.testzip() is None
    assert all(not Path(k).is_absolute() and ".." not in Path(k).parts for k in z.namelist())
    z.extractall(work)
deps = Path("/content/attoflow-inlegalbert-value/dependencies")
if not (deps/"transformers").exists():
    deps = state/"dependencies"
env = dict(os.environ,OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",
           TOKENIZERS_PARALLELISM="false",USE_TF="0",CUDA_VISIBLE_DEVICES="",
           PYTHONPATH=str(deps)+os.pathsep+os.environ.get("PYTHONPATH",""))
with (state/"hybrid.log").open("ab") as log:
    worker = subprocess.Popen([sys.executable,"-u",str(work/"hybrid.py"),str(state),str(work)],
                              stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
pid.write_text(str(worker.pid))
print("Packed joint CPU fit started",worker.pid,digest)
