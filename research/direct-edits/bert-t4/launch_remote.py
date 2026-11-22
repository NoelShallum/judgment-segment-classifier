"""Enter a staged value-weight study on the existing verified Colab T4."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

state=Path("/content/attoflow-inlegalbert-value")
stage=os.environ.get("ATTOFLOW_BERT_STAGE","priority")
assert stage in ("priority","full")
device=Path("/content/attoflow-t4/device.json")
assert "T4" in json.loads(device.read_text())["nvidia_smi"][0].split(",")[0]
source=Path("/content/attoflow-bert-value-"+stage+".zip")
digest=hashlib.sha256(source.read_bytes()).hexdigest()
assert digest == os.environ["ATTOFLOW_BERT_ARCHIVE_SHA256"]
state.mkdir(exist_ok=True)
directory=state/("source-"+stage)
directory.mkdir(exist_ok=True)
with zipfile.ZipFile(source) as archive:
    assert archive.testzip() is None
    archive.extractall(directory)
entry=state/("start-"+stage+".sh")
entry.write_text(
    "set -euo pipefail\n"
    "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false USE_TF=0\n"
    "cd /content/attoflow-inlegalbert-value\n"
    "if [[ ! -d dependencies/transformers ]]; then\n"
    "  python -m pip -q --disable-pip-version-check install --target dependencies --no-deps "
    "transformers==4.53.3 tokenizers==0.21.4 huggingface-hub==0.34.4\n"
    "fi\n"
    'export PYTHONPATH="/content/attoflow-inlegalbert-value/dependencies${PYTHONPATH:+:$PYTHONPATH}"\n'
    + f"python -u {directory}/run_t4.py {directory} {state} {stage}\n"
)
status=state/("pid-"+stage)
if status.is_file():
    p=Path("/proc")/status.read_text().strip()/"cmdline"
    assert not p.is_file() or entry.name.encode() not in p.read_bytes(), "Existing study still running."
with (state/(stage+".log")).open("ab") as log:
    child = subprocess.Popen(["bash",str(entry)],cwd=state,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
status.write_text(str(child.pid))
print(stage, "T4 study started", child.pid, digest)
