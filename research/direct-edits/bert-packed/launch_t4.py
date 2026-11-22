"""Start the uploaded packed study in the current verified Colab session."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

card = subprocess.check_output(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],text=True).strip().splitlines()
assert len(card) == 1 and card[0] in {"Tesla T4","NVIDIA T4","T4"},card
archive = Path("/content/attoflow-packed-worker.zip")
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
assert digest == os.environ["ATTOFLOW_PACKED_ARCHIVE_SHA256"]
state = Path("/content/attoflow-inlegalbert-packed")
state.mkdir(exist_ok=True)
pidfile = state/"pid"
if pidfile.exists():
    process = Path("/proc")/pidfile.read_text().strip()/"cmdline"
    assert not process.is_file() or b"packed-start.sh" not in process.read_bytes()
source = state/"data"
source.mkdir(exist_ok=True)
with ZipFile(archive) as bundle:
    checks = json.loads(bundle.read("contents-sha256.json"))
    assert all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in checks)
    for name,check in checks.items():
        assert hashlib.sha256(bundle.read(name)).hexdigest() == check
        bundle.extract(name,source)
entry = state/"packed-start.sh"
entry.write_text(
    "set -euo pipefail\n"
    "export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false USE_TF=0\n"
    "cd /content/attoflow-inlegalbert-packed\n"
    "dependencies=/content/attoflow-inlegalbert-value/dependencies\n"
    "if [[ ! -d \"$dependencies/transformers\" ]]; then\n"
    "  dependencies=/content/attoflow-inlegalbert-packed/dependencies\n"
    "  python -m pip -q --disable-pip-version-check install --target \"$dependencies\" --no-deps "
    "transformers==4.53.3 tokenizers==0.21.4 huggingface-hub==0.34.4\n"
    "fi\n"
    'export PYTHONPATH="$dependencies${PYTHONPATH:+:$PYTHONPATH}"\n'
    "checkpoint=/content/attoflow-inlegalbert-value/inlegalbert.safetensors\n"
    "if [[ -f \"$checkpoint\" ]]; then\n"
    "  python -u data/run_t4.py data . --checkpoint \"$checkpoint\"\n"
    "else\n"
    "  python -u data/run_t4.py data .\n"
    "fi\n"
)
with (state/"run.log").open("ab") as log:
    child = subprocess.Popen(["bash",str(entry)],cwd=state,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
pidfile.write_text(str(child.pid))
print("Packed InLegalBERT study started",child.pid,digest)
