"""Run the exact pinned PPLX acquisition and staged T4 inference."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

archive=Path("/content/pplx-attoflow.zip")
assert hashlib.sha256(archive.read_bytes()).hexdigest()==os.environ["ATTOFLOW_PPLX_SHA"]
stage=os.environ.get("ATTOFLOW_PPLX_STAGE","smoke")
assert stage in ("smoke","full")
where=Path("/content/attoflow-pplx")
where.mkdir(exist_ok=True)
with ZipFile(archive) as z:
    assert z.testzip() is None
    z.extractall(where)
runner=where/(stage+".sh")
runner.write_text(
    "set -euo pipefail\n"
    "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 USE_TF=0 TOKENIZERS_PARALLELISM=false\n"
    "if [[ ! -f dependencies/transformers/__init__.py ]]; then\n"
    "  python -m pip -q --disable-pip-version-check install --target dependencies transformers==5.4.0\n"
    "fi\n"
    "export PYTHONPATH=/content/attoflow-pplx/dependencies\n"
    f"python -u native_pplx.py {stage}\n"
    + ("python -u score_projection.py\n" if stage=="full" else "echo 'PPLX SMOKE READY'\n")
)
with (where/(stage+".log")).open("ab") as f:
    p=subprocess.Popen(["bash",str(runner)],stdout=f,stderr=subprocess.STDOUT,cwd=where,start_new_session=True)
(where/(stage+".pid")).write_text(str(p.pid))
print("PPLX",stage,p.pid,flush=True)
