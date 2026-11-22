"""Launch from the coordinator's existing attoflow-role-context-t4 Colab only."""

from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

root = Path("/content/reviewer2")
here = root / "entailment-edit"
here.mkdir(exist_ok=True)
gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
assert gpu in {"Tesla T4", "NVIDIA T4", "T4"}
expected = {"donor.json", "plan.json", "preparation.json", "protocol.py", "remote_capture.py", "windows.jsonl.gz"}
package = root / "entailment-input.zip"
with package.open("rb") as handle:
    assert file_digest(handle, "sha256").hexdigest() == os.environ["ATTO_ENTAIL_SHA256"]
with ZipFile(package) as payload:
    assert set(payload.namelist()) == expected | {"input-manifest.json"}
    manifest = json.loads(payload.read("input-manifest.json"))
    assert set(manifest) == expected
    for name, checksum in manifest.items():
        assert sha256(payload.read(name)).hexdigest() == checksum
    payload.extractall(here)
flag = here / "capture.exit"
flag.unlink(missing_ok=True)
env = dict(os.environ, PYTHONPATH=str(root / "deps"), HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_DISABLE_XET="1",
           HF_HUB_DOWNLOAD_TIMEOUT="90", USE_TF="0", USE_FLAX="0", OMP_NUM_THREADS="2",
           OPENBLAS_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
import shlex

script = "set -eu\ntrap 'echo $? > capture.exit' EXIT\n" + shlex.join([sys.executable, "-u", "remote_capture.py"])
with (here / "capture.log").open("wb") as log:
    worker = subprocess.Popen(["bash", "-c", script], cwd=here, env=env, stdout=log,
                              stderr=subprocess.STDOUT, start_new_session=True)
print("Started pinned ModernBERT NLI capture", worker.pid, gpu)
