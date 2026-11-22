"""Replay the same frozen UltraBERT weights with 2,048 tokens on the same cases."""
import hashlib
from pathlib import Path
import subprocess
from zipfile import ZipFile

gpu=subprocess.check_output(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],text=True).strip().splitlines()
assert len(gpu)==1 and gpu[0] in {"Tesla T4","NVIDIA T4","T4"}
work=Path("/content/ultrabert-context-comparison")
work.mkdir(exist_ok=True)
bundles={
    "/content/ultrabert-t4-worker-v2.zip":"9fca5fea170059eea547efc6cf2f2f8c7d403c0ace83d45009576b5b2e763afc",
    "/content/ultrabert-frozen-results.zip":"34177521a438c00437a7f976152f1e39c78689a81d6bc4f5c5cbf262d81c91be",
}
for path,digest in bundles.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
with ZipFile("/content/ultrabert-t4-worker-v2.zip") as z:
    assert z.testzip() is None
    z.extractall(work/"source")
with ZipFile("/content/ultrabert-frozen-results.zip") as z:
    target=work/"run"
    target.mkdir(exist_ok=True)
    for name in ("freeze.json","patch.safetensors"):
        (target/name).write_bytes(z.read("model/"+name))
entry=work/"ablation.sh"
entry.write_text(
    "set -euo pipefail\n"
    "export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 USE_TF=0\n"
    "python -u source/t4_guard.py\n"
    "if [[ ! -x venv/bin/python ]]; then python -m venv --without-pip --system-site-packages venv; fi\n"
    "python -m pip --python venv/bin/python install --disable-pip-version-check -q -r source/requirements.txt\n"
    "venv/bin/python -u source/t4_guard.py\n"
    "venv/bin/python -u source/fetch.py --out pinned-model\n"
    "venv/bin/python -u source/encode.py --manifest source/validation/manifest.json --out features --model pinned-model "
    "--partition shared --device cuda --cap 2048 --batch-size 2 --threads 2 --schedule shortest-first\n"
    "venv/bin/python -u source/fit.py evaluate --manifest source/validation/manifest.json "
    "--features features/native-fp32-v2-context-2048-cuda --run run --partition shared --incumbent source/validation/incumbent\n"
    "venv/bin/python -u -m zipfile -c extended-results.zip run/shared-report.json run/shared-predictions.npz\n"
)
with (work/"run.log").open("ab") as log:
    p=subprocess.Popen(["bash",str(entry)],cwd=work,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(work/"pid").write_text(str(p.pid))
print("Frozen UltraBERT T4 context comparison",p.pid)
