"""Stop the active Karpo job and preserve completed shards without running the model."""
import hashlib
import json
import os
from pathlib import Path
import signal
import time
from zipfile import ZIP_DEFLATED, ZipFile

root=Path("/content/attoflow-karpo")
cmd=root/"full.pid"
suspended=None
if cmd.exists():
    pid=int(cmd.read_text().strip())
    proc=Path("/proc")/str(pid)
    if (proc/"cmdline").exists() and b"run-full.sh" in (proc/"cmdline").read_bytes():
        os.killpg(os.getpgid(pid),signal.SIGSTOP)
        suspended=pid
        time.sleep(.10)
dest=root/"checkpoint-when-goal-cleared.zip"
data=root/"run"
preserve=[p for p in data.rglob("*") if p.is_file() and p.suffix in {".json",".npz",".safetensors"}]
preserve.extend(p for p in (root/"t4-full.log",root/"protocol.json",root/"token-plan.json",root/"source.json") if p.is_file())
notice={"suspended_process":suspended,"utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        "archived":len(preserve),"validation_scored":(data/"validation.json").is_file(),
        "completed_judgment_receipts":[str(p.relative_to(root)) for p in preserve if p.name=="complete.json"]}
with ZipFile(dest.with_suffix(".part"),"w",compression=ZIP_DEFLATED,compresslevel=2) as z:
    for f in sorted(preserve):
        z.write(f,str(f.relative_to(root)))
    z.writestr("STOP-RECEIPT.json",json.dumps(notice,indent=2))
dest.with_suffix(".part").replace(dest)
with dest.open("rb") as f:
    print("Stop snapshot",dest.stat().st_size,hashlib.file_digest(f,"sha256").hexdigest(),notice,flush=True)
