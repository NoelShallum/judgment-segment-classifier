from pathlib import Path
import subprocess

where=Path("/content/attoflow-karpo")
for arm in ("smoke","pilot","full"):
    p=where/(arm+".pid")
    f=where/("t4-"+arm+".log")
    if not p.is_file() or not f.is_file():
        continue
    live=Path("/proc")/p.read_text().strip()/"stat"
    with f.open("rb") as stream:
        stream.seek(max(0,f.stat().st_size-5200))
        tail=stream.read().decode(errors="replace").splitlines()[-12:]
    print(arm,live.read_text().split()[2] if live.exists() else "exited",tail)
print(subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.used,utilization.gpu","--format=csv,noheader"],text=True).strip())
print("Receipts",[str(z.relative_to(where)) for z in sorted(where.rglob("smoke.json"))])
