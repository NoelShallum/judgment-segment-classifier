from pathlib import Path
import subprocess

state=Path("/content/attoflow-pplx")
for phase in ("smoke","full"):
    marker=state/(phase+".pid")
    if not marker.exists():
        continue
    process=Path("/proc")/marker.read_text().strip()/"stat"
    logs=state/(phase+".log")
    with logs.open("rb") as opened:
        opened.seek(max(0,logs.stat().st_size-4200))
        lines=opened.read().decode(errors="replace").splitlines()[-10:]
    print(phase,"state",process.read_text().split()[2] if process.exists() else "exited","features",
          len(list((state/"features").glob("*.json"))),"report",(state/"results.zip").exists(),lines)
print(subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.used,utilization.gpu","--format=csv,noheader"],text=True).strip())
