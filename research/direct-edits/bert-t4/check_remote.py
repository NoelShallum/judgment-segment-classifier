from pathlib import Path
import subprocess

state=Path("/content/attoflow-inlegalbert-value")
for phase in ("priority","full"):
    marker=state/("pid-"+phase)
    if not marker.is_file():
        continue
    pid=marker.read_text().strip()
    p=Path("/proc")/pid/"stat"
    log=state/(phase+".log")
    out=state/(phase+"-results.zip")
    with log.open("rb") as f:
        f.seek(max(0,log.stat().st_size-4800))
        excerpt=f.read().decode(errors="replace").splitlines()[-9:]
    print(phase,"launcher",p.read_text().split()[2] if p.exists() else "exited",
          "result",out.stat().st_size if out.is_file() else None,"log",excerpt)
print(subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.used,utilization.gpu","--format=csv,noheader"],text=True).strip())
