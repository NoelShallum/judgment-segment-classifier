from pathlib import Path

state = Path("/content/attoflow-inlegalbert-packed")
p = state/"pid"
log = state/"run.log"
if p.is_file() and log.is_file():
    child = Path("/proc")/p.read_text().strip()/"stat"
    with log.open("rb") as f:
        f.seek(max(0,log.stat().st_size-7000))
        recent = f.read().decode(errors="replace").splitlines()[-14:]
    print("packed",child.read_text().split()[2] if child.exists() else "exited",
          "cases encoded",len(list((state/"features").glob("*.json"))),
          "result archived",(state/"packed-results.zip").is_file())
    print("\n".join(recent))
else:
    print("Packed worker has not started.")
