from pathlib import Path

state = Path("/content/attoflow-inlegalbert-packed")
log = state/"hybrid.log"
pid = state/"hybrid.pid"
if pid.exists() and log.exists():
    proc = Path("/proc")/pid.read_text().strip()/"stat"
    print("joint CPU worker",proc.read_text().split()[2] if proc.exists() else "exited",
          "archive",(state/"packed-hybrid-results.zip").exists())
    print("\n".join(log.read_text().splitlines()[-14:]))
