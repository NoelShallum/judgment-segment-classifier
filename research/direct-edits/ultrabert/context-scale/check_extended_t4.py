from pathlib import Path

stage=Path("/content/ultrabert-context-comparison")
if (stage/"pid").exists():
    pid=(stage/"pid").read_text().strip()
    p=Path("/proc")/pid/"stat"
    data=stage/"run.log"
    with data.open("rb") as f:
        f.seek(max(0,data.stat().st_size-5500))
        lines=f.read().decode(errors="replace").splitlines()[-12:]
    print("state",p.read_text().split()[2] if p.exists() else "exited",
          "ready",(stage/"extended-results.zip").is_file(),lines)
