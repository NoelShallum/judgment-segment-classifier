"""Run via colab exec after uploading the task bundle to the private user runtime."""
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

root = Path("/content/attoflow-role")
root.mkdir(exist_ok=True)
pid_file = root / "worker.pid"
running = False
if pid_file.is_file():
    try:
        pid = int(pid_file.read_text())
        running = b"colab_worker.py" in Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, ValueError):
        pass
if running:
    print("The benchmark worker is already running.")
elif (root / "results/direct-official-test.json").is_file():
    print("The full-test result is available.")
else:
    with ZipFile("/content/attoflow-role-bundle.zip") as archive:
        archive.extractall(root)
    with (root / "worker.log").open("ab", buffering=0) as log:
        child = subprocess.Popen([sys.executable, "-u", "colab_worker.py"], cwd=root,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    pid_file.write_text(str(child.pid))
    print("Launched the benchmark on the private Colab session.")
