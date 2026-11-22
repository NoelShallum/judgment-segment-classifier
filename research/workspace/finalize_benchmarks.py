"""Run full scoring as soon as all frozen inputs exist, and log each final result."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/"results"
N_BLOCKS=293


def ready_blocks():
    base=ROOT/"cache/full-test-fp32-v1"
    files={p.stem for folder in [base,base/"openvino-gpu-fp32",base/"openvino-cpu-fp32"]
           for p in folder.glob("[0-9][0-9][0-9][0-9].json") if p.with_suffix(".npy").is_file()}
    return len(files)


def launch(name, argv):
    handle=(ROOT/(name+".log")).open("ab",buffering=0)
    env={**os.environ,"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"2","ROLE_CPU_THREADS":"2"}
    try:
        child=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
    finally:
        handle.close()
    print("Started",name,flush=True)
    return child


def main():
    # This supervisor owns no GPU sessions. It only follows local, task-specific files.
    direct=RESULTS/"direct-official-test.json"
    jobs={}
    while not direct.is_file():
        for key,proc in list(jobs.items()):
            code=proc.poll()
            if code is None:continue
            del jobs[key]
            if code:raise RuntimeError(f"{key} exited with code {code}")
            print("Completed",key,flush=True)
        if not direct.is_file() and "direct" not in jobs and ready_blocks()==N_BLOCKS:
            jobs["direct"]=launch("final-direct",[sys.executable,"-u","score_official_test.py","direct"])
        time.sleep(15)
    print("Official direct-weight test results available.",flush=True)


if __name__=="__main__":
    main()
