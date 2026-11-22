import os,sys,subprocess,signal
from pathlib import Path
p=Path(__file__).resolve().parent;pid=int(sys.argv[1]);assert Path(f'/proc/{pid}/cwd').resolve()==p and b'collect.py\x00all\x00' in Path(f'/proc/{pid}/cmdline').read_bytes();count=len(list((p/'features').glob('train-*.npz')));assert count<16 or (sys.argv[2]=='verify_edits.py' and 17<=count<=20)
os.kill(pid,signal.SIGSTOP)
try:subprocess.run([str(p.parents[1]/'.venv/bin/python'),'-u',*sys.argv[2:]],cwd=p,check=True)
finally:os.kill(pid,signal.SIGCONT)
