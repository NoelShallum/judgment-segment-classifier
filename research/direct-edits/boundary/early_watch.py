import time,os,signal,sys,json,subprocess
from pathlib import Path
p=Path(__file__).resolve().parent;pid=int(sys.argv[1]);ids=json.loads((p/'protocol.json').read_text())['fit'][:5]
while not all((p/'features'/f'train-{k}.npz').exists() for k in ids):
    if not Path(f'/proc/{pid}').exists():raise RuntimeError('collector stopped')
    time.sleep(4)
assert Path(f'/proc/{pid}/cwd').resolve()==p and b'collect.py\x00all\x00' in Path(f'/proc/{pid}/cmdline').read_bytes()
os.kill(pid,signal.SIGSTOP)
try:subprocess.run([str(p.parents[1]/'.venv/bin/python'),'-u','early_check.py'],cwd=p,check=True)
finally:os.kill(pid,signal.SIGCONT)
