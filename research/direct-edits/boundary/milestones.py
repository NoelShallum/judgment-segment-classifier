"""Score milestones while the single-thread extractor yields its CPU. Resume-safe."""
import os,json,sys,time,subprocess,signal
from pathlib import Path
P=Path(__file__).resolve().parent
PRO=json.loads((P/'protocol.json').read_text()); pid=int(sys.argv[1]); binary=P.parents[1]/'.venv/bin/python'
def ours():
    proc=Path(f'/proc/{pid}')
    try: return (proc/'cwd').resolve()==P and b'collect.py\x00all\x00' in (proc/'cmdline').read_bytes()
    except FileNotFoundError:return False
for split,ids,command,output in [('train',PRO['select'],['score.py','fit'],'freeze.json'),('train',PRO['check'],['score.py','check'],'check.json'),('val',PRO['interim'],['score.py','four'],'four.json'),('val',PRO['validation'],['score.py','sixteen'],'sixteen.json'),('val',PRO['validation'],['verify_edits.py'],'forward-parity.json')]:
    if (P/'results'/output).exists():continue
    while not all((P/'features'/f'{split}-{k}.npz').exists() for k in ids):
        if not ours():raise RuntimeError('Extractor stopped before all feature files were available: '+output)
        time.sleep(5)
    paused=ours()
    if paused:os.kill(pid,signal.SIGSTOP)
    try:subprocess.run([str(binary),'-u',*command],cwd=P,check=True)
    finally:
        if paused and ours():os.kill(pid,signal.SIGCONT)
