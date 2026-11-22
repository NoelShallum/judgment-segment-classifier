"""Freeze the contextual-regime stage before conditionally evaluating the public test."""
import json
import subprocess
import sys
from pathlib import Path

import regime

h=Path(__file__).resolve().parent
stages=(24,72)
locks={n:json.loads((h/f'regime-{n}-freeze.json').read_text()) for n in stages}
n=max(stages,key=lambda k:locks[k]['quality'])
development=json.loads((h/f'regime-{n}-dev.json').read_text())
ok=locks[n]['config']['strength']>0 and all(
    development[g]['all']['selected']['macro_f1']>development[g]['all']['context']['macro_f1'] and
    development[g]['all']['selected']['pair_errors']<development[g]['all']['context']['pair_errors']
    for g in ('confirm','fresh'))
record=dict(chosen_stage=n,quality={k:float(v['quality']) for k,v in locks.items()},promoted=bool(ok),
            config=locks[n]['config'],receipts={k:regime.sha(h/f'regime-{k}-freeze.json') for k in locks},
            protocol_sha256=regime.sha(h/'protocol.json'))
regime.scale.dump(h/'outcome.json',record)
if ok:subprocess.run([sys.executable,'-u',str(h/'regime.py'),str(n),'official'],check=True)
print(record)
