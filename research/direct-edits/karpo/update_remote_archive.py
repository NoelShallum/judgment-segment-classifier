from pathlib import Path

where=Path("/content/attoflow-karpo")
archive=where/"archive.py"
old=archive.read_text()
before="selected=[p for p in w.glob('*.json')]+[p for p in (w/'fit').iterdir() if p.suffix in ('.json','.safetensors')]\nselected.extend(p for p in w.rglob('complete.json'))"
after="""selected=[p for p in w.rglob('*') if p.is_file() and p.suffix in ('.json','.safetensors','.npz')]
assert all((w/'fit'/arm/'readout.safetensors').is_file() for arm in ('edited','shuffled'))"""
assert before in old or after in old
contents=old.replace(before,after)
if "Karpo input-order sensitivity" not in contents:
    contents=contents.replace("w=Path('run')",'''w=Path('run')
# Karpo input-order sensitivity reuses the same frozen heads and scored cases.
import os
import subprocess
env=dict(os.environ,PYTHONPATH=".karpo-t4-deps")
subprocess.run(["bash","t4.sh","sensitivity"],check=True)
subprocess.run(["python","-u","program.py","validation"],env=env,check=True)''')
temp=archive.with_suffix(".tmp")
temp.write_text(contents)
temp.replace(archive)
print("Karpo archive now includes both physical readouts and order diagnostics.")
