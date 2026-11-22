import os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
for g in ('confirmation','pilot','paragraphs','homogeneous'):
    subprocess.run([sys.executable,str(root/'matched_metrics.py'),'--corpus',g],check=True)
