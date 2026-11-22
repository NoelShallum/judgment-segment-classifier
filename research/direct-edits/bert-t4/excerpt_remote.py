"""Replay the frozen T4 candidate on tokenized public excerpt inputs."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys

state=Path("/content/attoflow-inlegalbert-value")
archive=state/"excerpt-route.npz"
assert hashlib.sha256(archive.read_bytes()).hexdigest()==os.environ["ATTOFLOW_EXCERPT_SOURCE_SHA"]
source=state/"source-full"
program="""
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file
import torch
from native import NativeTail, VALUE, batches, donor_at, move, require_t4, save_arrays
from fit import Bank, Readout, audit_baseline, head_predict, replay

state=Path('/content/attoflow-inlegalbert-value')
source=state/'source-full'
device='cuda'
note=require_t4()
torch.set_num_threads(1)
model=donor_at(source,state/'inlegalbert.safetensors',device)
tail=NativeTail(model)
with np.load(state/'excerpt-route.npz',allow_pickle=False) as a:
    row={k:a[k] for k in a.files}
parts=[tail.capture(move(row['ids'][s],device,torch.long),move(row['mask'][s],device,torch.bool))[0]
       for s in batches(len(row['ids']),96)]
features={k:torch.cat([p[k] for p in parts]) for k in parts[0]}
count=len(row['ids'])
bank=Bank(row['doc_sha'].tolist(),np.arange(count+1),np.arange(count),np.zeros(count),row['base'],
          row['original'],row['prior'],move(row['aux'],device),features['keys'], features['cls'],features['pooled'],None)
head=Readout(source,device)
pred={'case model without edit':audit_baseline(bank,head)}
for name in ('targeted','shuffled'):
    patch=load_file(str(state/'run-full'/(name+'.safetensors')))
    pred[name]=replay(tail,head,bank,move(patch[VALUE+'.weight'],device),move(patch[VALUE+'.bias'],device))
new={k:move(v,device) for k,v in load_file(str(state/'run-full/refitted-heads.safetensors')).items()}
for name in ('pooled','attention-aware'):
    pred[name+' refit']=head_predict(head,bank,name,new)
save_arrays(state/'excerpt-predictions.npz',doc_sha=row['doc_sha'],**pred)
print('Matched T4 excerpts',count,note,flush=True)
"""
env=dict(os.environ,PYTHONPATH=os.pathsep.join((str(state/"dependencies"),str(source))),USE_TF="0",OMP_NUM_THREADS="1")
result=subprocess.run([sys.executable,"-u","-c",program],cwd=state,env=env,check=True)
assert result.returncode==0
