"""Replay the actual learned PPLX weights through the publisher's model on the T4."""
import os
from pathlib import Path
import subprocess
import sys

study=Path("/content/attoflow-pplx")
python="""
from pathlib import Path
import hashlib, json
import torch
from safetensors.torch import load_file
from transformers import AutoModel
from native_pplx import t4, capture, put

where=Path.cwd()
device=t4()
plan=json.loads((where/'protocol.json').read_text())
corpus=json.loads((where/'cases.json').read_text())
model=AutoModel.from_pretrained(str(where/'pinned-model'),trust_remote_code=True,local_files_only=True,
        dtype=torch.float32,attn_implementation='sdpa').eval().requires_grad_(False).to('cuda')
original=model.contextual_projection.weight.detach().clone()
result={}
with torch.inference_mode():
    for mode in ('isolated','contextual'):
        path=where/'scored'/(mode+'-native-projection.safetensors')
        edit=load_file(path)['contextual_projection.weight'].to('cuda')
        assert torch.equal(original[7:],edit[7:])
        words=[r[1] for r in corpus[plan['shared'][0]][:12]]
        input_rows=[words] if mode=='contextual' else [[x] for x in words]
        x, features=capture(model,input_rows)
        native_linear=torch.nn.functional.linear(x,edit[:7])
        model.contextual_projection.weight.copy_(edit)
        native=model._pool(features,'document')[...,:7]
        err=(native-native_linear).abs().max().item()
        same=torch.equal(native.argmax(-1),native_linear.argmax(-1))
        assert err<1e-5 and same
        result[mode]={'targets':len(words),'max_score_difference':err,'all_labels_match':same,
                      'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
put(where/'learned-native-parity.json',{'model':plan['model']['model_id'],'device':device,'native_replay':result})
print(result,flush=True)
"""
env=dict(os.environ,PYTHONPATH=str(study/"dependencies"),USE_TF="0",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
subprocess.run([sys.executable,"-u","-c",python],cwd=study,env=env,check=True)
