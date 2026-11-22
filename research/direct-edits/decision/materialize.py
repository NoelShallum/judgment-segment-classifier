"""Build a runnable copy of the publisher checkpoint. Replace only one native tensor's byte range."""
import argparse
import hashlib
import json
import os
import shutil
import struct
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from fetch import ROOT,MODEL,sha
from weights import INTERNAL,READOUT,ENC,HEAD,savej

def clone(source,target):
 try:
  import fcntl
  with source.open('rb') as inp,target.open('wb') as out:
   fcntl.ioctl(out.fileno(),0x40049409,inp.fileno())
 except OSError:shutil.copyfile(source,target)
def segment(path,key):
 with path.open('rb') as f:
  n=struct.unpack('<Q',f.read(8))[0]
  p=json.loads(f.read(n))[key]
 return n+8+p['data_offsets'][0],p['data_offsets'][1]-p['data_offsets'][0]
def hash_with_original(new,original,off,length):
 h=hashlib.sha256()
 with new.open('rb') as f,original.open('rb') as src:
  remaining=off
  while remaining:
   v=f.read(min(1<<20,remaining));h.update(v);remaining-=len(v)
  src.seek(off);h.update(src.read(length));f.seek(off+length)
  while part:=f.read(1<<20):h.update(part)
 return h.hexdigest()
def build(backend,arm,tier):
 base=ROOT/'weights'/backend/arm/tier
 proof=json.loads((base/'receipt.json').read_text())
 patched=base/'roles-internal.safetensors';target=base/'published-encoder-copy'
 target.mkdir(exist_ok=True);output=target/ENC.name;temp=output.with_suffix('.part')
 metadata=base/'physical-checkpoint.json'
 if metadata.exists() and output.is_file():
  m=json.loads(metadata.read_text())
  if sha(patched)==m['patch_sha256'] and sha(output)==m['edited_checkpoint_sha256']:return m
 assert sha(patched)==proof['files'][patched.name]['sha256']
 replacement=load_file(str(patched))[INTERNAL]
 offset,length=segment(ENC,INTERNAL)
 new_bytes=replacement.view(torch.uint8).numpy().tobytes()
 assert len(new_bytes)==length
 clone(ENC,temp)
 with temp.open('r+b') as f:
  f.seek(offset);f.write(new_bytes);f.flush();os.fsync(f.fileno())
 with safe_open(str(temp),'pt',device='cpu') as f:
  assert torch.equal(f.get_tensor(INTERNAL),replacement)
 witness=hash_with_original(temp,ENC,offset,length)
 assert witness==proof['encoder_sha256']==sha(ENC)
 temp.replace(output)
 for path in MODEL.iterdir():
  if path.name==ENC.name:continue
  link=target/path.name
  if not link.exists():link.symlink_to(os.path.relpath(path,target),target_is_directory=path.is_dir())
 info={'model_dir':str(target.resolve()),'tensor':INTERNAL,'source_checkpoint_sha256':witness,'edited_checkpoint_sha256':sha(output),
       'source_checkpoint_sha256_after_restoring_only_changed_range':witness,'byte_offset':offset,'bytes_in_tensor':length,
       'patch_sha256':sha(patched),'unchanged_original_readout_sha256':sha(target/HEAD.name),'source_tensor_name_and_size':proof['files'][patched.name],
       'head_only_native_file':str((base/'roles-readout.safetensors').resolve())}
 savej(metadata,info);print('PHYSICAL',backend,arm,info['edited_checkpoint_sha256'],flush=True)
 return info
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--backend',default='ov-GPU-f32');ap.add_argument('--arm',default='alone');ap.add_argument('--tier',default='pilot');args=ap.parse_args();build(args.backend,args.arm,args.tier)
