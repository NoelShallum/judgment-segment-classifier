"""Download named files at the publisher's pinned revision. Validate LFS objects."""
import concurrent.futures
import hashlib
import json
import os
import time
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parent
MODEL=ROOT/'model'
REV='16fb9d63cc34d1b4b5a1fb7cbf2d9d4a3dd5ad71'
REPO='fastino/GLiDE-0.4B'
FILES=('glide/encoder.py','glide/model.py','glide/prompt.py','decision_config.json','config.json','README.md','LICENSE','NOTICE','licenses/Ettin-encoder-400m-LICENSE','model.safetensors.index.json','special_tokens_map.json','tokenizer.json','tokenizer_config.json','readout.safetensors','render_tokenizer/tokenizer.json','render_tokenizer/tokenizer_config.json','render_tokenizer/chat_template.jinja','model-00001-of-00001.safetensors')
def sha(path):
 with open(path,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def fetch(entry):
 file=entry['rfilename']
 dest=MODEL/file
 dest.parent.mkdir(parents=True,exist_ok=True)
 exp=entry.get('lfs',{}).get('sha256')
 if not dest.exists() or dest.stat().st_size!=entry['size']:
  for tries in range(4):
   try:
    with requests.get('https://huggingface.co/'+REPO+'/resolve/'+REV+'/'+file,stream=True,timeout=(20,90)) as resp:
     resp.raise_for_status()
     with dest.with_suffix(dest.suffix+'.part').open('wb') as wr:
      for chunk in resp.iter_content(1024*1024):
       wr.write(chunk)
    os.replace(dest.with_suffix(dest.suffix+'.part'),dest)
    break
   except Exception:
    if tries==3: raise
    time.sleep(2)
 digest=sha(dest)
 assert dest.stat().st_size==entry['size'] and (not exp or digest==exp),file
 print(file,entry['size'],digest[:12],flush=True)
 return file,dict(sha256=digest,size=entry['size'],publisher_lfs_sha256=exp)
if __name__=='__main__':
 entries=requests.get('https://huggingface.co/api/models/'+REPO+'/revision/'+REV,params={'blobs':'true'},timeout=30).json()
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as e:
  files=dict(e.map(fetch,[file for file in entries['siblings'] if file['rfilename'] in FILES]))
 (ROOT/'source.json').write_text(json.dumps(dict(repo=REPO,revision=REV,files=files),indent=2))
