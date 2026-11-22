"""Portable source-FP32 CUDA extractor. No network use after fetch.py has downloaded the pinned publisher files."""
import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
import torch
from transformers.modeling_utils import no_init_weights
from fetch import ROOT,MODEL,sha
from require_t4 import require_t4
def main(splits):
 print('Verified T4 from nvidia-smi',require_t4(),flush=True)
 if not torch.cuda.is_available():raise RuntimeError('A CUDA runtime is required')
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 source=json.loads((ROOT/'source.json').read_text());cfg=json.loads((MODEL/'decision_config.json').read_text())
 spec=importlib.util.spec_from_file_location('publisher_encoder',MODEL/'glide/encoder.py');enc=importlib.util.module_from_spec(spec);spec.loader.exec_module(enc)
 with no_init_weights():model,tok=enc.load(str(MODEL),cfg,torch.device('cuda'),torch.float32)
 model.model.config.reference_compile=False;model.eval();data=json.loads((ROOT/'sampled-texts.json').read_text());policy=json.loads((ROOT/'packed-plan.json').read_text())
 prefix=[tok.cls_id]+tok.enc.encode(policy['prefix'],add_special_tokens=False);anchor=tok.enc.encode(policy['anchor'],add_special_tokens=False)
 where=ROOT/'content-hidden';where.mkdir(exist_ok=True)
 hidd={}
 def hook(_,args):hidd['prior']=args[0]
 handle=model.model.layers[27].register_forward_pre_hook(hook)
 try:
  for split in splits:
   for cid,records in data[split].items():
    saved=where/(split+'-'+cid+'.npz');rec=saved.with_suffix('.json')
    if saved.exists() and rec.exists() and json.loads(rec.read_text())['file']==sha(saved):continue
    start=time.monotonic();groups=[];tokens=len(prefix)+1;block=[]
    for i,(text,_truth) in enumerate(records):
     txt='\n'+text;raw=tok.enc(txt,add_special_tokens=False,return_offsets_mapping=True);ids=raw['input_ids']
     member=[j for j,(s,e) in enumerate(raw['offset_mapping']) if any(not ch.isspace() for ch in txt[s:e])]
     if block and (len(block)==8 or tokens+len(ids)+len(anchor)>512):groups.append(block);block=[];tokens=len(prefix)+1
     if len(ids)>7999-len(prefix)-len(anchor)-1:raise ValueError('Implement preregistered split policy for new overlength population')
     block.append((i,ids,member));tokens+=len(ids)+len(anchor)
    if block:groups.append(block)
    ref=np.load(ROOT/'portable-cues'/(split+'-'+cid+'.npz'),allow_pickle=False);known=ref['hidden'];maxerr=0.;pooled=np.empty_like(known);footprint=[];wc=0
    tokenroot=where/'token-states'/split/cid;tokenroot.mkdir(exist_ok=True,parents=True)
    for bix,group in enumerate(groups):
     seq=list(prefix);cues=[];positions=[];tokens_actual=[];offset=[0];rows=[]
     for row,raw,selected in group:
      rows.append(row);positions.extend(len(seq)+k for k in selected);tokens_actual.extend(raw[k] for k in selected);offset.append(len(positions));seq.extend(raw);seq.extend(anchor);cues.append(len(seq)-1)
     seq.append(tok.sep_id);footprint.append((seq,cues,rows));arr=torch.tensor(seq,device='cuda')[None];mask=torch.ones_like(arr,dtype=torch.bool);pos=torch.arange(len(seq),device='cuda')[None]
     with torch.inference_mode():
      hidden=model.model(input_ids=arr,attention_mask=mask,position_ids=pos).last_hidden_state[0]
      cue=hidden[cues].float().cpu().numpy();word=hidden[positions].float().cpu().numpy();prior=hidd['prior'][0].float().cpu().numpy()
     maxerr=max(maxerr,float(abs(cue-known[rows]).max()));wc+=len(positions)
     if maxerr>.002:raise RuntimeError(('CUDA output deviated from pinned FP32 cue reference',cid,bix,maxerr))
     for jj,row in enumerate(rows):pooled[row]=word[offset[jj]:offset[jj+1]].mean(0)
     with (tokenroot/f'{bix:04d}.npz').open('wb') as f:np.savez(f,rows=rows,offsets=offset,token_ids=tokens_actual,hidden=word,input_ids=np.array(seq,np.int32),content_positions=positions,cue_positions=cues,before_terminal_residual=prior)
    digest=hashlib.sha256(json.dumps(footprint).encode()).hexdigest()
    assert digest==str(ref['layout_sha256'])
    with saved.open('wb') as f:np.savez(f,positions=np.arange(len(records)),content_mean=pooled,input_hash=digest)
    rec.write_text(json.dumps({'file':sha(saved),'plan':sha(ROOT/'content-plan.json'),'max_known_cue_abs_error':maxerr,'content_tokens':wc,'rows':len(records),'blocks':len(groups),'backend':'cuda-source-f32-'+torch.cuda.get_device_name(),'seconds':time.monotonic()-start},indent=2))
    print(split,cid,len(records),'max cue drift',maxerr,'seconds',round(time.monotonic()-start,1),flush=True)
 finally:handle.remove()
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('splits',nargs='+',choices=['train','val']);q=a.parse_args();main(q.splits)
