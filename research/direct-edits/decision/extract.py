"""Run the pinned publisher's audited prompt and encoder, saving penultimate-position hidden vectors."""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from fetch import ROOT,MODEL,sha
from cohort import LABELS,case_rows

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_grad_enabled(False)
def load_local(name):
 spec=importlib.util.spec_from_file_location('glide_'+name,MODEL/'glide'/(name+'.py'))
 module=importlib.util.module_from_spec(spec)
 spec.loader.exec_module(module)
 return module
prompt=load_local('prompt')
encoder=load_local('encoder')
QUESTION={'type':'choice','instructions':'Which rhetorical role does the target sentence play in this Indian court judgment?'}
OPTIONS=[('None','metadata, headings, or other material'),('Facts','case events and procedural background'),
         ('Issue','a legal question for the court'),("Petitioner argument","petitioner's submission"),
         ("Respondent argument","respondent's submission"),('Reasoning',"court’s analysis or rationale"),
         ('Decision','court’s holding or order')]
def modules(backend):
 config=json.loads((MODEL/'decision_config.json').read_text())
 device='cuda' if backend=='cuda' else 'cpu'
 if backend=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA was requested but is unavailable')
 native_dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if backend=='cuda' else (torch.bfloat16 if backend=='bf16' else torch.float32)
 from transformers.modeling_utils import no_init_weights
 with no_init_weights():
  model,tokens=encoder.load(str(MODEL),config,torch.device(device),native_dtype)
 model.model.config.reference_compile=False
 if backend=='int8':
  import warnings
  from torch.ao.quantization import default_dynamic_qconfig
  from torch.ao.nn.quantized.dynamic import Linear
  warnings.filterwarnings('ignore',category=DeprecationWarning)
  # Quantize one linear module at a time; replacing all at once doubles memory.
  for parent in model.model.modules():
   for name,layer in list(parent.named_children()):
    if isinstance(layer,torch.nn.Linear) and layer is not model.model.layers[-1].mlp.Wo:
     layer.qconfig=default_dynamic_qconfig
     setattr(parent,name,Linear.from_float(layer))
 return model,tokens,prompt.Prompter(tokens,config['codes'])
def states_for(doc,order,arm):
 pack=[]
 for i in order:
  text=doc[i][0]
  if arm=='local':
   prior=doc[i-1][0] if i>0 else ''
   after=doc[i+1][0] if i+1<len(doc) else ''
   text='Previous: '+prior+'\nTarget: '+text+'\nNext: '+after
  elif arm!='alone': raise ValueError(arm)
  pack.append(text)
 return pack
def encode(model,tokens,template,text):
 # The actual head is invoked, with its input captured by pre-hook so that replay
 # and edits can be checked against the publisher's module and the same sentence.
 rendered=template.text(text,QUESTION,OPTIONS)
 ids=tokens([rendered])['input_ids'][0]
 x=torch.tensor([ids],dtype=torch.long,device=model.readout.weight.device)
 captured={}
 def capture(key,position=None):
  def hook(_,args):
   value=args[0][0] if position is None else args[0][0,position]
   captured[key]=value.detach().float().cpu().numpy().copy()
  return hook
 hooks=[model.readout.register_forward_pre_hook(capture('hidden')),
        model.model.final_norm.register_forward_pre_hook(capture('before_normalization',-2)),
        model.model.layers[-1].mlp.Wo.register_forward_pre_hook(capture('mlp_activation',-2))]
 try:
  with torch.inference_mode(): logits=model(x,torch.ones_like(x,dtype=torch.bool)).float().cpu().numpy()[0].copy()
 finally:
  for handle in hooks: handle.remove()
 return captured['hidden'],logits[:7],len(ids),captured['before_normalization'],captured['mlp_activation']
def smoke(backend):
 start=time.perf_counter()
 model,tokens,temp=modules(backend)
 print('loaded',backend,'seconds',round(time.perf_counter()-start,2),flush=True)
 samples=['Accordingly, the appeal is dismissed.','The question for determination is whether this application is barred by limitation.']
 rows=[]
 for text in samples:
  start=time.perf_counter()
  h,z,n,r,u=encode(model,tokens,temp,text)
  rows.append(dict(text=text,option_logits=z.tolist(),tokens=n,seconds=time.perf_counter()-start))
  print(backend,'tokens',n,'seconds',rows[-1]['seconds'],'label',int(z.argmax()),flush=True)
 np.savez(ROOT/('smoke-'+backend+'.npz'),**{str(i):row for i,row in enumerate([h,z])})
 (ROOT/('smoke-'+backend+'.json')).write_text(json.dumps(rows,indent=2))
def extract(backend,arm,subsets,ready=None,inference=None):
 plan=json.loads((ROOT/'plan.json').read_text())
 source=json.loads((ROOT/'source.json').read_text())
 model,tokenizer,temp=modules(backend) if ready is None else ready
 cache=ROOT/'features'/backend/arm;cache.mkdir(exist_ok=True,parents=True)
 for split,tier in [arg.split('-') for arg in subsets]:
  policy=plan[tier]
  gids=plan['fit_judgment_ids'][:policy['train_cases']] if split=='train' else plan['val_judgment_ids']
  docs=case_rows(split,set(gids))
  print('encoding',backend,arm,split,tier,'judgments',len(gids),flush=True)
  started=time.monotonic();done=0
  for gid in gids:
   doc=docs[gid]
   ids=list(range(len(doc))) if split=='train' else plan['validation_indices'][gid][:policy['validation_rows_per_shared_judgment']]
   texts=states_for(doc,ids,arm)
   dest=cache/f'{split}-{gid}.npz'
   keyhash=hashlib.sha256(json.dumps([source['revision'],sha(ROOT/'plan.json'),arm,backend,QUESTION,OPTIONS],ensure_ascii=False).encode()).hexdigest()
   saved={}
   if dest.exists():
    with np.load(dest,allow_pickle=False) as blob:
     assert str(blob['recipe'])==keyhash
     saved={int(i):(h,z,int(n),str(d),r,u) for i,h,z,n,d,r,u in zip(blob['idx'],blob['hidden'],blob['logit'],blob['ntok'],blob['text_sha256'],blob['before_normalization'],blob['mlp_activation'])}
   def commit():
    indexes=sorted(saved)
    tmp=dest.with_suffix('.tmp')
    with tmp.open('wb') as f:
     np.savez(f,recipe=keyhash,idx=indexes,hidden=np.stack([saved[i][0] for i in indexes]),logit=np.stack([saved[i][1] for i in indexes]),ntok=[saved[i][2] for i in indexes],text_sha256=[saved[i][3] for i in indexes],before_normalization=np.stack([saved[i][4] for i in indexes]),mlp_activation=np.stack([saved[i][5] for i in indexes]))
    tmp.replace(dest)
   for pos,text in zip(ids,texts):
    h=hashlib.sha256(text.encode()).hexdigest()
    if pos in saved:
     assert saved[pos][3]==h
     continue
    vec,z,n,r,u=(inference or encode)(model,tokenizer,temp,text)
    saved[pos]=(vec,z,n,h,r,u)
    done+=1
    if done%4==0: commit()
   commit()
   print(arm,split,gid,'rows',len(ids),'new',done,'seconds',round(time.monotonic()-started,1),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser()
 ap.add_argument('mode',choices=['smoke','extract']);ap.add_argument('--backend',choices=['fp32','int8','bf16','cuda'],default='fp32')
 ap.add_argument('--arm',choices=['alone','local'],default='alone');ap.add_argument('subsets',nargs='*')
 args=ap.parse_args()
 if args.mode=='smoke': smoke(args.backend)
 else: extract(args.backend,args.arm,args.subsets)
