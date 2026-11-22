"""Validate the GLiDE export before using its live CPU/GPU readouts on the legal cohort."""
import argparse
import json
import time
import numpy as np
from transformers import AutoTokenizer
from extract import encoder,prompt,QUESTION,OPTIONS,extract
from fetch import ROOT,MODEL,sha
from ov_graph import OV,compile_graph,inputs
from weights import source,arr,freeze,score,savej
def ready(device,hint,compiler=compile_graph):
 cfg=json.loads((MODEL/'decision_config.json').read_text())
 enc=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 render=AutoTokenizer.from_pretrained(MODEL/'render_tokenizer',local_files_only=True)
 tokens=encoder.EncoderTokenizer(render,enc,cfg['max_len'],cfg['trunc_head'])
 temp=prompt.Prompter(tokens,cfg['codes'])
 return (compiler(device,hint),arr(source()[0])[:7]),tokens,temp
def encode(ctx,tokens,template,text):
 comp,W=ctx
 prompt_text=template.text(text,QUESTION,OPTIONS)
 ids=tokens([prompt_text])['input_ids'][0]
 n=len(ids)
 width=min(7999,32*((n+31)//32))
 outputs=comp(inputs(ids,width=width))
 h=np.array(outputs['hidden'][0,0],np.float32)
 r=np.array(outputs['before_normalization'][0,0],np.float32)
 u=np.array(outputs['mlp_activation'][0,0],np.float32)
 return h,h@W.T,n,r,u
def verify(r,device,hint):
 items=json.loads((ROOT/'smoke-fp32.json').read_text())
 proof=[]
 for item in items:
  t=time.monotonic();h,z,n,_,_=encode(*r,item['text'])
  delta=abs(z-np.asarray(item['option_logits']))
  proof.append(dict(tokens=n,seconds=time.monotonic()-t,max_absolute_logit_difference=float(delta.max()),mean_absolute_logit_difference=float(delta.mean()),
               exporter_top_option=int(z.argmax()),torch_top_option=int(np.argmax(item['option_logits']))))
  print('parity',proof[-1],flush=True)
 native_h=np.load(ROOT/'smoke-fp32.npz')['0'];cos=float(h@native_h/(np.linalg.norm(h)*np.linalg.norm(native_h)))
 accepted=all(row['exporter_top_option']==row['torch_top_option'] and row['max_absolute_logit_difference']<.15 for row in proof) and cos>.999
 out=dict(precision=hint,device=device,torch_source_dtype='pinned BF16 weights promoted to FP32 for reference arithmetic',native_source=sha(ROOT/'smoke-fp32.json'),
          rows=proof,last_probe_hidden_cosine=cos,early_acceptance=accepted)
 savej(OV/(f'parity-{device}-{hint}.json'),out)
 if not accepted: raise ValueError(out)
 return out
if __name__=='__main__':
 cli=argparse.ArgumentParser();cli.add_argument('mode',choices=['verify','run']);cli.add_argument('--device',choices=['CPU','GPU'],default='GPU');cli.add_argument('--hint',choices=['f32','f16'],default='f32');cli.add_argument('--tier',choices=['pilot','extended'],default='pilot');cli.add_argument('--arm',choices=['alone','local','both'],default='alone')
 args=cli.parse_args()
 r=ready(args.device,args.hint);verify(r,args.device,args.hint)
 if args.mode=='run':
  key='ov-'+args.device+'-'+args.hint
  for arm in ('alone','local') if args.arm=='both' else (args.arm,):
   extract(key,arm,['train-'+args.tier],r,inference=encode)
   freeze(key,arm,args.tier)
   extract(key,arm,['val-'+args.tier],r,inference=encode)
   score(key,arm,args.tier)
   from report import create
   create()
