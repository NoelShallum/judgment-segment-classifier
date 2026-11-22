"""Locate which layer first departs from the HF reference in the hardware export."""
import argparse
import os,time
from pathlib import Path
import numpy as np
import openvino as ov
import torch
from safetensors import safe_open
from transformers import ModernBertConfig
from transformers.modeling_utils import no_init_weights
from transformers.models.modernbert.modeling_modernbert import ModernBertEmbeddings,ModernBertEncoderLayer
from ov_graph import OV,inputs,op,MODEL
from fetch import ROOT
torch.set_num_threads(1)
def compare(device,precision):
 n=30;width=32
 ids=np.asarray([50281]+[2053,420,139,317,9403,122,52]*4+[50282],np.int32)
 pack=inputs(ids,width=width)
 cfg=ModernBertConfig.from_json_file(str(MODEL/'config.json'));cfg._attn_implementation='sdpa';cfg.reference_compile=False
 ref={}
 with torch.inference_mode(),safe_open(str(MODEL/'model-00001-of-00001.safetensors'),'pt') as f:
  def load(unit,pre):
   unit.load_state_dict({name:f.get_tensor(pre+name).float() for name,_ in unit.named_parameters()})
   return unit.eval()
  with no_init_weights(): e=ModernBertEmbeddings(cfg)
  z=load(e,'model.embeddings.')(torch.tensor(ids.astype(np.int64))[None])
  ref['embed']=z[0].numpy().copy()
  pos=torch.arange(n)[None];glob=torch.zeros(1,1,n,n);local=glob.masked_fill(abs(pos-pos.T)>64,-3e38)
  for i in range(3):
   with no_init_weights(): layer=ModernBertEncoderLayer(cfg,i)
   layer=load(layer,'model.layers.'+str(i)+'.')
   qkv=layer.attn.Wqkv(layer.attn_norm(z))
   att=layer.attn(layer.attn_norm(z),attention_mask=glob,sliding_window_mask=local,position_ids=pos)[0]
   ref[f'qkv{i}']=qkv[0].numpy().copy()
   mid=z+att;ref[f'att{i}']=mid[0].numpy().copy()
   h1,h2=layer.mlp.Wi(layer.mlp_norm(mid)).chunk(2,dim=-1)
   m=layer.mlp.act(h1)*h2;ref[f'gate{i}']=m[0].numpy().copy()
   z=layer(z,attention_mask=glob,sliding_window_mask=local,position_ids=pos)[0]
   ref[f'end{i}']=z[0].numpy().copy()
 del e,layer
 core=ov.Core();whole=core.read_model(str(OV/'encoder.xml'))
 mat={}
 for node in whole.get_ordered_ops():
  if node.type_info.name=='MatMul':
   original=node.input_value(1).get_node()
   while original.type_info.name=='Convert': original=original.input_value(0).get_node()
   mat[original.get_friendly_name()]=node
 def plus(node):
  return next(port.get_node() for port in node.output(0).get_target_inputs() if port.get_node().type_info.name=='Add')
 results={}
 for i in range(3):
  base=f'model_layers_{i}_'
  qkv,attention,wo=(mat[base+k+'_weight_f16'] for k in ('attn_Wqkv','attn_Wo','mlp_Wo'))
  if not i:results['embed']=qkv.input_value(0)
  results[f'qkv{i}']=qkv.output(0);results[f'att{i}']=plus(attention).output(0);results[f'gate{i}']=wo.input_value(0);results[f'end{i}']=plus(wo).output(0)
 params=[x for x in whole.get_parameters() if x.get_friendly_name()!='cue']
 graph=ov.Model(list(results.values()),params)
 opts={'INFERENCE_PRECISION_HINT':precision,'PERFORMANCE_HINT':'LATENCY','NUM_STREAMS':'1'}
 if device=='CPU':opts['INFERENCE_NUM_THREADS']='1'
 comp=core.compile_model(graph,device,opts)
 del pack['cue']
 out=comp(pack)
 for i,(name,target) in enumerate(ref.items()):
  found=np.asarray(out[i][0,:len(ids)]).ravel();target=target.ravel()
  print(name,'cos',float(found@target/(np.linalg.norm(found)*np.linalg.norm(target))),'relative norm',float(np.linalg.norm(found-target)/np.linalg.norm(target)), 'max',float(abs(found-target).max()),flush=True)
if __name__=='__main__':
 q=argparse.ArgumentParser();q.add_argument('--device',default='CPU');q.add_argument('--precision',default='f32');a=q.parse_args();compare(a.device,a.precision)
