"""Export the pinned GLiDE ModernBERT graph with the same rotary, 64-neighbor and global attention masks as its publisher runtime."""
import argparse
import json
import math
import struct
import time
from pathlib import Path
import numpy as np
import openvino as ov
from openvino.runtime import opset13 as op
from fetch import ROOT,MODEL,sha
OV=ROOT/'openvino'; SOURCE=MODEL/'model-00001-of-00001.safetensors'
def export():
 OV.mkdir(exist_ok=True)
 source=json.loads((ROOT/'source.json').read_text())
 assert sha(SOURCE)==source['files'][SOURCE.name]['sha256']
 with SOURCE.open('rb') as f:
  n=struct.unpack('<Q',f.read(8))[0]
  header=json.loads(f.read(n))
 start=8+n
 used=set();errors=[]
 cfg=json.loads((MODEL/'config.json').read_text())
 assert (cfg['hidden_size'],cfg['num_hidden_layers'],cfg['num_attention_heads'],cfg['local_attention'],cfg['global_attn_every_n_layers'])==(1024,28,16,128,3)
 assert cfg['global_rope_theta']==cfg['local_rope_theta']==160000.0 and not cfg['norm_bias']
 def get(name):
  entry=header[name]
  assert entry['dtype']=='BF16' and name not in used
  used.add(name)
  raw=np.memmap(SOURCE,dtype='<u2',mode='r',offset=start+entry['data_offsets'][0],shape=tuple(entry['shape']))
  x=(raw.astype('uint32')<<16).view('float32')
  half=x.astype(np.float16)
  errors.append(float(np.max(np.abs(x-half))))
  return op.convert(op.constant(half,name=name.replace('.','_')+'_f16'),'f32')
 axis=-1
 def norm(z,n):
  return op.multiply(op.mvn(z,op.constant(np.array([-1],np.int64)),True,cfg['norm_eps'],'inside_sqrt'),get(n))
 def linear(z,n): return op.matmul(z,get(n),False,True)
 def parts(z,n):
  node=op.split(z,np.array(axis,np.int64),n)
  return [node.output(i) for i in range(n)]
 def heads(z):
  return op.transpose(op.reshape(z,np.array([0,0,16,64],np.int64),True),np.array([0,2,1,3],np.int64))
 def as_vector(z):
  return op.reshape(op.transpose(z,np.array([0,2,1,3],np.int64)),np.array([0,0,1024],np.int64),True)
 def rotary(z,c,s):
  a,b=parts(z,2)
  rotated=op.concat([op.negative(b),a],axis=-1)
  return op.add(op.multiply(z,c),op.multiply(rotated,s))
 ids=op.parameter([1,-1],np.int32,'input_ids')
 cosine=op.parameter([1,1,-1,64],np.float32,'cosine')
 sine=op.parameter([1,1,-1,64],np.float32,'sine')
 mask=op.parameter([1,1,1,-1],np.float32,'global_mask')
 local=op.parameter([1,1,-1,-1],np.float32,'local_mask')
 chosen=op.parameter([-1],np.int32,'cue')
 z=norm(op.gather(get('model.embeddings.tok_embeddings.weight'),ids,np.array(0,np.int64)),'model.embeddings.norm.weight')
 for i in range(28):
  prefix=f'model.layers.{i}.'
  n=z if i==0 else norm(z,prefix+'attn_norm.weight')
  q,k,v=(heads(a) for a in parts(linear(n,prefix+'attn.Wqkv.weight'),3))
  q=rotary(q,cosine,sine);k=rotary(k,cosine,sine)
  attn=op.scaled_dot_product_attention(q,k,v,mask if i%3==0 else local,causal=False)
  z=op.add(z,linear(as_vector(attn),prefix+'attn.Wo.weight'))
  n=norm(z,prefix+'mlp_norm.weight')
  value,gate=parts(linear(n,prefix+'mlp.Wi.weight'),2)
  hidden=op.multiply(op.gelu(value,'ERF'),gate)
  z=op.add(z,linear(hidden,prefix+'mlp.Wo.weight'))
 before=z;h=norm(z,'model.final_norm.weight')
 assert set(header)-used<= {'__metadata__'},(set(header)-used)
 results=[]
 for name,v in [('hidden',h),('before_normalization',before),('mlp_activation',hidden)]:
  node=op.gather(v,chosen,np.array(1,np.int64))
  node.output(0).get_tensor().set_names({name})
  results.append(node)
 params=[ids,cosine,sine,mask,local,chosen]
 names=['input_ids','cosine','sine','global_mask','local_mask','cue']
 for v,name in zip(params,names): v.output(0).get_tensor().set_names({name})
 model=ov.Model(results,params,'fastino_glide_0p4b')
 ov.serialize(model,str(OV/'encoder.xml'),str(OV/'encoder.bin'))
 meta={'source':source['files'][SOURCE.name]['sha256'],'revision':source['revision'],'float16_weight_storage_max_abs_error':max(errors),
       'openvino':ov.__version__,'graph_code_sha256':sha(Path(__file__)),'xml':sha(OV/'encoder.xml'),'bin':sha(OV/'encoder.bin'),'source_tensors':len(used)}
 (OV/'source.json').write_text(json.dumps(meta,indent=2))
 print('exported',meta,flush=True)
def inputs(ids,cues=None,width=None):
 n=len(ids);width=width or n
 assert n<=width
 ext=np.zeros(width,np.int32);ext[:n]=ids
 pos=np.arange(width,dtype=np.float32)
 inv=1.0/(160000.0**(np.arange(0,64,2,dtype=np.float32)/64))
 angle=pos[:,None]*inv[None,:];omega=np.concatenate((angle,angle),axis=1)
 good=(np.arange(width)<n)
 local=(np.abs(pos[:,None]-pos[None,:])<=64)&good[None,:]
 return dict(input_ids=ext[None,:],cosine=np.cos(omega)[None,None],sine=np.sin(omega)[None,None],
             global_mask=np.where(good,0.,-10000.).astype('float32')[None,None,None,:],
             local_mask=np.where(local,0.,-10000.).astype('float32')[None,None],
             cue=np.asarray(cues if cues is not None else [n-2],np.int32))
def compile_graph(device,hint):
 root=OV/'cache';root.mkdir(exist_ok=True)
 core=ov.Core()
 cfg={'PERFORMANCE_HINT':'LATENCY','INFERENCE_PRECISION_HINT':hint,'NUM_STREAMS':'1','CACHE_DIR':str(root)}
 if device=='CPU':cfg['INFERENCE_NUM_THREADS']='1'
 print('compiling',core.get_property(device,'FULL_DEVICE_NAME'),hint,flush=True)
 return core.compile_model(str(OV/'encoder.xml'),device,cfg)
if __name__=='__main__':export()
