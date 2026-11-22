"""Run raw text through an independently compiled encoder containing the physically edited terminal projection."""
import argparse
import json
import numpy as np
import openvino as ov
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoTokenizer
from cohort import case_rows
from extract import encoder,prompt,states_for,QUESTION,OPTIONS
from fetch import ROOT,MODEL,sha
from ov_graph import OV,op,inputs
from weights import INTERNAL,READOUT,arr,source,normed,savej
def verify(arm,tier,backend):
 wd=ROOT/'weights'/backend/arm/tier
 phys=json.loads((wd/'physical-checkpoint.json').read_text());whole=sha(wd/'roles-internal.safetensors')
 assert whole==phys['patch_sha256']
 modeldir=wd/'published-encoder-copy'
 with safe_open(str(modeldir/'model-00001-of-00001.safetensors'),'pt',device='cpu') as f:new=arr(f.get_tensor(INTERNAL))
 original,g,old=source();native_w=arr(original)[:7];delta=new-arr(old);g=arr(g)
 # Replace the actual Constant feeding the terminal matmul, after reading the
 # edited matrix from the complete HF checkpoint; every other graph weight is original.
 core=ov.Core();graph=core.read_model(str(OV/'encoder.xml'));name='model_layers_27_mlp_Wo_weight_f16'
 links=[n for n in graph.get_ops() if n.get_friendly_name()==name];assert len(links)==1
 node=links[0];replacement=op.constant(new.astype(np.float16),name='edited_final_mlp_weight');dest=list(node.output(0).get_target_inputs())
 assert len(dest)>0
 for port in dest:port.replace_source_output(replacement.output(0))
 graph.validate_nodes_and_infer_types()
 config={'INFERENCE_PRECISION_HINT':'f32','PERFORMANCE_HINT':'LATENCY','NUM_STREAMS':'1','CACHE_DIR':str(OV/'cache')}
 comp=core.compile_model(graph,'GPU',config)
 spec=json.loads((MODEL/'decision_config.json').read_text());t=encoder.EncoderTokenizer(AutoTokenizer.from_pretrained(MODEL/'render_tokenizer',local_files_only=True),AutoTokenizer.from_pretrained(MODEL,local_files_only=True),7999,1024);renderer=prompt.Prompter(t,spec['codes'])
 plan=json.loads((ROOT/'plan.json').read_text());ids=plan['val_judgment_ids'][:2];docs=case_rows('val',set(ids));proof=[]
 for gid in ids:
  row=plan['validation_indices'][gid][0]
  if arm.startswith('packed-'):
   from packed import grouping
   tok,markers,members=next((seq,cues,idx) for seq,cues,idx in grouping(docs[gid],t,arm) if row in idx)
   locations=[markers[members.index(row)]]
  else:
   st=states_for(docs[gid],[row],arm)[0];tok=t([renderer.text(st,QUESTION,OPTIONS)])['input_ids'][0]
   locations=[len(tok)-2]
  features=np.load(ROOT/'features'/backend/arm/f'val-{gid}.npz',allow_pickle=False);i=features['idx'].tolist().index(row)
  r,u=features['before_normalization'][i],features['mlp_activation'][i]
  expected=(normed((r+u@delta.T)[None],g)@native_w.T)[0]
  out=comp(inputs(tok,cues=locations,width=min(7999,32*((len(tok)+31)//32))))
  actual=out['hidden'][0,0].astype('float32')@native_w.T
  difference=float(np.max(abs(expected-actual)))
  proof.append(dict(case=gid,position=row,prediction_class=int(actual.argmax()),maximum_abs_logit_difference=difference,
                    agrees_with_frozen_offline_prediction=bool(actual.argmax()==expected.argmax())))
  print('PHYSICAL F32',proof[-1],flush=True)
  assert difference<.01 and np.argmax(actual)==np.argmax(expected)
 savej(wd/'ov-runtime-proof.json',dict(source='complete physically patched native HF checkpoint',changed_tensor=INTERNAL,gpu_precision='f32',checks=proof))
if __name__=='__main__':
 q=argparse.ArgumentParser();q.add_argument('--arm',default='alone');q.add_argument('--tier',default='pilot');q.add_argument('--backend',default='ov-GPU-f32');a=q.parse_args();verify(a.arm,a.tier,a.backend)
