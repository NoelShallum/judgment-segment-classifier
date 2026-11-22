"""Gather ordered-cue and genuine content-token states in the same pinned encoder pass. Start only in an assigned GPU window."""
import json
import argparse
import fcntl
import hashlib
import time
import numpy as np
from tokenizers import Tokenizer
from cohort import case_rows
from fetch import ROOT,MODEL,sha
from ov_worker import ready,verify
from ov_graph import inputs,OV,op
import openvino as ov
from packed import grouping,ANCHOR
from readout_study import raw_mask,run
ROOTOUT=ROOT/'content-hidden'
def compile_with_terminal_input(device,hint):
 core=ov.Core();graph=core.read_model(str(OV/'encoder.xml'))
 mat=None
 for n in graph.get_ordered_ops():
  if n.type_info.name!='MatMul':continue
  w=n.input_value(1).get_node()
  while w.type_info.name=='Convert':w=w.input_value(0).get_node()
  if w.get_friendly_name()=='model_layers_27_attn_Wqkv_weight_f16':mat=n;break
 if mat is None:raise ValueError('Terminal attention projection missing')
 mul=mat.input_value(0).get_node();mvn=[q.get_node() for q in mul.input_values() if q.get_node().type_info.name=='MVN']
 if len(mvn)!=1:raise ValueError('Terminal pre-attention normalization unavailable')
 before=op.add(mvn[0].input_value(0),np.float32(0));before.output(0).get_tensor().set_names({'before_terminal_layer'});graph.add_results([op.result(before)])
 graph.validate_nodes_and_infer_types()
 conf={'INFERENCE_PRECISION_HINT':hint,'PERFORMANCE_HINT':'LATENCY','NUM_STREAMS':'1','CACHE_DIR':str(OV/'cache')}
 if device=='CPU':conf['INFERENCE_NUM_THREADS']='1'
 return core.compile_model(graph,device,conf)
def emit(ready_model,split,device,limit=None):
 (comp,_),tok,_=ready_model;native=Tokenizer.from_file(str(MODEL/'tokenizer.json'));plan=json.loads((ROOT/'packed-full-plan.json').read_text())
 cids=plan['fixed_fit_ids'] if split=='train' else plan['fixed_validation_ids']
 cids=cids if limit is None else cids[:limit]
 docs=case_rows(split,set(cids))
 root=ROOT/'features/ov-GPU-f32/packed-ordered'
 mark=tok.enc.encode(ANCHOR,add_special_tokens=False)
 ROOTOUT.mkdir(exist_ok=True)
 for cid in cids:
  fname=ROOTOUT/(split+'-'+cid+'.npz')
  receipt=fname.with_suffix('.json')
  locks=ROOTOUT/'.locks';locks.mkdir(exist_ok=True)
  lock=(locks/(split+'-'+cid)).open('a')
  try:
   fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
   if fname.exists() and receipt.exists():
    prior=json.loads(receipt.read_text())
    if prior['file']==sha(fname) and prior['plan']==sha(ROOT/'content-plan.json'):continue
    raise ValueError('Existing content-token cache does not match tag or hash')
   started=time.monotonic();inference_time=0.
   records=docs[cid];blocks=list(grouping(records,tok,'packed-ordered'))
   digest=hashlib.sha256(json.dumps(blocks).encode()).hexdigest();layout=json.loads((root/'layouts'/(split+'-'+cid+'.json')).read_text());assert digest==layout['sha256_input_ids_cues_targets']
   ref=np.load(root/(split+'-'+cid+'.npz'),allow_pickle=False);base={int(i):z for i,z in zip(ref['idx'],ref['hidden'])}
   pooled=np.zeros((len(records),1024),np.float32);maxerr=0.;count=0;store=ROOTOUT/'token-states'/split/cid;store.mkdir(parents=True,exist_ok=True)
   for bix,(ids,cues,members) in enumerate(blocks):
    gather=list(cues);ranges=[];token_ids=[];offsets=[0]
    for cue,row in zip(cues,members):
     raw,real=raw_mask(native,records[row][0]);end=cue-len(mark)+1;start=end-len(raw)
     assert np.array_equal(ids[start:end],raw) and ids[end:cue+1]==mark and start>=1
     loc=np.arange(start,end)[real]
     assert len(loc)>0 and loc.max()<end<=cue
     begin=len(gather);gather.extend(loc.tolist());ranges.append((begin,len(gather)));token_ids.extend(raw[real].tolist());offsets.append(len(token_ids))
    width=min(7999,32*((len(ids)+31)//32));inference_start=time.monotonic();outputs=comp(inputs(ids,cues=gather,width=width));inference_time+=time.monotonic()-inference_start
    hid=np.asarray(outputs['hidden'][0],np.float32)
    for j,(row,(a,b)) in enumerate(zip(members,ranges)):
     maxerr=max(maxerr,float(np.abs(hid[j]-base[row]).max()));pooled[row]=hid[a:b].mean(0)
    if maxerr>1.e-3:raise ValueError(('changed reference cue',cid,bix,maxerr))
    # Store canonical full-precision token embeddings for later nonuniform pooling without any repeated model pass.
    target=store/f'{bix:04d}.npz';stage=target.with_suffix('.part')
    with stage.open('wb') as f:
     np.savez(f,rows=members,offsets=offsets,token_ids=token_ids,hidden=hid[len(members):],
              input_ids=np.asarray(ids,np.int32),content_positions=np.asarray(gather[len(members):],np.int32),
              cue_positions=np.asarray(cues,np.int32),before_terminal_residual=np.asarray(outputs['before_terminal_layer'][0,:len(ids)],np.float32))
    stage.replace(target)
    count+=len(token_ids)
   stage=fname.with_suffix('.part')
   with stage.open('wb') as f:np.savez(f,positions=np.arange(len(records)),content_mean=pooled,input_hash=digest)
   stage.replace(fname);receipt.write_text(json.dumps({'file':sha(fname),'plan':sha(ROOT/'content-plan.json'),'max_known_cue_abs_error':maxerr,'content_tokens':count,'rows':len(records),'blocks':len(blocks),'backend':'openvino-f32-'+device,'total_case_seconds':time.monotonic()-started,'inference_seconds':inference_time,'full_precision_token_state_folder':str(store.resolve())},indent=2))
   print(split,'case',cid,'rows',len(records),'blocks',len(blocks),'word tokens',count,'max cue drift',maxerr,flush=True)
  finally:lock.close()
if __name__=='__main__':
 cli=argparse.ArgumentParser();cli.add_argument('--device',choices=['CPU','GPU'],default='CPU');cli.add_argument('--limit-cases',type=int);cli.add_argument('--extract-only',action='store_true');cli.add_argument('splits',nargs='+',choices=['train','val'])
 a=cli.parse_args()
 if a.device=='GPU':raise RuntimeError('Subsequent GPU experiments require an nvidia-smi-verified T4; use colab_content.py on the assigned T4')
 r=ready(a.device,'f32',compiler=compile_with_terminal_input);verify(r,a.device,'f32')
 for split in a.splits:emit(r,split,a.device,a.limit_cases)
 if not a.extract_only:run(True)
