"""Diagnose and directly edit native terminal-layer cue queries from saved pre-attention residuals."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file,save_file
from scipy.special import erf,softmax
from fetch import ROOT,MODEL,sha
from weights import wtproof,savej
D=1024;H=16;S=64
PREFIX='model.layers.27.'
FUSED=PREFIX+'attn.Wqkv.weight'
DIR=ROOT/'attention-write';DIR.mkdir(exist_ok=True)
CONTENT=Path(os.environ.get("ATTOFLOW_GLIDE_CONTENT", str(ROOT/"content-hidden")))
def params():
 codes=ROOT/'weights/ov-GPU-f32/packed-ordered/whole16/roles-readout.safetensors'
 W=load_file(str(codes))['weight'][:7].float().numpy()
 with safe_open(str(MODEL/'model-00001-of-00001.safetensors'),'pt',device='cpu') as f:
  return {name:f.get_tensor(key).float().numpy() for name,key in [('qkv',FUSED),('att_norm',PREFIX+'attn_norm.weight'),('att_wo',PREFIX+'attn.Wo.weight'),('mlp_norm',PREFIX+'mlp_norm.weight'),('wi',PREFIX+'mlp.Wi.weight'),('wo',PREFIX+'mlp.Wo.weight'),('final','model.final_norm.weight')]}|{'readout':W}
def normal(x,g):
 centered=x-x.mean(-1,keepdims=True)
 return centered/np.sqrt((centered*centered).mean(-1,keepdims=True)+1.e-5)*g
def rotary(x,pos,inverse=False):
 omega=np.outer(np.asarray(pos,np.float32),1/(160000.**(np.arange(0,S,2,dtype=np.float32)/S)))
 omega=np.concatenate((omega,omega),axis=1)[:,None,:]
 cos=np.cos(omega);sin=np.sin(omega)
 if inverse:sin=-sin
 a,b=np.split(x,2,axis=-1)
 return x*cos+np.concatenate((-b,a),axis=-1)*sin
def block_arrays(z,cue,positions,offsets,w):
 norm=normal(z,w['att_norm'])
 raw=norm@w['qkv'][D:].T;k,v=np.split(raw,2,axis=1);k=rotary(k.reshape(-1,H,S),np.arange(len(z)));v=v.reshape(-1,H,S)
 x=norm[cue];qo=(x@w['qkv'][:D].T).reshape(-1,H,S)
 owner=np.full(len(z),-1,np.int32)
 for i,(a,b) in enumerate(zip(offsets,offsets[1:])):owner[positions[a:b]]=i
 return dict(x=x,qo=qo,k=k,v=v,owner=owner,pos=cue,base=z[cue],noncue=norm)
def replay(b,w,qdelta=None):
 q=b['qo'] if qdelta is None else b['qo']+(b['x']@qdelta.T).reshape(-1,H,S)
 q=rotary(q,b['pos'])
 # left operand [head,cue,dim], right operand [head,dim,token]
 logit=np.matmul(q.transpose(1,0,2),b['k'].transpose(1,2,0)).transpose(1,0,2)/np.float32(8.)
 p=softmax(logit,axis=-1)
 v=np.matmul(p.transpose(1,0,2),b['v'].transpose(1,0,2)).transpose(1,0,2).reshape(-1,D)
 mid=b['base']+v@w['att_wo'].T
 wi=normal(mid,w['mlp_norm'])@w['wi'].T
 a,g=np.split(wi,2,axis=1)
 h=normal(mid+(a*.5*(1.+erf(a/np.sqrt(np.float32(2.))))*g)@w['wo'].T,w['final'])
 return h,p,logit,q
def diagnostics(b,p):
 rows=[]
 for i in range(len(p)):
  own=b['owner']==i;other=(b['owner']>=0)&(~own)
  if not other.any():continue
  # np.take avoids transposing the two head/position axes under boolean indexing.
  o=p[i][:,own].sum(1);n=p[i][:,other].sum(1)
  rows.append([float(o.mean()),float(n.mean()),float(np.mean(np.log((o/own.sum()+1e-10)/(n/other.sum()+1e-10)))),float((o/own.sum()<=n/other.sum()).mean())])
 return rows
def add_slot_diagnostics(store,values):
 for i,record in enumerate(values):
  key='first' if i==0 else ('last' if i==len(values)-1 else 'interior')
  store.setdefault(key,[]).append(record)
def summarize_slots(store):
 return {key:{'cues':len(records),'average':np.mean(records,axis=0).tolist()} for key,records in store.items()}
def own_minus_other_logits(b,logits):
 result=[]
 for i in range(len(logits)):
  own=b['owner']==i;other=(b['owner']>=0)&(~own)
  if other.any():result.append((logits[i][:,own].mean(1)-logits[i][:,other].mean(1)).mean())
 return result
def read_file(fn,w):
 with np.load(fn,allow_pickle=False) as d:
  rows=d['rows'].copy()
  b=block_arrays(d['before_terminal_residual'],d['cue_positions'],d['content_positions'],d['offsets'],w)
 return rows,b
def block_files(split):
 p=json.loads((ROOT/'packed-full-plan.json').read_text())
 cids=p['fixed_fit_ids'] if split=='train' else p['fixed_validation_ids']
 for cid in cids:
  expected=CONTENT/(split+'-'+cid+'.json')
  if not expected.exists():raise FileNotFoundError('Contextual content inputs are not ready: '+str(expected))
  files=sorted((CONTENT/'token-states'/split/cid).glob('*.npz'))
  assert len(files)==json.loads(expected.read_text())['blocks']
  for fn in files:yield cid,fn
def check_reference(cid,rows,h,w,split):
 f=ROOT/'features/ov-GPU-f32/packed-ordered'/(split+'-'+cid+'.npz')
 with np.load(f,allow_pickle=False) as cached:
  ix={int(r):j for j,r in enumerate(cached['idx'])};ref=cached['hidden'][[ix[int(r)] for r in rows]]
 error=float(np.max(np.abs(h@w['readout'].T-ref@w['readout'].T)))
 flips=int(np.sum((h@w['readout'].T).argmax(1)!=(ref@w['readout'].T).argmax(1)))
 if error>1e-3 or flips:raise ValueError(('Unchanged terminal replay failed',cid,error,flips))
 return error
def training(dowrite):
 w=params();records=[];pairs=[];keys=[];targets=[];protect=[];maxerr=0.;slots={}
 for cid,fn in block_files('train'):
  rows,b=read_file(fn,w);h,p,z,q=replay(b,w);maxerr=max(maxerr,check_reference(cid,rows,h,w,'train'))
  block_metric=diagnostics(b,p);records.extend(block_metric);add_slot_diagnostics(slots,block_metric);pairs.extend(own_minus_other_logits(b,z))
  for j in np.flatnonzero(~np.isin(np.arange(len(b['noncue'])),b['pos'])):
   digest=hashlib.sha256(f'glide-query-preserve|{cid}|{fn.stem}|{int(j)}'.encode()).digest()
   protect.append((digest,b['noncue'][j].copy()))
  for i in range(len(rows)):
   own=b['owner']==i;other=(b['owner']>=0)&(~own)
   if not other.any():continue
   diff=b['k'][own].mean(0)-b['k'][other].mean(0)
   v=.25*8*diff/np.maximum((diff*diff).sum(1,keepdims=True),1e-8)
   cap=.12*np.linalg.norm(q[i],axis=1,keepdims=True);v*=np.minimum(1.,cap/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-8))
   targets.append(rotary(v[None],b['pos'][i:i+1],inverse=True).reshape(D));keys.append(b['x'][i])
 obs=np.asarray(records);summary={'train_cues_with_competitors':len(obs),'mean_own_mass_other_mass_log_per_token_rate_ratio_fraction_heads_preferring_other':obs.mean(0).tolist(),'mean_native_own_minus_other_attention_logit':float(np.mean(pairs)),'median_per_cue':np.median(obs,axis=0).tolist(),'position_in_packed_block':summarize_slots(slots),'original_logits_vs_fp32_max_error':maxerr}
 savej(DIR/'training-diagnostic.json',summary)
 print('attention diagnostic',summary,flush=True)
 if not dowrite:return
 anchor=np.stack([v for _,v in sorted(protect,key=lambda pair:pair[0])[:128]]).astype(float)
 _,sing,vt=np.linalg.svd(anchor,full_matrices=False);basis=vt[sing>sing[0]*1e-8]
 X=np.stack(keys).astype(float);X-=X@basis.T@basis;Y=np.stack(targets).astype(float)
 gram=X@X.T;reg=.15*np.diag(gram).mean()
 computed=Y.T@np.linalg.solve(gram+np.eye(len(X))*reg,X)
 U,L,V=np.linalg.svd(computed,full_matrices=False);L=L[:8];U=U[:,:8];V=V[:8]
 cap=.05*np.linalg.norm(w['qkv'][:D]);L*=min(1.,cap/np.linalg.norm(L));changed=(U*L)@V
 rng=np.random.default_rng(762211);u,_=np.linalg.qr(rng.standard_normal((D,8)));keys0=rng.standard_normal((8,D));keys0-=keys0@basis.T@basis;v,_=np.linalg.qr(keys0.T)
 random=(u*L)@v.T;summary.update({'protected_rows':len(anchor),'protected_input_subspace_rank':len(basis),'rank_before_storage':8,'singular_values_before_storage':L.tolist(),'controls':{}})
 assert np.allclose(np.linalg.norm(changed),np.linalg.norm(random))
 with safe_open(str(MODEL/'model-00001-of-00001.safetensors'),'pt',device='cpu') as f:orig=f.get_tensor(FUSED)
 base_q=orig[:D].float()
 def native_q(delta,amplitude):return (base_q+torch.from_numpy(delta).float()*amplitude).to(orig.dtype)
 target_after_rounding=float(torch.linalg.vector_norm(native_q(changed,1.).float()-base_q))
 # Match the actual stored tensor norms, including donor-dependent BF16 rounding.
 lo,hi=.5,1.5
 for _ in range(18):
  mid=(lo+hi)/2;measured=float(torch.linalg.vector_norm(native_q(random,mid).float()-base_q))
  if measured<target_after_rounding:lo=mid
  else:hi=mid
 for name,delta,amplitude in [('target',changed,1.),('norm_matched',random,(lo+hi)/2)]:
  src=orig.clone();src[:D]=native_q(delta,amplitude)
  assert torch.equal(src[D:],orig[D:]);actual=src[:D].float().numpy()-orig[:D].float().numpy()
  pth=DIR/(name+'-qkv.safetensors');save_file({FUSED:src},str(pth))
  norm=float(np.linalg.norm(actual));assert abs(norm/target_after_rounding-1.)<.005
  summary['controls'][name]={'file':sha(pth),'changed_rows':'0-1023 only','bf16_norm_match_scale':amplitude,'stored_q_delta_frobenius':norm,'target_stored_q_delta_frobenius':target_after_rounding,'before_rounding_protected_displacement_max':float(np.max(np.abs(anchor@delta.T)))*amplitude,'after_rounding_protected_displacement_max':float(np.max(np.abs(anchor@actual.T))),'relative_frobenius_to_q':float(np.linalg.norm(actual)/np.linalg.norm(w['qkv'][:D])),'tensor':wtproof(orig,src)}
 savej(DIR/'frozen.json',summary);print('Q slice physically frozen',flush=True)
def synthetic_parity():
 from transformers import ModernBertConfig
 from transformers.modeling_utils import no_init_weights
 from transformers.models.modernbert.modeling_modernbert import ModernBertEncoderLayer
 w=params();rng=np.random.default_rng(77);z=rng.normal(0,.75,(39,D)).astype('float32');cue=np.array([9,22,37]);inds=np.array([3,4,5,12,13,19,24,25]);offset=np.array([0,3,6,8]);pos=torch.arange(len(z))[None]
 cfg=ModernBertConfig.from_json_file(str(MODEL/'config.json'));cfg._attn_implementation='sdpa';cfg.reference_compile=False
 with no_init_weights():layer=ModernBertEncoderLayer(cfg,27)
 with safe_open(str(MODEL/'model-00001-of-00001.safetensors'),'pt',device='cpu') as f:layer.load_state_dict({n:f.get_tensor(PREFIX+n).float() for n,_ in layer.named_parameters()})
 layer.eval();mask=torch.zeros(1,1,len(z),len(z));b=block_arrays(z,cue,inds,offset,w)
 with torch.inference_mode():ref=normal(layer(torch.tensor(z)[None],attention_mask=mask,sliding_window_mask=mask,position_ids=pos)[0][0,cue].numpy(),w['final'])
 h,p,_,_=replay(b,w);err=float(np.abs(h@w['readout'].T-ref@w['readout'].T).max())
 vec=rng.normal(size=(3,H,S)).astype('float32');back=rotary(rotary(vec,cue),cue,True)
 result={'fixture':'synthetic intermediate residuals; no legal benchmark outcome','max_source_logit_error':err,'max_rotary_inverse_error':float(abs(vec-back).max()),'rows':len(cue),'native_source_tensor':FUSED,'shape':list(w['qkv'].shape)}
 savej(DIR/'implementation-check.json',result);print(result,flush=True);assert err<1e-4 and result['max_rotary_inverse_error']<1.e-5
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['synthetic','diagnose','freeze']);a=p.parse_args()
 if a.mode=='synthetic':synthetic_parity()
 else:training(a.mode=='freeze')
