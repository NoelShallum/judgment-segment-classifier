"""Stream official encoder layers on one CPU thread for two deterministic real training blocks; no GPU call."""
import json
import numpy as np
import torch
from safetensors import safe_open
from transformers import ModernBertConfig
from transformers.modeling_utils import no_init_weights
from transformers.models.modernbert.modeling_modernbert import ModernBertEncoderLayer
from tokenizers import Tokenizer
from attention_write import params,block_arrays,replay,check_reference,diagnostics,normal,DIR
from cohort import case_rows
from fetch import ROOT,MODEL,sha
from weights import savej
def encode(tokenizer,s):return tokenizer.encode(s,add_special_tokens=False)
def make_block(cid,doc,tok):
 protocol=json.loads((ROOT/'packed-plan.json').read_text())
 prefix=[50281]+encode(tok,protocol['prefix']).ids;anchor=encode(tok,protocol['anchor']).ids
 ids=list(prefix);cues=[];rows=[];content=[];parts=[0]
 for i,(s,_) in enumerate(doc):
  a=encode(tok,'\n'+s)
  if i and (len(rows)==8 or len(ids)+len(a.ids)+len(anchor)+1>512):break
  sel=[j for j,(x,y) in enumerate(a.offsets) if any(not c.isspace() for c in ('\n'+s)[x:y])]
  content.extend(len(ids)+j for j in sel);parts.append(len(content));ids.extend(a.ids);ids.extend(anchor);rows.append(i);cues.append(len(ids)-1)
 ids.append(50282)
 return np.array(ids),np.array(cues),np.array(content),np.array(parts),np.array(rows)
def run():
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 p=json.loads((ROOT/'packed-full-plan.json').read_text());cases=p['fixed_fit_ids'][:2];docs=case_rows('train',set(cases));tok=Tokenizer.from_file(str(MODEL/'tokenizer.json'))
 inputs={k:make_block(k,docs[k],tok) for k in cases}
 nf=MODEL/'model-00001-of-00001.safetensors';cfg=ModernBertConfig.from_json_file(str(MODEL/'config.json'));cfg._attn_implementation='sdpa';cfg.reference_compile=False
 dest=DIR/'cpu-probes';dest.mkdir(exist_ok=True)
 with torch.inference_mode(),safe_open(str(nf),'pt',device='cpu') as f:
  emb=f.get_tensor('model.embeddings.tok_embeddings.weight');g=f.get_tensor('model.embeddings.norm.weight').float().numpy();buffers={}
  for c,(idx,*info) in inputs.items():
   z=torch.from_numpy(normal(emb[idx].float().numpy(),g))[None];pos=torch.arange(len(idx))[None]
   glob=torch.zeros((1,1,len(idx),len(idx)));local=glob.masked_fill((pos-pos.T).abs()>64,-3e38);buffers[c]=[z,pos,glob,local]
  del emb
  for i in range(27):
   with no_init_weights(): layer=ModernBertEncoderLayer(cfg,i)
   prefix=f'model.layers.{i}.';layer.load_state_dict({n:f.get_tensor(prefix+n).float() for n,_ in layer.named_parameters()});layer.eval()
   for v in buffers.values():v[0]=layer(v[0],position_ids=v[1],attention_mask=v[2],sliding_window_mask=v[3])[0]
   del layer
   if i in (0,7,15,26):print('real CPU probe layer',i,flush=True)
  for c,(idx,cues,pos,offs,rows) in inputs.items():
   np.savez(dest/(c+'.npz'),before_terminal_residual=buffers[c][0][0].numpy(),input_ids=idx,cue_positions=cues,content_positions=pos,offsets=offs,rows=rows)
 w=params();all_diag=[];res={}
 for c in cases:
  with np.load(dest/(c+'.npz')) as d:
   b=block_arrays(d['before_terminal_residual'],d['cue_positions'],d['content_positions'],d['offsets'],w)
   h,prob,_,_=replay(b,w);err=check_reference(c,d['rows'],h,w,'train');stats=diagnostics(b,prob);all_diag.extend(stats)
   res[c]={'cues':len(d['rows']),'tokens':len(d['input_ids']),'max_reference_logit_error':err,'mean_own_mass_other_mass_log_per_token_rate_ratio_fraction_heads_preferring_other':np.asarray(stats).mean(0).tolist()}
 summary={'subset':'First ordered block of the first two pre-registered TRAINING judgments, selected without labels. This is a mechanism probe only; no validation score.','cases':res,'overall':np.asarray(all_diag).mean(0).tolist()}
 savej(dest/'diagnostic.json',summary);print(summary,flush=True)
if __name__=='__main__':run()
