import os, sys
from pathlib import Path
p=Path(__file__).resolve().parent
sys.path.insert(0,str(p/'deps'));sys.path.append(str(p.parent/'ultrabert/deps311'))
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='1'
os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['HF_HUB_OFFLINE']='1'
import torch
from gliner2.models.boundary.engine import BoundaryExtractor
torch.set_num_threads(1);torch.set_num_interop_threads(1)
m=BoundaryExtractor.from_pretrained(str(p/'model'),use_flashdeberta=False).eval();m.processor.change_mode(is_training=False)
sc=m.create_schema().entities(['other judgment material','facts','issue','petitioner argument','respondent argument','court reasoning','decision']).build();print('schema',sc,flush=True)
a='The appeal was dismissed. The hearing was held on Monday.'
b=m.processor.collate_fn_inference([(a,sc)],architecture='boundary',max_len=None,error_policy='raise');print(list(vars(b))[:23],flush=True)
with torch.inference_mode():
 c=m._encode_core(b); print({k:(getattr(v,'shape',v) if k in ('query_states','text_states','word_offsets','ext_specs') else type(v).__name__) for k,v in c.items()},flush=True)
 h=m.boundary_head; s=h.boundary_encoder(c['text_states'],c['text_mask']); g=h.boundary_query_head(s.states,s.mask,c['text_states'],c['text_mask'],c['query_states'],c['query_mask']); print('margins',g.inside_logits.shape,g.start_logits[0,:,0].tolist(),flush=True)
