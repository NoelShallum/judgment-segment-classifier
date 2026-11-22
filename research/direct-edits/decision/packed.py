"""Score every sentence in an ordered block from a single bidirectional encoder pass with one recorded readout position per sentence."""
import argparse
from collections import defaultdict
import hashlib
import shutil
import json
import time
import numpy as np
from cohort import case_rows,LABELS
from fetch import ROOT,sha
from ov_graph import inputs
from ov_worker import ready,verify
from weights import freeze,score
PREFIX='Indian court judgment. Decide the rhetorical role of each sentence. Options: (A) None, (B) Facts, (C) Issue, (D) Argument of the petitioner, (E) Argument of the respondent, (F) Reasoning of the court, (G) Decision. Respond with the option letter.'
ANCHOR='\nAnswer: ('
PROTOCOL={'format':'One list of seven options followed by full sentences; each sentence ends with its own readout cue. This is a new prompt format, distinct from publisher-native single-decision prompting.',
          'prefix':PREFIX,'anchor':ANCHOR,'maximum_sentences_per_block':8,'maximum_tokens_per_block':512,
          'target_input':'A single unusually long sentence gets its own longer block, up to 7,999 with both ends retained. No sentence is silently dropped.',
          'ordering':'ordered natural blocks or same-block deterministic shuffle; block partition and target text are identical.',
          'fit':'same complete training judgments, readout and last feed-forward ridge writes as native trial; use no validation labels in selection.'}
def grouping(d,tok,method):
 prefix=[tok.cls_id]+tok.enc.encode(PREFIX,add_special_tokens=False)
 mark=tok.enc.encode(ANCHOR,add_special_tokens=False)
 chunks=[];group=[];size=len(prefix)+1
 for i,(s,_) in enumerate(d):
  raw=tok.enc.encode('\n'+s,add_special_tokens=False)
  maximum=7999-len(prefix)-len(mark)-1
  if len(raw)>maximum:raw=raw[:maximum//2]+raw[-(maximum-maximum//2):]
  seq=raw+mark
  if group and (method=='packed-alone' or len(group)>=8 or size+len(seq)>512):
   chunks.append(group);group=[];size=len(prefix)+1
  group.append((i,seq));size+=len(seq)
 if group:chunks.append(group)
 for chunk in chunks:
  if method=='packed-rotated':
   cut=(len(chunk)+1)//2
   chunk=chunk[cut:]+chunk[:cut]
  elif method=='packed-shuffled':
   chunk=sorted(chunk,key=lambda val:hashlib.sha256(json.dumps(['glide-pack-order-2211',val[0],val[1]]).encode()).digest())
  inp=list(prefix);positions=[];members=[]
  for i,seq in chunk:
   inp.extend(seq);positions.append(len(inp)-1);members.append(i)
  inp.append(tok.sep_id)
  yield inp,positions,members
def extract_pack(r,backend,method,split,tier):
 (comp,W),tok,_=r
 p=json.loads((ROOT/'plan.json').read_text())
 g={'train_cases':12,'validation_rows_per_shared_judgment':4} if tier in ('whole16','match64') else p[tier]
 gids=p['fit_judgment_ids'][:g['train_cases']] if split=='train' else p['val_judgment_ids']
 docs=case_rows(split,set(gids));cache=ROOT/'features'/backend/method;cache.mkdir(parents=True,exist_ok=True)
 signature=hashlib.sha256(json.dumps([PROTOCOL,method,backend,sha(ROOT/'plan.json')],sort_keys=True).encode()).hexdigest()
 start=time.monotonic();count=0
 for gid in gids:
  d=docs[gid]
  wanted=set(range(len(d)) if split=='train' or tier=='whole16' else p['validation_indices'][gid][:g['validation_rows_per_shared_judgment']])
  path=cache/(split+'-'+gid+'.npz')
  if path.exists():
   with np.load(path,allow_pickle=False) as old:
    if str(old['recipe'])!=signature:raise ValueError('Existing packed features have a different protocol')
    if wanted<=set(old['idx'].tolist()):continue
  values={}
  if path.exists():
   with np.load(path,allow_pickle=False) as old:
    for j,i in enumerate(old['idx']): values[int(i)]=tuple(old[k][j] for k in ('hidden','before_normalization','mlp_activation','logit','ntok','text_sha256'))
  block_list=list(grouping(d,tok,method))
  counterfactual=list(grouping([(s,'.invalid-label') for s,_ in d],tok,method))
  assert block_list==counterfactual,'A label influenced an inference input'
  assert sorted(i for _,_,block in block_list for i in block)==list(range(len(d)))
  footprint=hashlib.sha256(json.dumps(block_list).encode()).hexdigest()
  lay={str(i):[slot,len(members),blockno] for blockno,(ids,cues,members) in enumerate(block_list) for slot,i in enumerate(members)}
  (cache/'layouts').mkdir(exist_ok=True)
  (cache/'layouts'/(split+'-'+gid+'.json')).write_text(json.dumps({'sha256_input_ids_cues_targets':footprint,'invalid_label_counterfactual_identical':True,'index_to_slot_group_size_and_index':lay,'rows':len(d)}))
  for ids,cues,members in block_list:
   if not (set(members)&(wanted-set(values))):continue
   n=len(ids);width=min(7999,32*((n+31)//32))
   out=comp(inputs(ids,cues,width))
   hs,rs,us=(out[k][0].astype('float32') for k in ('hidden','before_normalization','mlp_activation'))
   logits=hs@W.T
   for j,i in enumerate(members): values[i]=(hs[j],rs[j],us[j],logits[j],n,hashlib.sha256(d[i][0].encode()).hexdigest())
   count+=len(members)
  idx=sorted(values)
  tmp=path.with_suffix('.part')
  with tmp.open('wb') as f:
   np.savez(f,recipe=signature,idx=idx,**{k:np.stack([values[i][j] for i in idx]) for j,k in enumerate(('hidden','before_normalization','mlp_activation','logit','ntok','text_sha256'))})
  tmp.replace(path)
  print(method,split,gid,'targets',len(wanted),'block_rows',len(idx),'new_block_rows',count,'seconds',round(time.monotonic()-start,1),flush=True)
if __name__=='__main__':
 cli=argparse.ArgumentParser();cli.add_argument('arm',choices=['packed-alone','packed-ordered','packed-shuffled','packed-rotated','full-study']);cli.add_argument('--device',default='GPU');cli.add_argument('--hint',default='f32');cli.add_argument('--tier',default='pilot')
 a=cli.parse_args(); backend='ov-'+a.device+'-'+a.hint
 (ROOT/'packed-plan.json').write_text(json.dumps(PROTOCOL,indent=2))
 full=json.loads((ROOT/'packed-full-plan.json').read_text())
 assert full['source_plan_sha256']==sha(ROOT/'plan.json')
 r=ready(a.device,a.hint);verify(r,a.device,a.hint)
 sequence=[('packed-ordered','whole16'),('packed-rotated','match64'),('packed-alone','match64')] if a.arm=='full-study' else [(a.arm,a.tier)]
 for arm,tier in sequence:
  if arm=='packed-rotated':
   parent=ROOT/'weights'/backend/'packed-ordered'/'whole16';dst=ROOT/'weights'/backend/arm/tier
   dst.mkdir(parents=True,exist_ok=True)
   recipe=json.loads((parent/'receipt.json').read_text())
   recipe['weight_source']='frozen ordered eight-sentence arm';recipe['ordered_receipt_sha256']=sha(parent/'receipt.json')
   for filename,proof in recipe['files'].items():
    assert sha(parent/filename)==proof['sha256'];shutil.copyfile(parent/filename,dst/filename)
   (dst/'receipt.json').write_text(json.dumps(recipe,indent=2))
  else:
   extract_pack(r,backend,arm,'train',tier);freeze(backend,arm,tier)
  extract_pack(r,backend,arm,'val',tier);score(backend,arm,tier)
  from report import create
  create()
