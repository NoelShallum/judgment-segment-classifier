from pathlib import Path
p=Path('packed.py');s=p.read_text()
s=s.replace('import hashlib','import hashlib\nimport shutil')
s=s.replace("""  if method=='packed-shuffled':""","""  if method=='packed-rotated':
   cut=(len(chunk)+1)//2
   chunk=chunk[cut:]+chunk[:cut]
  elif method=='packed-shuffled':""")
s=s.replace(" g=p[tier]"," g={'train_cases':12,'validation_rows_per_shared_judgment':4} if tier in ('whole16','match64') else p[tier]")
s=s.replace("  wanted=set(range(len(d)) if split=='train' else p['validation_indices'][gid][:g['validation_rows_per_shared_judgment']])","  wanted=set(range(len(d)) if split=='train' or tier=='whole16' else p['validation_indices'][gid][:g['validation_rows_per_shared_judgment']])")
s=s.replace("""  for ids,cues,members in grouping(d,tok,method):
   if not (set(members)&wanted-set(values)):continue""", """  block_list=list(grouping(d,tok,method))
  counterfactual=list(grouping([(s,'.invalid-label') for s,_ in d],tok,method))
  assert block_list==counterfactual,'A label influenced an inference input'
  assert sorted(i for _,_,block in block_list for i in block)==list(range(len(d)))
  footprint=hashlib.sha256(json.dumps(block_list).encode()).hexdigest()
  lay={str(i):[slot,len(members),blockno] for blockno,(ids,cues,members) in enumerate(block_list) for slot,i in enumerate(members)}
  (cache/'layouts').mkdir(exist_ok=True)
  (cache/'layouts'/(split+'-'+gid+'.json')).write_text(json.dumps({'sha256_input_ids_cues_targets':footprint,'invalid_label_counterfactual_identical':True,'index_to_slot_group_size_and_index':lay,'rows':len(d)}))
  for ids,cues,members in block_list:
   if not (set(members)&(wanted-set(values))):continue""")
s=s.replace("""cli.add_argument('arm',choices=['packed-alone','packed-ordered','packed-shuffled'])""","""cli.add_argument('arm',choices=['packed-alone','packed-ordered','packed-shuffled','packed-rotated','full-study'])""")
s=s.replace(""" (ROOT/'packed-plan.json').write_text(json.dumps(PROTOCOL,indent=2))
 r=ready(a.device,a.hint);verify(r,a.device,a.hint)
 extract_pack(r,backend,a.arm,'train',a.tier);freeze(backend,a.arm,a.tier)
 extract_pack(r,backend,a.arm,'val',a.tier);score(backend,a.arm,a.tier)""", """ (ROOT/'packed-plan.json').write_text(json.dumps(PROTOCOL,indent=2))
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
  create()""")
p.write_text(s)
