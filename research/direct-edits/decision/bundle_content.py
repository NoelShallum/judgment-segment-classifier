import io,json,zipfile
import numpy as np
from fetch import ROOT,sha
p=json.loads((ROOT/'packed-full-plan.json').read_text());where=ROOT/'features/ov-GPU-f32/packed-ordered';target=ROOT/'glide-content-colab.zip'
with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
 for name in ['fetch.py','require_t4.py','colab_content.py','plan.json','packed-plan.json','packed-full-plan.json','content-plan.json','attention-plan.json','sampled-texts.json']:z.write(ROOT/name,name)
 for split,cases in [('train',p['fixed_fit_ids']),('val',p['fixed_validation_ids'])]:
  for cid in cases:
   d=np.load(where/(split+'-'+cid+'.npz'),allow_pickle=False);layout=json.loads((where/'layouts'/(split+'-'+cid+'.json')).read_text());stream=io.BytesIO()
   np.savez(stream,hidden=d['hidden'],layout_sha256=layout['sha256_input_ids_cues_targets']);z.writestr('portable-cues/'+split+'-'+cid+'.npz',stream.getvalue())
 z.writestr('README.txt','On a T4-only Colab VM, run python require_t4.py before installation or model fetching. Install transformers==4.53.3, safetensors, numpy and requests. Unpack this archive and run python fetch.py for the pinned Fastino revision. Run python -u colab_content.py train val. This rechecks nvidia-smi before loading weights and verifies each input and cue against existing reference vectors. First return the small content-hidden/*.json and content-hidden/*.npz files (sentence means) and retain the larger token-states/ directory for the controlled terminal-attention intervention. Original public LegalSeg and Fastino license/NOTICE apply. Use normal CUDA fp32 arithmetic.\n')
print(target.name,target.stat().st_size,sha(target))
