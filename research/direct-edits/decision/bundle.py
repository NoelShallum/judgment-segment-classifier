"""Build a small Colab job. Public weights download from the publisher's exact commit on the VM."""
import io
import json
import zipfile
from fetch import ROOT
from cohort import case_rows
plan=json.loads((ROOT/'plan.json').read_text())
data={'source':plan['source'],'train':case_rows('train',set(plan['fit_judgment_ids'])),'val':case_rows('val',set(plan['val_judgment_ids']))}
out=ROOT/'glide-colab-worker.zip'
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
 for name in ('fetch.py','extract.py','weights.py','run.py','cohort.py','compare.py','plan.json'):
  z.write(ROOT/name,name)
 z.writestr('sampled-texts.json',json.dumps(data,ensure_ascii=False))
 z.writestr('README.txt','In a CUDA Colab VM with torch: install transformers==4.53.3 safetensors numpy requests, unpack, run python fetch.py then: python -u run.py --backend cuda --tier extended alone local. The worker writes split-specific weights, score JSON, and direct-tensor forward parity proofs in weights/cuda/. Keep both controls and source.json. Dataset contains only the selected public seven-role LegalSeg judgments. Full source CSV SHA-256 is in plan.json. Corpus https://huggingface.co/datasets/L-NLProc/LegalSeg_CSV. GLiDE weights are under Fastino Community License https://huggingface.co/fastino/GLiDE-0.4B/blob/16fb9d63cc34d1b4b5a1fb7cbf2d9d4a3dd5ad71/LICENSE.\n')
print(out.name,out.stat().st_size,flush=True)
