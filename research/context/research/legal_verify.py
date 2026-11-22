"""Materialize and compare legal donor output tensors on full token forwards."""
import json,sys,hashlib
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
import torch
from transformers import BertTokenizerFast
from legal_encode import H,ROOT,MODELS,check,get,records,windows,sha
from safetensors.numpy import load_file
@torch.inference_mode()
def main(official=False):
    if not official:
        tok=BertTokenizerFast.from_pretrained(H/'tokenizer',local_files_only=True)
        ns={}
        docs=records('fresh');groups=[docs[j]['text'] for j in (0,21,48,95)]
        for k in MODELS:
            m=get(k);samples=[]
            for texts in groups:
                for j in (0,len(texts)//3,len(texts)//2,len(texts)-1):
                    ids=[tok.encode(texts[a],add_special_tokens=False) if 0<=a<len(texts) else [] for a in ((j-2,j-1,j) if k=='past' else (j-1,j,j+1))]
                    samples.append(windows(ids,tok.sep_token_id,tok.cls_token_id)[0][0])
            width=max(map(len,samples));x=np.zeros((len(samples),width),np.int64)
            for a,v in enumerate(samples):x[a,:len(v)]=v
            xt=torch.as_tensor(x,device='cuda');mask=xt!=0;pool=m.bert(xt,attention_mask=mask).pooler_output.float()
            # Functional classification through the very same native module with explicit activation promotion.
            handle=m.classifier.register_forward_pre_hook(lambda module,args:(args[0].float(),))
            ww=m.classifier.weight.clone();bb=m.classifier.bias.clone()
            for name in ('edited','pair','shuffled'):
                p=load_file(str(H/(k+'-'+name+'.safetensors')))
                m.classifier.weight.copy_(torch.as_tensor(p['weight'],device='cuda'));m.classifier.bias.copy_(torch.as_tensor(p['bias'],device='cuda'))
                expected=pool.cpu().numpy()@p['weight'].T+p['bias'];real=m(xt,attention_mask=mask).logits.cpu().numpy()
                np.testing.assert_allclose(expected,real,rtol=1e-5,atol=2e-5);np.testing.assert_array_equal(expected.argmax(1),real.argmax(1))
                d=float((m.classifier.weight-ww).norm());db=float((m.classifier.bias-bb).norm())
                ns[k+'-'+name]=dict(examples=len(real),max_absolute_error=float(abs(real-expected).max()),label_mismatches=0,weight_frobenius=d,bias_frobenius=db,source_weight_norm=float(ww.norm()),tensor='classifier.weight and classifier.bias',rows_changed=int(((m.classifier.weight-ww).abs().sum(1)>0).sum()))
            handle.remove();del m;torch.cuda.empty_cache()
        (H/'verification.json').write_text(json.dumps(ns,indent=2))
    names=['freeze.json','evaluation.json','verification.json']+(['official.json','test-predictions.npz'] if official else [])
    names+=['legal_encode.py','legal_solve.py','legal_verify.py']
    paths=[H/n for n in names]+list(H.glob('*-predictions.npz'))+list(H.glob('*-fit.json'))+list(H.glob('*-select.json'))+list(H.glob('*-confirm.json'))+list(H.glob('*-fresh.json'))
    paths.extend(p for p in H.glob('*.safetensors') if 'donor' not in p.name)
    if official:paths+=list(H.glob('*-test.json'))
    paths=list(dict.fromkeys(paths))
    dest=ROOT/('legal-official-results.zip' if official else 'legal-results.zip');temp=dest.with_suffix('.part')
    with ZipFile(temp,'w',ZIP_DEFLATED) as z:
        for p in paths:z.write(p,p.name)
        z.writestr('SHA256SUMS.json',json.dumps({p.name:sha(p) for p in paths},indent=2))
    temp.replace(dest);print('native verification',check(),len(paths),sha(dest),flush=True)
if __name__=='__main__':main(len(sys.argv)>1 and sys.argv[1]=='official')
