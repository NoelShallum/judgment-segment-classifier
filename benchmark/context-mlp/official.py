"""Apply the native MLP patch only after the prespecified two-cohort decision."""
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch
from safetensors.numpy import load_file
from transformers import BertTokenizerFast
import ffn

H=ffn.H
def savezip(files):
    dest=H.parent/'context-mlp-official.zip'
    with ZipFile(dest.with_suffix('.part'),'w',ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.name)
        z.writestr('SHA256SUMS.json',json.dumps({p.name:ffn.enc.sha(p) for p in files}))
    dest.with_suffix('.part').replace(dest)
@torch.inference_mode()
def main():
    ffn.enc.check()
    freeze=json.loads((H/'freeze.json').read_text());dev=json.loads((H/'results.json').read_text())
    passed=bool(freeze['config']['strength']>0 and all(
        cohort['all']['selected']['macro_f1']>cohort['all']['context']['macro_f1'] and
        cohort['all']['selected']['pair_errors']<cohort['all']['context']['pair_errors']
        for cohort in [dev['confirm'],dev['fresh']]))
    lock={'promoted':passed,'freeze_sha256':ffn.enc.sha(H/'freeze.json'),'protocol_sha256':ffn.enc.sha(H/'protocol.json'),
          'evidence_sha256':ffn.enc.sha(H/'results.json'),'config':freeze['config']}
    ffn.scale.dump(H/'official-lock.json',lock)
    print('Promote native write',passed,flush=True)
    if not passed:
        savezip([H/'official-lock.json']);return
    assert ffn.enc.sha(H/'native-patch.safetensors')==freeze['patch_sha256']
    c=freeze['config'];old,base=ffn.scale.legacy('test');original=base.raw
    m=ffn.enc.get(ffn.KIND);block=m.bert.encoder.layer[-1].output.dense
    tok=BertTokenizerFast.from_pretrained(ffn.enc.H/'tokenizer',local_files_only=True)
    patch=load_file(str(H/'native-patch.safetensors'));i=patch['cols']
    w=block.weight[:,i].clone();b=block.bias.clone();original_p=old.argmax(1)
    pred={'context':original_p}
    for mode in ('edit','shuffled'):
        block.weight[:,i]=(w.float()+ffn.gput(patch[mode+'-weight'])).half()
        block.bias.copy_((b.float()+ffn.gput(patch[mode+'-bias'])).half())
        h=ffn.capture(m,tok,'test',physical_only=True).hidden
        raw=h@m.classifier.weight.cpu().numpy().T+m.classifier.bias.cpu().numpy()
        dr=ffn.lib.probabilities(.65*raw)-ffn.lib.probabilities(.65*original)
        doc=ffn.modified(ffn.legal.fetch_group('test'),m,h)[0] if c['rule']=='doc' else old
        pred[mode]=ffn.rule(c,old,doc,dr)
        pred[mode+'-standalone']=raw.argmax(1)
        block.weight[:,i]=w;block.bias.copy_(b)
    pred['selected']=pred['edit'];np.savez_compressed(H/'official-predictions.npz',**ffn.arraypack(base,pred))
    ffn.scale.dump(H/'official-results.json',dict(config=c,cases=len(base.case_ids),rows=len(base.gold),
          all={k:ffn.lib.metric(base.gold,p) for k,p in pred.items()},official_lock_sha256=ffn.enc.sha(H/'official-lock.json')))
    savezip([H/n for n in ('official.py','official-lock.json','official-results.json','official-predictions.npz')])
if __name__=='__main__':main()
