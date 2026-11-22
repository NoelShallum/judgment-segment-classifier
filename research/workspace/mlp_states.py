"""Capture the final BERT feed-forward keys and residuals from the pinned FP32 encoder."""
from pathlib import Path
import json
import numpy as np
from encoder import fingerprint,TOK
from openvino_worker import ART
import openvino as ov
from openvino.runtime import opset13 as op
from transformers import BertTokenizerFast
from program_trial import fixture_rows,official_sample
from role_fixtures import ANCHORS,CHECKS
from seven_paths import ROOT,FOLDERS,folder_for
from windowing import token_windows

CACHE=ROOT/'cache/mlp-11-states'
def engine():
    core=ov.Core();old=core.read_model(ART/'encoder.xml')
    ops={op.get_friendly_name():op for op in old.get_ops()}
    names=['__module.bert.encoder.layer.11.intermediate.intermediate_act_fn/aten::gelu/Gelu','__module.bert.encoder.layer.11.output/aten::add/Add']
    zero=op.constant(np.array(0,dtype=np.int64));axis=op.constant(np.array(1,dtype=np.int64))
    taps=[op.gather(ops[n],zero,axis) for n in names]
    combined=op.concat(taps+[old.output(0).get_node().input_value(0)],1)
    graph=ov.Model([combined],old.get_parameters(),'bert-final-mlp-writes')
    settings={'INFERENCE_PRECISION_HINT':'f32','PERFORMANCE_HINT':'LATENCY','NUM_STREAMS':'1','CACHE_DIR':str(CACHE/'compiled')}
    return core.compile_model(graph,'GPU',settings)
def encode(session,tok,name,texts):
    target=CACHE/(name+'.npz')
    sig=fingerprint(texts)
    if target.exists():
        with np.load(target) as d:
            if str(d['signature'])==sig:return
    content=tok(texts,add_special_tokens=False,return_attention_mask=False,return_token_type_ids=False,verbose=False)['input_ids']
    flat=[(i,ids,w) for i,seq in enumerate(content) for ids,w in token_windows(tok,seq,512)]
    out=np.empty((len(flat),4608),dtype=np.float32)
    bins={w:[] for w in (16,32,64,128,256,512)}
    for i,(_,ids,_) in enumerate(flat):bins[next(k for k in bins if len(ids)<=k)].append(i)
    for width,idx in bins.items():
        batch=min(24,max(1,1536//width))
        for start in range(0,len(idx),batch):
            here=idx[start:start+batch]
            ids=np.zeros((batch,width),np.int64);mask=np.zeros_like(ids)
            for j,i in enumerate(here+[here[-1]]*(batch-len(here))):
                tokens=flat[i][1];ids[j,:len(tokens)]=tokens;mask[j,:len(tokens)]=1
            out[here]=session({'input_ids':ids,'attention_mask':mask})[0][:len(here)]
    tmp=target.with_suffix('.part')
    with tmp.open('wb') as f:np.savez(f,signature=sig,keys=out[:,:3072],residual=out[:,3072:3840],pooled=out[:,3840:],owners=[q[0] for q in flat],mass=[q[2] for q in flat])
    tmp.replace(target);print('Captured final MLP',name,len(texts),len(flat),flush=True)
def main():
    CACHE.mkdir(exist_ok=True);s=engine();tok=BertTokenizerFast.from_pretrained(TOK,local_files_only=True)
    keys=fixture_rows(ANCHORS)+fixture_rows(CHECKS);encode(s,tok,'keys',[r[2] for r in keys])
    for k in ('select','confirm'):
        rows,_,_=official_sample(k,160);encode(s,tok,k,[r[2] for r in rows])
    for k in FOLDERS:
        rows=json.loads((folder_for(k)/'approved.json').read_text());encode(s,tok,k,[r['text'] for r in rows])
if __name__=='__main__':main()
