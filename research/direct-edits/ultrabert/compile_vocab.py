"""Verify and time an inference-only seven-column compilation of the physical embedding write."""
import json
from pathlib import Path
import time
import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file,save_file
from tokenizers import Tokenizer
from prepare import ROOT,save
from extract import groups
from vocab import SRC,fit_coeffs,target_logs
from portable_vocab import classify
from study import h
import hashlib

def main():
    stats=ROOT/'vocab-counts.npz';folder=ROOT/'vocab-screen'
    with np.load(stats) as a:df=a['token_document_frequency'].astype(float);prior=a['class_counts'];pp=a['position'];tx=a['log_transition']
    out={'position.logp':np.log((pp+.5)/(pp.sum(1,keepdims=True)+3.5)).astype(np.float32),'transition.logp':tx.astype(np.float32)}
    r=json.loads((folder/'report.json').read_text())
    with safe_open(str(ROOT/'model/model.safetensors'),framework='numpy') as f:base=f.get_tensor(SRC)
    ckhash={}
    for name,m in r['experiments'].items():
        _,s,gain,ps,ts=m['best_selection'];w=fit_coeffs(base,target_logs(df,prior,s),df.sum(1))
        patch=load_file(str(folder/f'{name}-patch.safetensors'));a=patch['vocab.role_residual'];lift=patch['role.to_hidden']
        matrix=np.zeros((len(base),7),np.float32);sha=hashlib.sha256()
        for i in range(0,len(base),1024):
            material=(base[i:i+1024]+(a[i:i+1024]@lift).astype(np.float32)).astype(np.float32)
            sha.update(material.tobytes());matrix[i:i+1024]=(material@w).astype(np.float32)
        assert sha.hexdigest()==m['tensor']['materialized_sha256']
        ckhash[name]=sha.hexdigest();out[name+'.token_to_class']=matrix
        out[name+'.position_gain']=np.array([ps],np.float32);out[name+'.transition_gain']=np.array([ts],np.float32)
    path=folder/'compiled-token-scores.safetensors'
    save_file(out,str(path),metadata={'source_tensor':SRC,'base_revision':'545ac9d9c7aece444f014ada5a1e9818132b4d35','mode':'exact task-specific logit compilation evaluated in float32'})
    model=load_file(str(path));tok=Tokenizer.from_file(str(ROOT/'model/tokenizer.json'))
    gs=groups('common')['common']
    cv=np.load(ROOT/'vocab-common16-predictions.npz')
    times={};agreement={}
    for key,doc in [('text',False),('structure',True)]:
        rows=[r['text'] for item in gs for r in item['rows']]
        spans=[len(item['rows']) for item in gs]
        # Warm-up, followed by three independent timed re-tokenizations and complete decodes.
        classify(rows[:10],model,tok,doc)
        time_list=[]
        for _ in range(3):
            st=time.perf_counter();loaded=[];j=0
            for n in spans:
                loaded.extend(classify(rows[j:j+n],model,tok,doc));j+=n
            time_list.append(time.perf_counter()-st)
        p=np.asarray(loaded);agreement[key]=int(np.count_nonzero(p!=cv[key]))
        times[key]={'wall_seconds':time_list,'median_ms_per_row':1000*float(np.median(time_list))/len(rows)}
        assert agreement[key]==0,(key,agreement[key])
    rec={'source':'real FP32 original 49,152 × 768 input embeddings plus stored low-rank write, projected once into seven score directions',
         'disagreements_vs_native_embedding_predictions_on_shared16':agreement,'cases':len(gs),'rows':len(rows),'timing_cpu_single_thread':times,
         'timing_note':'Measured during other same-host jobs; model loading excluded; tokenize and seven-state document decoding included.',
         'tensor_sha256':ckhash,'bytes_logits':path.stat().st_size,'bytes_tokenizer':(ROOT/'model/tokenizer.json').stat().st_size,
         'inference_file':'portable_vocab.py','weights':str(path.relative_to(ROOT)),'tokenizer':'model/tokenizer.json'}
    save(folder/'portable-replay.json',rec);print(rec,flush=True)
if __name__=='__main__':main()
