"""Evaluate the same pre-existing tensors across ordered and singleton inputs, crossing head fit and input context."""
import json
import numpy as np
from safetensors.torch import load_file
from weights import arr,read_rows,normed,source,INTERNAL,savej
from compare import stats,scores
from fetch import ROOT,sha
def run():
 b='ov-GPU-f32'
 weights=ROOT/'weights'/b
 train=source()
 native,final_norm,ffn=arr(train[0])[:7],arr(train[1]),arr(train[2])
 # The original large and small replays used exactly the same 357 training rows.
 folders={'ordered':weights/'packed-ordered'/'whole16','singleton':weights/'packed-alone'/'match64'}
 for path in folders.values():
  receipt=json.loads((path/'receipt.json').read_text());assert receipt['train_sentences']==357
 samples={'ordered':read_rows(b,'packed-ordered','val','match64'),'singleton':read_rows(b,'packed-alone','val','match64')}
 y=samples['ordered']['y'];c=samples['ordered']['sample_case']
 for field in ('y','sample_case','position'):assert np.array_equal(samples['ordered'][field],samples['singleton'][field])
 h0=samples['ordered']['hidden'];h1=samples['singleton']['hidden'];cos=(h0*h1).sum(1)/(np.linalg.norm(h0,axis=1)*np.linalg.norm(h1,axis=1))
 scenes={};cm={};estimate={}
 rng=np.random.default_rng(760230);bs=rng.integers(0,16,(4000,16))
 for readout,folder in folders.items():
  W=arr(load_file(str(folder/'roles-readout.safetensors'))['weight'])[:7]
  d=arr(load_file(str(folder/'roles-internal.safetensors'))[INTERNAL])-ffn
  for inp,x in samples.items():
   logits={'native':x['logit'],'direct_head':x['hidden']@W.T,'internal':normed(x['before_normalization']+x['mlp_activation']@d.T,final_norm)@native.T}
   for kind,z in logits.items():
    key=(readout,inp,kind);pred=z.argmax(1);cm[key],estimate[key]=stats(y,pred,c);scenes[key]=pred
 out={'description':'Adaptive CPU replay of frozen weights on 64 identical target sentences. Within each row below the weight tensor is exactly identical while input mode changes. Both learned tensor sources fit the exact same 357 training rows.','median_within_sentence_hidden_cosine':float(np.median(cos)),'minimum_within_sentence_hidden_cosine':float(cos.min()),'results':{}}
 for kind in ('native','direct_head','internal'):
  out['results'][kind]={}
  for readout in folders:
   left=(readout,'ordered',kind);right=(readout,'singleton',kind)
   diff=scores(cm[right][bs].sum(1))-scores(cm[left][bs].sum(1))
   out['results'][kind][readout]={'ordered_macro_f1':estimate[left]['macro_f1'],'singleton_macro_f1':estimate[right]['macro_f1'],'singleton_minus_ordered_ci':np.quantile(diff,[.025,.975]).tolist(),'changed_labels':int((scenes[left]!=scenes[right]).sum())}
 savej(ROOT/'packed-crossed-context.json',out);print(out,flush=True)
if __name__=='__main__':run()
