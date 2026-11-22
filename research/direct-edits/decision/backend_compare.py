"""Compare the exact same original and edited tensor values under CPU INT8 and Intel GPU FP32."""
import argparse,json
import numpy as np
from safetensors.torch import load_file
from scipy.special import softmax
from compare import stats,scores
from fetch import ROOT,sha
from weights import arr,normed,read_rows,source,INTERNAL,savej
def run(arm,tier='pilot'):
 full='ov-GPU-f32';low='int8';wd=ROOT/'weights'/full/arm/tier
 data=[read_rows(backend,arm,'val',tier) for backend in (full,low)]
 a,b=data
 for k in ('y','sample_case','position','ntok'):assert np.array_equal(a[k],b[k]),k
 native,g,base=source();W,g,I=arr(native)[:7],arr(g),arr(base)
 head=arr(load_file(str(wd/'roles-readout.safetensors'))['weight'])[:7]
 changed=arr(load_file(str(wd/'roles-internal.safetensors'))[INTERNAL])-I
 y,c=a['y'],a['sample_case'];rng=np.random.default_rng(760230);boots=rng.integers(0,16,size=(4000,16))
 def features(x):
  return {'unchanged':x['logit'],'same_written_readout':x['hidden']@head.T,'same_written_internal':normed(x['before_normalization']+x['mlp_activation']@changed.T,g)@W.T}
 za,zb=map(features,data)
 norms=lambda x:np.maximum(np.linalg.norm(x,axis=1),1.e-10)
 cosine=np.sum(a['hidden']*b['hidden'],axis=1)/(norms(a['hidden'])*norms(b['hidden']))
 out={'judgments':len(set(c)),'rows':len(y),'source_fp32_fit_receipt':sha(wd/'receipt.json'),'median_final_hidden_cosine':float(np.median(cosine)),'minimum_final_hidden_cosine':float(cosine.min()),'methods':{}}
 for key,v in za.items():
  w=zb[key];p=v.argmax(1);q=w.argmax(1);cm1,m1=stats(y,p,c);cm2,m2=stats(y,q,c)
  d=scores(cm2[boots].sum(1))-scores(cm1[boots].sum(1))
  out['methods'][key]={'FP32':m1,'CPU_INT8_same_weights':m2,'argmax_agreement':float((p==q).mean()),
    'mean_abs_logit_error':float(abs(v-w).mean()),'maximum_abs_logit_error':float(abs(v-w).max()),'mean_abs_probability_error':float(abs(softmax(v,axis=1)-softmax(w,axis=1)).mean()),
    'quantized_minus_FP32_macro_f1_paired_95':np.quantile(d,[.025,.975]).tolist()}
 dst=ROOT/(arm+'-backend-comparison.json');savej(dst,out);print('CPU control',arm,[(n,v['argmax_agreement'],v['CPU_INT8_same_weights']['macro_f1']) for n,v in out['methods'].items()],flush=True);return dst
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('arm',choices=['alone','local']);a=p.parse_args();run(a.arm)
