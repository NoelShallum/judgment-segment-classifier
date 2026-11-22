import json
import numpy as np
from scipy.special import softmax
from safetensors.torch import load_file
from compare import stats,scores
from weights import source,arr,normed,INTERNAL,read_rows,savej
from fetch import ROOT,sha
def evaluate():
 b='ov-GPU-f32';arm='packed-ordered';tier='whole16';wd=ROOT/'weights'/b/arm/tier
 x=read_rows(b,arm,'val',tier);z=x['logit'];u=x['mlp_activation'];h=x['hidden'];r=x['before_normalization'];y=x['y']
 p=json.loads((wd/'predictions.json').read_text());gold=np.array(p['gold']);c=x['sample_case'];i=x['position']
 assert np.array_equal(y,gold) and np.array_equal(c,p['cases']) and np.array_equal(i,p['positions'])
 target,g,old=source();w=arr(target)[:7];g=arr(g);old=arr(old)
 a=arr(load_file(str(wd/'roles-readout.safetensors'))['weight'])[:7]
 new=arr(load_file(str(wd/'roles-internal.safetensors'))[INTERNAL])
 allx=np.stack([z,h@a.T,normed(r+u@(new-old).T,g)@w.T])
 modes=['original','roles-readout','roles-internal']
 for idx,name in enumerate(modes):assert np.array_equal(allx[idx].argmax(-1),p['predictions'][name])
 sources={}
 for cid in np.unique(c):
  file=ROOT.parent/'pooler/shared-cases'/f'{cid}.json';sources[cid]=np.array(json.loads(file.read_text())['predictions']['unchanged'])
 inc=np.asarray([sources[cid][int(pos)] for cid,pos in zip(c,i)])
 inc_m,inc_s=stats(y,inc,c);rng=np.random.default_rng(760230);boots=rng.integers(0,16,(4000,16));inb=scores(inc_m[boots].sum(1))
 conf=softmax(allx,axis=-1);lab=conf.argmax(-1)
 agree=(lab==lab[0:1]).all(0)&(conf.max(-1)>=.80).all(0)
 fused=np.where(agree,lab[0],inc)
 def describe(pred):
  ms,ss=stats(y,pred,c);delta=scores(ms[boots].sum(1))-inb
  return dict(scores=ss,corrected_vs_incumbent=int(((pred==y)&(inc!=y)).sum()),broken_vs_incumbent=int(((pred!=y)&(inc==y)).sum()),changed_labels=int((pred!=inc).sum()),
              paired_macro_f1_change_vs_incumbent=ss['macro_f1']-inc_s['macro_f1'],paired_case_bootstrap95=np.quantile(delta,[.025,.975]).tolist())
 diagnostic={}
 for name,cc in zip(modes,lab):
  choose=np.where((inc!=y)&(cc==y),cc,inc)
  diagnostic[name]=describe(choose)
 out={'prespecification':sha(ROOT/'packed-fusion-diagnostic.json'),'incumbent':inc_s,'unanimous_precision_gate':describe(fused),'flagged_rows':int(agree.sum()),'complementary_correct_reference':diagnostic}
 savej(ROOT/'packed-complementarity.json',out)
 print('fusion',out['unanimous_precision_gate'],flush=True)
if __name__=='__main__':evaluate()
