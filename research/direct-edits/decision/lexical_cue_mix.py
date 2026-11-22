"""Test whether static target-token evidence complements the existing contextual answer cue without another encoder pass."""
import json
import numpy as np
from safetensors.torch import load_file
from readout_study import banks,solve,DIR
from weights import source,arr,metrics,savej
from compare import stats,scores
from fetch import ROOT,sha
def centered(x):return x-x.mean(1,keepdims=True)
def normalized_logits(x,W,scale):
 return centered(x@W.T)/scale
def run():
 train=banks('train');val=banks('val');t=source()[0]
 group=train['cases'];y=train['y'];target=val['y'];cases=val['cases'];base=arr(t)[:7]
 gids=list(dict.fromkeys(group));ng=np.array([gids.index(c)%3 for c in group]);alphas=(0.,.25,.5,.75,1.);results={a:np.zeros(len(y),int) for a in alphas}
 for fold in range(3):
  fit=ng!=fold;evals=[]
  for k in ('cue','lexical'):
   w=arr(solve(train[k][fit],t,y[fit]))[:7]
   scale=max(float(np.std(centered(train[k][fit]@w.T))),1e-8)
   evals.append(normalized_logits(train[k][~fit],w,scale))
  for a in alphas:results[a][~fit]=(a*evals[0]+(1.-a)*evals[1]).argmax(1)
 grid={a:metrics(y,r) for a,r in results.items()};chosen=max(alphas,key=lambda a:(grid[a]['macro_f1'],grid[a]['accuracy'],a))
 current=ROOT/'weights/ov-GPU-f32/packed-ordered/whole16/roles-readout.safetensors';lex=DIR/'static-tokens/roles-readout.safetensors'
 Wc=arr(load_file(str(current))['weight'])[:7];Wt=arr(load_file(str(lex))['weight'])[:7];sc=[float(np.std(centered(train[k]@w.T))) for k,w in zip(('cue','lexical'),(Wc,Wt))]
 recipe={'diagnostic':'Adaptive three-training-fold selection on matched output scores from two pre-existing directly written heads, gradient-free; no shared validation labels select the weight.',
        'fit_rows':len(y),'fit_fold_grouped_by_judgment':dict(zip(gids,[gids.index(c)%3 for c in gids])),'grid':{str(a):{'macro_f1':m['macro_f1'],'accuracy':m['accuracy']} for a,m in grid.items()},'cue_fraction':chosen,'scales':sc,'source_checkpoints':{'cue':sha(current),'target_tokens':sha(lex)}}
 dest=DIR/'cue-lexical-sum';dest.mkdir(exist_ok=True);savej(dest/'fit.json',recipe)
 uc=normalized_logits(val['cue'],Wc,sc[0]);ut=normalized_logits(val['lexical'],Wt,sc[1]);p=(chosen*uc+(1.-chosen)*ut).argmax(1);pure=uc.argmax(1)
 a,sa=stats(target,p,cases);b,sb=stats(target,pure,cases);rng=np.random.default_rng(760230);boots=rng.integers(0,16,(4000,16))
 result={'metrics':sa,'same_row_contextual_cue':sb,'paired_macro_f1_delta_95':np.quantile(scores(a[boots].sum(1))-scores(b[boots].sum(1)),[.025,.975]).tolist(),'fixed_vs_cue':int(((pure!=target)&(p==target)).sum()),'broken_vs_cue':int(((pure==target)&(p!=target)).sum())}
 savej(dest/'validation.json',result);print('mixed target/cue',chosen,sc,sa['macro_f1'],result['paired_macro_f1_delta_95'],flush=True)
if __name__=='__main__':run()
