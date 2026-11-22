"""Compare case-conditioned output weights with a context-matched prior-free control."""
import argparse,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from safetensors.numpy import load_file,save_file
from case_style import H,R,LIVE,PLAN,EYE,BINS,sha,save
from importlib.util import spec_from_file_location,module_from_spec
def refs():
    spec=spec_from_file_location('frozen_metrics',H.parent/'context-memory/solve.py')
    m=module_from_spec(spec);spec.loader.exec_module(m);return m
measure=refs()
def bank(group):
    groups=PLAN['groups'];cases=groups[group]
    rows=[]
    folder=LIVE/('reviewer-round2/new-cases' if group=='fresh' else 'bert-t4/cases')
    with np.load(H.parent/'t4-return'/(group+'.npz')) as d:
        np.testing.assert_array_equal(d['case_ids'],cases)
        base=d['reference'];offs=d['offsets'];rstart=d['review_offsets'];scores=d['scores']
    for j,k in enumerate(cases):
        with np.load(folder/(k+'.npz'),allow_pickle=False) as z:
            assert z['case_sha256'].item()==PLAN['case_text_sha256'][k]
            a,b=offs[j:j+2];u,v=rstart[j:j+2]
            c=np.zeros((b-a,9),np.float32);loc=z['positions'];c[loc,0]=1;c[loc,1:8]=scores[u:v];c[loc,8]=z['prior'].max(1)
            rows.append((c,z['gold']))
    return SimpleNamespace(case_ids=cases,offsets=offs,base=base,gold=np.concatenate([r[1] for r in rows]),reviews=np.concatenate([r[0] for r in rows]))
def design(d,kind):
    with np.load(H/'profiles.npz') as z:
        mapping={k:i for i,k in enumerate(z['case_ids'])}
        prof={key:z[key] for key in ('ridge','nearest','shuffled','global_prior','doc_style')}
    acc=[]
    for id_,a,b in zip(d.case_ids,d.offsets,d.offsets[1:]):
        n=b-a;ix=np.arange(n);p=(ix+.5)/n;bin_ix=np.minimum((p*BINS).astype(int),BINS-1);idx=mapping[id_]
        old=EYE[d.base[a:b]];r=d.reviews[a:b]
        vals=[old,r,measure.coords(n)]
        cs=np.vstack([np.zeros((1,7)),old.cumsum(0,dtype=np.float64)])
        parts=[]
        for band in (2,7,24):
            lo=np.maximum(ix-band,0);hi=np.minimum(ix+band+1,n)
            l=(cs[ix]-cs[lo])/np.maximum(ix-lo,1)[:,None]
            f=(cs[hi]-cs[ix+1])/np.maximum(hi-ix-1,1)[:,None]
            parts.extend([l,f])
        vals+=parts+[np.broadcast_to(old.mean(0),(n,7)),old*p[:,None]]
        if kind!='context':
            for typ in (('ridge','nearest') if kind=='case' else ('shuffled',)):
                total=prof[typ][idx];glob=prof['global_prior'][bin_ix];v=total[bin_ix]
                vals += [v,v-glob,v*old,parts[2]*v,np.broadcast_to(total.mean(0),(n,7))]
            stat=prof['doc_style'][idx];vals.extend([np.broadcast_to(stat,(n,len(stat)))])
        acc.append(np.column_stack(vals).astype(np.float32))
    return np.concatenate(acc)
def basis(x,p):
    n=np.clip((x-p['mu'])*p['inv'],-6,6)
    return np.column_stack((n,np.maximum(n@p['rand']+p['rb'],0),np.ones(len(n),np.float32)))
def fit():
    t,v=bank('fit'),bank('select')
    old=measure.metric(v.gold,v.base);lock={};scores={}
    for kind in ('context','case','shuffled-profile'):
        x=design(t,kind);rng=np.random.default_rng(202611225)
        p={'mu':x.mean(0,dtype=np.float64).astype(np.float32),'inv':(1/np.maximum(x.std(0,dtype=np.float64),.03)).astype(np.float32),'rand':(rng.normal(size=(x.shape[1],512))/np.sqrt(x.shape[1])).astype(np.float32),'rb':rng.uniform(-.5,.5,512).astype(np.float32)}
        x=basis(x,p);vx=basis(design(v,kind),p)
        best=None;fallback=None;tmp={}
        for rare in (.0,.35,.65):
            w=measure.role_weights(t.gold,rare)
            g=x.T@(x*w[:,None]);r=x.T@(EYE[t.gold]*w[:,None])
            for lam in (.025,.20):
                coef=measure.ridge(g.astype(np.float64),r.astype(np.float64),lam,True);resp=vx@coef
                tmp[(rare,lam)]=coef
                for anchor in (.0,.10,.24,.42):
                    for scope in ('pair','all'):
                        pred=measure.decide(resp,v.base,anchor,0,scope);m=measure.metric(v.gold,pred)
                        rec=(measure.scorekey(m),dict(rare=rare,lam=lam,anchor=anchor,scope=scope),m)
                        if fallback is None or rec[0]>fallback[0]:fallback=rec
                        if m['pair_errors']<=old['pair_errors'] and (best is None or rec[0]>best[0]):best=rec
        _,cfg,sc=best or fallback;p['coef']=tmp[(cfg['rare'],cfg['lam'])]
        file=H/(kind+'.safetensors');save_file({k:np.ascontiguousarray(val) for k,val in p.items()},str(file))
        lock[kind]=dict(config=cfg,sha256=sha(file));scores[kind]=sc
        print('selected',kind,cfg,sc['macro_f1'],sc['pair_errors'],flush=True)
    save('freeze.json',dict(models=lock,selection=scores,baseline=old,profile_sha256=sha(H/'profiles.npz'),note='Ridge on 384 train cases, select on 48. All nonblank sentences. No iterative gradient updates.'))
def score():
    lock=json.loads((H/'freeze.json').read_text());assert lock['profile_sha256']==sha(H/'profiles.npz');results={}
    for g in ('select','confirm','fresh'):
        d=bank(g);pred={'incumbent':d.base}
        for kind,v in lock['models'].items():
            path=H/(kind+'.safetensors');assert sha(path)==v['sha256']
            w=load_file(str(path));c=v['config'];out=basis(design(d,kind),w)@w['coef'];pred[kind]=measure.decide(out,d.base,c['anchor'],0,c['scope'])
        o={k:measure.metric(d.gold,v) for k,v in pred.items()}
        if g=='select':
            for k in lock['models']:assert o[k]['confusion']==lock['selection'][k]['confusion']
        results[g]=dict(cases=len(d.case_ids),rows=len(d.gold),scores=o,confidence={k:measure.delta_ci(d,pred['case'],v) for k,v in pred.items() if k!='case'})
        print(g,{k:(round(v['macro_f1']*100,3),v['pair_errors']) for k,v in o.items()},flush=True)
        np.savez_compressed(H/(g+'-predictions.npz'),case_ids=d.case_ids,offsets=d.offsets,gold=d.gold,**pred)
    save('evaluation.json',dict(frozen_sha256=sha(H/'freeze.json'),groups=results))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=('fit','score'));a=ap.parse_args()
    (fit if a.phase=='fit' else score)()
