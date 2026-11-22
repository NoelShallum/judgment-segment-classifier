"""Write InLegalBERT classifier rows analytically, then test longer document-level context."""
import argparse,json,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from safetensors.numpy import load_file,save_file
from scipy.linalg import eigh
import solve as lib
from legal_encode import H, MODELS,sha,check
E=lib.EYE
def read(kind,g):
    with np.load(H/(kind+'-'+g+'.npz'),allow_pickle=False) as z:return SimpleNamespace(**{k:z[k] for k in z.files})
def dump(name,o):
    f=H/name;p=f.with_suffix('.part');p.write_text(json.dumps(o,indent=2,allow_nan=False)+'\n');p.replace(f)
def write(name,params):
    save_file({k:np.ascontiguousarray(v) for k,v in params.items()},str(H/(name+'.safetensors')))
def get(name):return load_file(str(H/(name+'.safetensors')))
def original(kind):
    with np.load(H/(kind+'-source.npz')) as z:return z['w'],z['b']
def head_output(data,p):return data.hidden.astype(np.float32)@p['weight'].T+p['bias']
def equations(kind,d,idx,rare,sh=False):
    h=d.hidden[idx].astype(np.float32);y=d.gold[idx]
    mu=h.mean(0,dtype=np.float64).astype(np.float32);inv=(1/np.maximum(h.std(0,dtype=np.float64),.035)).astype(np.float32)
    x=np.column_stack(((h-mu)*inv,np.ones(len(h),np.float32)))
    t=E[y[np.random.default_rng(265131).permutation(len(y))] if sh else y]
    g,r=lib.gram(x,t,lib.role_weights(y,rare))
    return mu,inv,g,r
def solve_head(kind,eq,lam,gain):
    mu,inv,g,r=eq;w,b=original(kind)
    a=lib.ridge(g,r,lam,True);s=(a[:-1]*inv[:,None]).T;bias=a[-1]-mu@s.T
    # Fixed scale aligns one-hot ridge outputs with the author's logit scale.
    s*=6;bias*=6
    return {'weight':((1-gain)*w+gain*s).astype(np.float32),'bias':((1-gain)*b+gain*bias).astype(np.float32)}
def pair_head(d,kind,idx,rare=.25,lam=.1):
    h=d.hidden[idx].astype(np.float32);y=d.gold[idx];w,b=original(kind)
    old=d.raw[idx,5]-d.raw[idx,0]
    t=np.where(y==0,-4,np.where(y==5,4,old))-old
    mu=h.mean(0);inv=1/np.maximum(h.std(0),.035)
    x=np.column_stack(((h-mu)*inv,np.ones(len(h),np.float32)))
    g,r=lib.gram(x,t[:,None],lib.role_weights(y,rare))
    a=lib.ridge(g,r,lam,True).ravel()
    coeff=a[:-1]*inv;bias=a[-1]-mu@coeff
    ww=w.copy();bb=b.copy()
    for j,fac in ((0,-.5),(5,.5)):ww[j]+=fac*coeff;bb[j]+=fac*bias
    return {'weight':ww,'bias':bb}
def choose(kind,t,v):
    ix=np.arange(len(t.gold));best=None
    for rare in (0.,.40,.7):
        eq=equations(kind,t,ix,rare)
        for lam in (.035,.20):
            for gain in (.4,.85,1.):
                p=solve_head(kind,eq,lam,gain)
                m=lib.metric(v.gold,head_output(v,p).argmax(1));k=lib.scorekey(m)
                if best is None or k>best[0]:best=(k,p,dict(rare=rare,lam=lam,gain=gain),m)
    return best[1:]
def pca(d):
    h=d.hidden[::3].astype(np.float32);mu=h.mean(0);c=h-mu
    vals,vecs=eigh((c.T@c).astype(np.float64)/len(c),subset_by_index=[h.shape[1]-48,h.shape[1]-1])
    return {'mu':mu,'proj':(vecs/np.sqrt(np.maximum(vals,.012))).astype(np.float32)}
def fetch_group(g):
    d={k:read(k,g) for k in MODELS}
    for key in ('offsets','case_ids','gold','base'):np.testing.assert_array_equal(getattr(d['past'],key),getattr(d['both'],key))
    return d
def semantic(d,k,p):
    return np.clip((d[k].hidden.astype(np.float32)-p['mu'])@p['proj'],-5,5)
def basis(d,streams,pcs,mode):
    base=d['past'].base
    vals=[E[base]]
    if mode!='parent':
        for k in MODELS:
            vals.append(lib.probabilities(d[k].raw*.65))
            if mode in ('edited','semantic'):vals.append(lib.probabilities(streams[k]*.80))
    values=np.column_stack(vals);blocks=[]
    vectors=np.column_stack([semantic(d,k,pcs[k]) for k in MODELS]) if mode=='semantic' else None
    for a,b in zip(d['past'].offsets,d['past'].offsets[1:]):
        v=values[a:b];n=len(v);ix=np.arange(n);z=((ix+.5)/n)[:,None]
        pre=np.vstack((np.zeros((1,v.shape[1])),v.cumsum(0,dtype=np.float64)));parts=[v,lib.coords(n)]
        for band in (1,4,14,45):
            lo=np.maximum(ix-band,0);hi=np.minimum(ix+band+1,n)
            parts.extend(((pre[ix]-pre[lo])/np.maximum(ix-lo,1)[:,None],(pre[hi]-pre[ix+1])/np.maximum(hi-ix-1,1)[:,None]))
        g=v.mean(0)
        parts.extend((np.broadcast_to(g,(n,len(g))),v*g[None,:],z*v,z*z*v))
        if mode!='parent':
            parts.append((v[:,:7,None]*v[:,None,7:14]).reshape(n,-1))
        if vectors is not None:
            h=vectors[a:b];hp=np.vstack((np.zeros((1,h.shape[1])),h.cumsum(0,dtype=np.float64)));hs=[h]
            for band in (3,15):
                lo=np.maximum(ix-band,0);hi=np.minimum(ix+band+1,n)
                hs.extend(((hp[ix]-hp[lo])/np.maximum(ix-lo,1)[:,None],(hp[hi]-hp[ix+1])/np.maximum(hi-ix-1,1)[:,None]))
            parts+=hs
        blocks.append(np.column_stack(parts).astype(np.float32))
    return np.concatenate(blocks)
def fit_doc(t,streams,v,vs,pcs,mode):
    x,tp=lib.docprep(basis(t,streams,pcs,mode));xv=lib.docbasis(basis(v,vs,pcs,mode),tp)
    y=t['past'].gold;b=v['past'].base;truth=v['past'].gold
    baseline=lib.metric(truth,b);eligible=[];allc=[];coefs={}
    for rare in (0.,.35,.60):
        g,r=lib.gram(x,E[y],lib.role_weights(y,rare))
        for lam in (.025,.18):
            coef=lib.ridge(g,r,lam,True);e=xv@coef;coefs[(rare,lam)]=coef
            for anchor in (0.,.06,.14,.28):
                for scope in ('all','pair'):
                    for margin in (0.,.05):
                        p=lib.decide(e,b,anchor,margin,scope);m=lib.metric(truth,p);c=dict(rare=rare,lam=lam,anchor=anchor,scope=scope,margin=margin)
                        entry=(lib.scorekey(m),c,m);allc.append(entry)
                        if m['pair_errors']<=baseline['pair_errors']:eligible.append(entry)
    _,c,m=max(eligible or allc,key=lambda q:q[0])
    tp['coef']=coefs[c['rare'],c['lam']]
    ys=y[np.random.default_rng(412005).permutation(len(y))]
    g,r=lib.gram(x,E[ys],lib.role_weights(y,c['rare']))
    sh=lib.ridge(g,r,c['lam'],True);sh[:-1]*=np.linalg.norm(tp['coef'][:-1])/max(np.linalg.norm(sh[:-1]),1e-10);tp['shufcoef']=sh
    return tp,c,m
def fit():
    t=fetch_group('fit');v=fetch_group('select')
    a=t['past'];fold=np.repeat(np.arange(len(a.case_ids))%3,np.diff(a.offsets));ix=np.arange(len(a.gold))
    cfg={};selector={};pcs={};streams={};vs={}
    for kind in MODELS:
        p,c,m=choose(kind,t[kind],v[kind]);write(kind+'-edited',p);cfg[kind]=c;selector[kind]=m
        ep=pair_head(t[kind],kind,ix);write(kind+'-pair',ep)
        neg=solve_head(kind,equations(kind,t[kind],ix,c['rare'],True),c['lam'],c['gain']);w,b=original(kind)
        ratio=np.linalg.norm(p['weight']-w)/max(np.linalg.norm(neg['weight']-w),1e-8)
        neg={'weight':w+ratio*(neg['weight']-w),'bias':b+ratio*(neg['bias']-b)};write(kind+'-shuffled',neg)
        pcs[kind]=pca(t[kind]);write(kind+'-pca',pcs[kind])
        streams[kind]=np.empty((len(ix),7),np.float32);vs[kind]=head_output(v[kind],p)
        for f in range(3):
            p=solve_head(kind,equations(kind,t[kind],ix[fold!=f],c['rare']),c['lam'],c['gain'])
            stream=t[kind].hidden[fold==f].astype(np.float32)@p['weight'].T+p['bias']
            streams[kind][fold==f]=stream
        print('legal head',kind,c,m['macro_f1'],flush=True)
    for mode in ('parent','authors','edited','semantic'):
        p,c,m=fit_doc(t,streams,v,vs,pcs,mode);write('doc-'+mode,p);cfg['doc-'+mode]=c;selector['doc-'+mode]=m
        print('legal document',mode,c,m['macro_f1'],m['pair_errors'],flush=True)
    incumbent=lib.metric(v['past'].gold,v['past'].base)
    options=[k for k in ('doc-edited','doc-semantic') if selector[k]['pair_errors']<=incumbent['pair_errors']]
    chosen=max(options or ('doc-edited','doc-semantic'),key=lambda k:lib.scorekey(selector[k]))
    files={p.name:sha(p) for p in H.glob('*.safetensors') if 'donor' not in p.name}
    dump('freeze.json',dict(selected=chosen,cfg=cfg,selection=selector,baseline=incumbent,files=files,inputs={k:sha(H/(k+'-fit.npz')) for k in MODELS},protocol='Fit on 192 whole train judgments. Select on 48. Three-fold case crossfit for features written from labels. Comparison with the fixed incumbent, both unchanged donors, five-role protection and shuffled control.'))
def predictions(d,lock):
    for p,h in lock['files'].items():assert sha(H/p)==h
    out={'incumbent':d['past'].base};edited={};pc={}
    for k in MODELS:
        out[k+'-original']=d[k].raw.argmax(1);pc[k]=get(k+'-pca');edited[k]=head_output(d[k],get(k+'-edited'))
        for mode in ('edited','pair','shuffled'):out[k+'-'+mode]=head_output(d[k],get(k+'-'+mode)).argmax(1)
    for mode in ('parent','authors','edited','semantic'):
        name='doc-'+mode;p=get(name);c=lock['cfg'][name];x=lib.docbasis(basis(d,edited,pc,mode),p)
        for key,mat in (('', 'coef'),('-shuffled','shufcoef')):
            out[name+key]=lib.decide(x@p[mat],d['past'].base,c['anchor'],c['margin'],c['scope'])
    return out
def evaluate(groups):
    lock=json.loads((H/'freeze.json').read_text());results={}
    for g in groups:
        d=fetch_group(g);out=predictions(d,lock);truth=d['past'];k=lock['selected'];p=out[k]
        scores={name:lib.metric(truth.gold,v) for name,v in out.items()}
        if g=='select':
            for mode in ('parent','authors','edited','semantic'):assert scores['doc-'+mode]['confusion']==lock['selection']['doc-'+mode]['confusion']
        results[g]=dict(cases=len(truth.case_ids),rows=len(truth.gold),scores=scores,ci={q:lib.delta_ci(truth,p,out[q]) for q in ['incumbent','doc-authors','doc-parent',k+'-shuffled']},
                        fixed=int(((p==truth.gold)&(truth.base!=truth.gold)).sum()),broken=int(((p!=truth.gold)&(truth.base==truth.gold)).sum()))
        np.savez_compressed(H/(g+'-predictions.npz'),case_ids=truth.case_ids,offsets=truth.offsets,gold=truth.gold,**out)
        print('legal scores',g,scores[k]['macro_f1'],scores['incumbent']['macro_f1'],scores[k]['pair_errors'],scores['incumbent']['pair_errors'],flush=True)
    dump('official.json' if 'test' in groups else 'evaluation.json',dict(selected=lock['selected'],freeze_sha256=sha(H/'freeze.json'),groups=results))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=('fit','score','official'));a=ap.parse_args();check()
    if a.phase=='fit':fit()
    else:evaluate(['test'] if a.phase=='official' else ['select','confirm','fresh'])
