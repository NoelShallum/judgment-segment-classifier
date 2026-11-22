"""Program a native Qwen MLP and a separate contextual readout, locking choices before confirmation."""
import argparse, hashlib, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve
from safetensors.numpy import load_file, save_file

H=Path(__file__).resolve().parent
ROLES=['None','Facts','Issue','Petitioner argument','Respondent argument','Reasoning','Decision']
EYE=np.eye(7,dtype=np.float32)

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def savej(name,item):
    p=H/name;stage=p.with_suffix('.tmp');stage.write_text(json.dumps(item,indent=2,allow_nan=False)+'\n');stage.replace(p)
def load(name):
    with np.load(H/(name+'-features.npz'),allow_pickle=False) as f:return SimpleNamespace(**{k:f[k] for k in f.files})
def write_weights(name,w):
    vals={k:np.ascontiguousarray(v) for k,v in w.items()}
    save_file(vals,str(H/name))
    for k,v in load_file(str(H/name)).items():np.testing.assert_array_equal(v,vals[k])
def metric(y,p):
    c=np.bincount(y.astype(np.int64)*7+p.astype(np.int64),minlength=49).reshape(7,7)
    f=2*c.diagonal()/np.maximum(c.sum(0)+c.sum(1),1)
    return dict(macro_f1=float(f.mean()),accuracy=float(c.trace()/c.sum()),pair_errors=int(c[0,5]+c[5,0]),none_to_reason=int(c[0,5]),reason_to_none=int(c[5,0]),roles=f.tolist(),confusion=c.tolist())
def delta_ci(data,p,other):
    def block(q):
        return np.stack([np.bincount(data.gold[a:b].astype(int)*7+q[a:b],minlength=49).reshape(7,7) for a,b in zip(data.offsets,data.offsets[1:])])
    a,b=block(p),block(other)
    ix=np.random.default_rng(91034).integers(0,len(a),(2500,len(a)))
    def f(mat):
        return (2*np.diagonal(mat,axis1=-2,axis2=-1)/np.maximum(mat.sum(-1)+mat.sum(-2),1)).mean(-1)
    u,v=a[ix].sum(1),b[ix].sum(1)
    return {'macro_delta':np.quantile(f(u)-f(v),[.025,.975]).tolist(),'pair_delta':np.quantile(u[:,0,5]+u[:,5,0]-v[:,0,5]-v[:,5,0],[.025,.975]).tolist()}
def scorekey(m):return (m['macro_f1'],-m['pair_errors'],m['accuracy'])
def hw(x):return torch.as_tensor(x,dtype=torch.float32,device='cuda')
def gram(x,t,weights):
    xt=hw(x);yt=hw(t);w=hw(weights)[:,None]
    g=(xt.T@(xt*w)).cpu().numpy().astype(np.float64); rhs=(xt.T@(yt*w)).cpu().numpy().astype(np.float64)
    return (g+g.T)*.5,rhs
def ridge(g,r,lam,intercept=False):
    diag=np.maximum(np.diag(g),1)/1.
    if intercept:diag[-1]=max(diag[-1]*.001,.00001)
    return cho_solve(cho_factor(g+np.diag(diag*lam),check_finite=False),r,check_finite=False).astype(np.float32)
def role_weights(y,rare):
    w=(np.bincount(y,minlength=7).clip(min=1)**(-rare))[y];return (w/w.mean()).astype(np.float32)
def rowcenter(x):return x-x.mean(1,keepdims=True)
def probabilities(x):
    q=np.exp(np.clip(x-x.max(1,keepdims=True),-36,0));return q/q.sum(1,keepdims=True)
def native_response(d,params,kind):
    raw=rowcenter(d.raw)/params['amplitude'][0]
    if kind=='source':return raw
    if kind=='head':return d.hidden.astype(np.float32)@params['coef']
    return raw+(d.z.astype(np.float32)@params['coef'])/params['amplitude'][0]
def native_equations(d,kept,kind,rare,shuffle):
    y=d.gold[kept];raw=rowcenter(d.raw[kept])
    amp=max(float(np.sqrt(np.mean(raw*raw))),.01)
    x=(d.hidden if kind=='head' else d.z)[kept].astype(np.float32)
    inv=1/np.maximum(np.sqrt(np.mean(x*x,axis=0,dtype=np.float64)),1e-5)
    rhs_label=y[np.random.default_rng(72641).permutation(len(y))] if shuffle else y
    if kind=='pair':
        direction=np.zeros(7,np.float32);direction[[0,5]]=[-.5,.5]
        # Other roles are protected in the fitting targets, and five output differences are fixed by construction.
        target=np.where(rhs_label==5,amp,np.where(rhs_label==0,-amp,raw[:,5]-raw[:,0]))
        residual=(target-(raw[:,5]-raw[:,0]))[:,None]*direction
    elif kind=='full':residual=amp*(EYE[rhs_label]-1/7)-raw
    else:residual=EYE[rhs_label]-1/7
    g,r=gram(x*inv.astype(np.float32),residual.astype(np.float32),role_weights(y,rare))
    return g,r,inv.astype(np.float32),amp
def native_params(eq,lam,gain):
    g,r,inv,amp=eq
    return {'coef':(ridge(g,r,lam)*inv[:,None])*gain,'amplitude':np.array([amp],np.float32)}
def normalize_control(target,negative,kind):
    a,b=target['coef'],negative['coef']
    if kind=='head':
        ratio=np.linalg.norm(a)/max(np.linalg.norm(b),1e-10)
    else:
        with np.load(H/'metadata.npz') as f: p=(f['prototypes']*f['norm'][None,:]).astype(np.float64)
        dual=np.linalg.pinv(p)
        ratio=np.linalg.norm(dual@a.T)/max(np.linalg.norm(dual@b.T),1e-10)
    negative['coef']*=np.float32(ratio);return negative
def choose_native(train,val,kind):
    ids=np.arange(len(train.gold));best=None
    for rare in (0.,.40):
        eq=native_equations(train,ids,kind,rare,False)
        for lam in (.04,.30):
            for gain in ((1.,) if kind=='head' else (.65,1.)):
                p=native_params(eq,lam,gain);res=metric(val.gold,native_response(val,p,kind).argmax(1))
                key=scorekey(res)
                if best is None or key>best[0]:best=(key,p,dict(rare=rare,lam=lam,gain=gain),res)
    return best[1:]
def fit_native_cfg(train,ix,kind,cfg,shuffle=False):
    return native_params(native_equations(train,ix,kind,cfg['rare'],shuffle),cfg['lam'],cfg['gain'])
def coords(n):
    i=np.arange(n,dtype=np.float32);t=(i+.5)/n
    return np.column_stack((t,t*t,np.sqrt(t),np.sqrt(1-t),1/(i+1),1/(n-i),np.full(n,np.log1p(n)/6),np.sin(np.pi*t),np.cos(np.pi*t)))
def design(d,streams,mode):
    native,head,pair=streams['full'],streams['head'],streams['pair']
    cols=[EYE[d.base]]
    if mode in ('native','combined'):cols+=[probabilities(5*native),np.tanh(((pair[:,5]-pair[:,0])/2))[:,None]]
    if mode in ('head','combined'):cols+=[probabilities(5*head)]
    s=np.column_stack(cols).astype(np.float32); out=[]
    for a,b in zip(d.offsets,d.offsets[1:]):
        v=s[a:b];n=len(v);ix=np.arange(n);p=np.vstack([np.zeros((1,v.shape[1])),np.cumsum(v,axis=0,dtype=np.float64)])
        pieces=[v,coords(n)]
        for width in (1,4,14,48):
            left=(ix-width).clip(min=0); right=(ix+width+1).clip(max=n)
            l=(p[ix]-p[left])/np.maximum(ix-left,1)[:,None]
            r=(p[right]-p[ix+1])/np.maximum(right-ix-1,1)[:,None]
            pieces.extend((l,r))
        cohort=np.broadcast_to(v.mean(0),(n,v.shape[1]))
        t=((ix+.5)/n)[:,None]
        pieces.extend((cohort,t*v,t*t*v,v*cohort))
        if v.shape[1]>7:
            pieces.append((v[:,:7,None]*v[:,None,7:14]).reshape(n,-1))
        out.append(np.column_stack(pieces).astype(np.float32))
    return np.concatenate(out)
def docbasis(x,params):
    v=np.clip((x-params['center'])*params['scale'],-6,6)
    rand=np.maximum(v@params['rand']+params['rbias'],0)
    return np.column_stack((v,rand,np.ones(len(x),np.float32)))
def docprep(x):
    mu=x.mean(0,dtype=np.float64).astype(np.float32); inv=1/np.maximum(x.std(0,dtype=np.float64).astype(np.float32),.03)
    rng=np.random.default_rng(2026389)
    mat=(rng.standard_normal((len(mu),384))/np.sqrt(len(mu))).astype(np.float32)
    off=rng.uniform(-.6,.6,384).astype(np.float32)
    params={'center':mu,'scale':inv,'rand':mat,'rbias':off}
    return docbasis(x,params),params
def decide(response,base,anchor,margin,scope):
    z=response+anchor*EYE[base];pred=z.argmax(1).astype(np.int8)
    if margin:pred=np.where(z[np.arange(len(base)),pred]-z[np.arange(len(base)),base]>=margin,pred,base)
    if scope=='pair':pred=np.where(np.isin(base,[0,5])&np.isin(pred,[0,5]),pred,base)
    return pred.astype(np.int8)
def select_doc(d,fit_streams,v,val_streams,mode):
    train_x,params=docprep(design(d,fit_streams,mode))
    val_x=docbasis(design(v,val_streams,mode),params)
    baseline=metric(v.gold,v.base)
    candidates=[]
    cache={}
    for rare in (.0,.35,.60):
        g,r=gram(train_x,EYE[d.gold],role_weights(d.gold,rare))
        for lam in (.025,.20):
            coef=ridge(g,r,lam,True);resp=val_x@coef
            cache[(rare,lam)]=coef
            for anchor in (.0,.08,.18,.32):
                for margin in (.0,.08):
                    for scope in ('all','pair'):
                        pred=decide(resp,v.base,anchor,margin,scope)
                        m=metric(v.gold,pred)
                        candidates.append((scorekey(m),m,dict(rare=rare,lam=lam,anchor=anchor,margin=margin,scope=scope)))
    eligible=[x for x in candidates if x[1]['pair_errors']<=baseline['pair_errors']]
    key,score,cfg=max(eligible or candidates,key=lambda x:x[0])
    coef=cache[(cfg['rare'],cfg['lam'])]
    params['coef']=coef
    y=d.gold[np.random.default_rng(260835).permutation(len(d.gold))]
    g,r=gram(train_x,EYE[y],role_weights(d.gold,cfg['rare']))
    sh=ridge(g,r,cfg['lam'],True)
    # Match the feature-weight norm. Keep the shuffled fit's own class intercept.
    sh[:-1]*=np.linalg.norm(coef[:-1])/max(np.linalg.norm(sh[:-1]),1e-10)
    params['shufcoef']=sh
    return params,cfg,score
def native_set(d,p):return {k:native_response(d,v,k) for k,v in p.items()}
def fitted():
    t,v=load('fit'),load('select');assert set(t.case_ids).isdisjoint(v.case_ids)
    params={}; cfg={}; selector={};foldviews={k:np.zeros((len(t.gold),7),np.float32) for k in ('head','pair','full')}
    allids=np.arange(len(t.gold)); docfold=np.array([i%3 for i in range(len(t.case_ids))]); fold=np.repeat(docfold,np.diff(t.offsets))
    for kind in ('full','pair','head'):
        p,c,score=choose_native(t,v,kind)
        params[kind]=p;cfg[kind]=c;selector[kind]=score
        write_weights(kind+'.safetensors',p)
        control=normalize_control(p,fit_native_cfg(t,allids,kind,c,True),kind)
        write_weights(kind+'-shuffled.safetensors',control)
        for f in range(3):
            keep=np.flatnonzero(fold!=f);test=np.flatnonzero(fold==f);k=fit_native_cfg(t,keep,kind,c)
            sub=SimpleNamespace(**{q:getattr(t,q)[test] for q in ('hidden','z','raw')})
            foldviews[kind][test]=native_response(sub,k,kind)
        print('fit native',kind,c,score['macro_f1'],flush=True)
    valviews=native_set(v,params)
    modes=('parent','head','native','combined')
    for mode in modes:
        p,c,s=select_doc(t,foldviews,v,valviews,mode)
        write_weights('doc-'+mode+'.safetensors',p);cfg['doc-'+mode]=c;selector['doc-'+mode]=s
        print('fit document',mode,c,s['macro_f1'],s['pair_errors'],flush=True)
    eligible=[m for m in ('native','combined') if selector['doc-'+m]['pair_errors']<=metric(v.gold,v.base)['pair_errors']]
    chosen=max(eligible or ['native','combined'],key=lambda m:scorekey(selector['doc-'+m]))
    files={p.name:sha(p) for p in H.glob('*.safetensors')}
    savej('frozen.json',dict(selected=chosen,config=cfg,files=files,selector=selector,baseline=metric(v.gold,v.base),
                            split_hashes={k:sha(H/(k+'-features.npz')) for k in ('fit','select')},
                            rule='Select the highest 7-role macro-F1 among edited native or combined document candidates whose selector pair swaps are at or below the incumbent, when available.',
                            comparisons=['Document-aware native edit only','Frozen-backbone head only','Parent only','Combined','Matched-norm shuffled weight controls'],
                            protocol_sha256=sha(H/'protocol.json'),stages='Three case-wise crossfit partitions construct fit-only scores for the document readout. One selection on the 48 judgments. Confirmations open after lock.'))
def predict(d,lock):
    for path,digest in lock['files'].items():assert sha(H/path)==digest
    params={k:load_file(str(H/(k+'.safetensors'))) for k in ('full','pair','head')}
    raw=native_set(d,params); out={'incumbent':d.base,'donor':d.raw.argmax(1)}
    for k in params:
        out['standalone-'+k]=raw[k].argmax(1)
        sh=load_file(str(H/(k+'-shuffled.safetensors')))
        out['standalone-'+k+'-shuffled']=native_response(d,sh,k).argmax(1)
    for mode in ('parent','head','native','combined'):
        name='doc-'+mode
        p=load_file(str(H/(name+'.safetensors')));c=lock['config'][name]
        x=docbasis(design(d,raw,mode),p)
        for sn,col in [('', 'coef'),('-shuffled','shufcoef')]:
            out[name+sn]=decide(x@p[col],d.base,c['anchor'],c['margin'],c['scope'])
    return out
def evaluate():
    lock=json.loads((H/'frozen.json').read_text())
    summary={}
    for g in ('select','confirm','fresh'):
        d=load(g);o=predict(d,lock);target='doc-'+lock['selected']
        scores={k:metric(d.gold,p) for k,p in o.items()}
        if g=='select':
            for a in ('parent','head','native','combined'):
                assert scores['doc-'+a]['confusion']==lock['selector']['doc-'+a]['confusion']
        p=o[target];n=len(p)
        ci={k:delta_ci(d,p,o[k]) for k in ('incumbent','doc-parent','doc-head',target+'-shuffled')}
        summary[g]={'cases':len(d.case_ids),'rows':n,'scores':scores,'CIs':ci,'fixes':int(((p==d.gold)&(d.base!=d.gold)).sum()),'regressions':int(((p!=d.gold)&(d.base==d.gold)).sum())}
        np.savez_compressed(H/(g+'-predictions.npz'),case_ids=d.case_ids,offsets=d.offsets,gold=d.gold,**o)
        print('score',g,scores[target]['macro_f1'],scores['incumbent']['macro_f1'],scores[target]['pair_errors'],scores['incumbent']['pair_errors'],flush=True)
    savej('evaluation.json',dict(frozen_sha256=sha(H/'frozen.json'),selected=lock['selected'],groups=summary))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=('fit','score'));args=ap.parse_args()
    if args.phase=='fit':
        from encode_context import verify_t4
        verify_t4();torch.set_num_threads(1);fitted()
    else:evaluate()
