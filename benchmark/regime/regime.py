"""Program case-conditioned emission and transition weights from training counts."""
import ctypes
import json
import sys
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from safetensors.numpy import load_file, save_file
import run_scale as scale
import solve as lib
from legal_encode import check, sha

H = Path(__file__).resolve().parent
S = scale.H
R = 6
BINS = 49 * 4 * 6 * 3

def normalize(x):
    return x / np.maximum(x.sum(-1, keepdims=True), 1e-12)

def statistics(x):
    p, q = [lib.probabilities(.65*x[:,k:k+7]) for k in (0,62)]
    w, v = p.argmax(1), q.argmax(1)
    best = ((p+q)/2).argmax(1)
    n = len(p); pos = np.minimum(5,(np.arange(n)*6)//n)
    certainty = (p.max(1)>.68).astype(int)*2+(q.max(1)>.68)
    left = np.r_[best[0],best[:-1]] == best
    right = np.r_[best[1:],best[-1]] == best
    code = (((w*7+v)*4+certainty)*6+pos)*3 + left.astype(int)+right.astype(int)
    local = np.column_stack([p,q,(p+q)/2,lib.EYE[best]])
    sections = np.array_split(local,min(4,n))
    quarters = [sections[min(i,len(sections)-1)].mean(0) for i in range(4)]
    f = np.concatenate([local.mean(0),local.std(0),*quarters,
                        np.bincount(w*7+v,minlength=49)/n,
                        [np.log1p(n)/8,(w==v).mean(),left.mean()]])
    return code, best, f

def profile(y, best):
    hist=np.bincount(y,minlength=7)
    err=np.bincount(best*7+y,minlength=49).reshape(7,7)
    tr=np.bincount(y[:-1]*7+y[1:],minlength=49).reshape(7,7)
    return np.r_[2*np.sqrt(normalize(hist+.1)),np.sqrt((err+.4)/(hist.sum()+19.6)).flatten(),
                 .5*np.sqrt(normalize(tr+2)).flatten()]

def cluster(x):
    rng=np.random.default_rng(260516)
    center=[x[rng.integers(len(x))]];dist=((x-center[0])**2).sum(1)
    for _ in range(R-1):
        center.append(x[rng.choice(len(x),p=dist/dist.sum())]);dist=np.minimum(dist,((x-center[-1])**2).sum(1))
    center=np.stack(center)
    for _ in range(18):
        labels=((x[:,None]-center[None])**2).sum(2).argmin(1)
        center=np.stack([x[labels==k].mean(0) if (labels==k).any() else center[k] for k in range(R)])
    return labels

def fit(n):
    records=[]
    for i in range(n):
        d=scale.readbank(S/f'train-{i:02d}.npz')
        for a,b in zip(d.offsets,d.offsets[1:]):
            co,w,f=statistics(d.x[a:b]); y=d.y[a:b].astype(np.int64)
            records.append((co,y,f,profile(y,w)))
    x=np.stack([r[2] for r in records]); labs=cluster(np.stack([r[3] for r in records]))
    mu=x.mean(0); sd=np.maximum(x.std(0),.035); z=np.c_[np.clip((x-mu)/sd,-4,4),np.ones(len(x))]
    g=z.T@z;g.flat[::len(g)+1]+=.10*len(z)
    out=dict(mu=mu,sd=sd)
    assignments=[labs,np.random.default_rng(482903).permutation(labs)]
    for m,labels in zip(('learned','shuffled'),assignments):
        out[m+'gate']=np.linalg.solve(g,z.T@np.eye(R)[labels])
        count=np.zeros((R+1,BINS,7));link=np.zeros((R+1,7,7));initial=np.zeros((R+1,7))
        for (co,y,_,_),k in zip(records,labels):
            c=np.bincount(co*7+y,minlength=BINS*7).reshape(BINS,7)
            t=np.bincount(y[:-1]*7+y[1:],minlength=49).reshape(7,7)
            count[k]+=c;count[-1]+=c;link[k]+=t;link[-1]+=t;initial[k,y[0]]+=1;initial[-1,y[0]]+=1
        global_prior=normalize(count[-1].sum(0)+1)
        coarse=normalize(count[-1].reshape(49,4,6,3,7).sum((2,3))+18*global_prior)
        back=np.repeat(coarse.reshape(-1,7),18,axis=0)
        global_obs=normalize(count[-1]+24*back)
        global_link=normalize(link[-1]+40*global_prior)
        out[m+'obs']=np.concatenate([normalize(count[:-1]+70*global_obs),global_obs[None]])
        out[m+'edge']=np.concatenate([normalize(link[:-1]+90*global_link),global_link[None]])
        out[m+'prior']=normalize(count.sum(1)+1)
        out[m+'initial']=normalize(initial+15*normalize(initial[-1]+1))
    return {k:np.ascontiguousarray(v.astype(np.float32)) for k,v in out.items()}

def get_panel(group):
    if group=='test':
        x,off,y,ids=scale.fixed(group)
    else:
        d=scale.readbank(S/(group+'-panel.npz')); x,off,y,ids=d.x,d.offsets,d.y,d.case
    old,base=scale.legacy(group);np.testing.assert_array_equal(y,base.gold)
    mask=scale.maskinfo(group,type('B',(),dict(case=ids,offsets=off))()) if group!='test' else np.ones(len(y),bool)
    docs=[statistics(x[a:b]) for a,b in zip(off,off[1:])]
    return dict(docs=docs,off=off,y=y,ids=ids,mask=mask,old=old)

def predictor():
    f=ctypes.CDLL(str(H/'posterior.so')).posterior
    arr=np.ctypeslib.ndpointer(np.float32,flags='C_CONTIGUOUS')
    f.argtypes=[arr,arr,arr,ctypes.c_int,arr];f.restype=None
    return f

def posteriors(data,param,gate,temp,sham=False):
    f=predictor();tag='shuffled' if sham else 'learned';outputs=[];locals_=[]
    for code,_,feat in data['docs']:
        z=np.r_[np.clip((feat-param['mu'])/param['sd'],-4,4),1]
        s=np.exp(np.clip(5*(z@param[tag+'gate']),-8,8));s/=s.sum();w=np.r_[gate*s,1-gate]
        mix=lambda suffix:np.tensordot(w,param[tag+suffix],axes=1)
        # Marginalize the source regime before passing seven-state messages.
        local=np.einsum('i,ijn->jn',w,param[tag+'obs'][:,code,:])
        prior=mix('prior'); edge=normalize(mix('edge')**temp *prior[None,:]**(1-temp))
        obs=np.clip(local/np.maximum(prior,.003),.02,100)
        out=np.empty((len(code),7),np.float32)
        f(np.ascontiguousarray(obs,np.float32),np.ascontiguousarray(edge,np.float32),np.ascontiguousarray(mix('initial'),np.float32),len(out),out)
        outputs.append(out);locals_.append(local.astype(np.float32))
    return np.concatenate(outputs),np.concatenate(locals_)

def prediction(d,post,local,c):
    u=c['strength'];old=d['old']
    score=old+u*(post-local) if c['form']=='correction' else (1-u)*old+u*post
    if c['scope']=='pair':
        cand=old.copy();cand[:,[0,5]]=score[:,[0,5]];score=cand
    return score.argmax(1).astype(np.int8)

def select(d,params):
    y,mask=d['y'],d['mask'];base=d['old'].argmax(1)
    initial,m,nm=scale.key(y,base,mask);best=None;grid=[]
    for gate in (0.,.85):
        for temp in (.35,.75,1.0):
            post,local=posteriors(d,params,gate,temp)
            for form in ('correction','mixture'):
                for strength in (0.,.12,.28,.5):
                    for scope in ('all','pair'):
                        c=dict(gate=gate,temp=temp,form=form,strength=strength,scope=scope)
                        p=prediction(d,post,local,c)
                        k,a,b=scale.key(y,p,mask)
                        ok=a['pair_errors']<=m['pair_errors'] and b['pair_errors']<=nm['pair_errors']
                        grid.append([c,k[0],a['pair_errors'],b['pair_errors'],bool(ok)])
                        if ok and (best is None or k>best[0]):best=(k,c)
    return best,grid

def evaluate(d,par,c):
    old=d['old'].argmax(1);preds={'context':old}
    for name,sham in [('selected',False),('shuffled_regime',True)]:
        post,local=posteriors(d,par,c['gate'],c['temp'],sham)
        preds[name]=prediction(d,post,local,c);preds['posterior_only'+('_shuffled' if sham else '')]=post.argmax(1)
    return preds,{'rows':len(old),'cases':len(d['ids']),'all':{k:lib.metric(d['y'],v) for k,v in preds.items()},'unmatched':{k:lib.metric(d['y'][d['mask']],v[d['mask']]) for k,v in preds.items()}}

def archive(prefix,fnames):
    dest=H.parent/(prefix+'.zip')
    with ZipFile(dest.with_suffix('.part'),'w',ZIP_DEFLATED) as z:
        for p in fnames: z.write(p,p.name)
        z.writestr('SHA256SUMS.json',json.dumps({p.name:sha(p) for p in fnames}))
    dest.with_suffix('.part').replace(dest)

def run(n,official):
    check();prefix=f'regime-{n}';weights=H/(prefix+'.safetensors');lock=H/(prefix+'-freeze.json')
    if official:
        receipt=json.loads(lock.read_text());assert sha(weights)==receipt['weights_sha256']
        par=load_file(str(weights));c=receipt['config'];groups=('test',)
    else:
        par=fit(n);save_file(par,str(weights));d=get_panel('select')
        (key,c),grid=select(d,par)
        scale.dump(lock,dict(config=c,quality=key[0],weights_sha256=sha(weights),train_receipt=sha(S/'protocol.json'),grid=grid))
        groups=('select','confirm','fresh')
    out={};files=[weights,lock,Path(__file__),H/'posterior.cpp',H/'protocol.json']
    for g in groups:
        panel=get_panel(g);p,m=evaluate(panel,par,c);out[g]=m;dest=H/(prefix+'-'+g+'.npz')
        np.savez_compressed(dest,case_ids=panel['ids'],offsets=panel['off'],gold=panel['y'],**p);files.append(dest)
        print(prefix,g,c,m['all']['selected']['macro_f1'],m['all']['selected']['pair_errors'],flush=True)
    label=prefix+('-official' if official else '-dev');report=H/(label+'.json');scale.dump(report,out);archive(label,files+[report])

if __name__=='__main__':
    run(int(sys.argv[1]),len(sys.argv)>2 and sys.argv[2]=='official')
