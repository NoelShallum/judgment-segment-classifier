"""Write a train-only semantic key/value memory and test its error corrections on novel passages."""
import argparse,hashlib,json,os,time
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
import torch
import solve as lib
import run_scale as scale
from scale_encode import H,fixed
from legal_encode import check,sha
E=lib.EYE

def feat(x,off):
    near=[];probs=[];sem=[]
    for sub in (x[:,:62],x[:,62:]):
        probs.append(lib.probabilities(.65*sub[:,:7]));probs.append(lib.probabilities(.8*sub[:,7:14]))
        sem.append(sub[:,14:][:,-24:])
    p=np.column_stack(probs);h=np.column_stack(sem);basis=[]
    for a,b in zip(off,off[1:]):
        v=p[a:b];s=h[a:b];n=len(s);i=np.arange(n);t=(i+.5)/n
        vp=np.vstack((np.zeros((1,v.shape[1])),v.cumsum(0,dtype=np.float64)));hp=np.vstack((np.zeros((1,s.shape[1])),s.cumsum(0,dtype=np.float64)))
        pieces=[v,s]
        for band in (4,15):
            lo=np.maximum(0,i-band);hi=np.minimum(n,i+band+1)
            l=(vp[i]-vp[lo])/np.maximum(i-lo,1)[:,None];r=(vp[hi]-vp[i+1])/np.maximum(hi-i-1,1)[:,None]
            # Keep both author-view distributions for preceding and following discourse.
            pieces.extend((l[:,np.r_[:7,14:21]],r[:,np.r_[:7,14:21]]))
            if band==4:pieces.append((hp[hi]-hp[lo])/(hi-lo)[:,None])
        pieces.extend((np.broadcast_to(v.mean(0)[np.r_[:7,14:21]],(n,14)),lib.coords(n)))
        basis.append(np.column_stack(pieces).astype(np.float32))
    return np.concatenate(basis),.5*(probs[0]+probs[2])
def normfit(x):
    mu=x.mean(0,dtype=np.float64).astype(np.float32);sd=np.maximum(x.std(0,dtype=np.float64),.05).astype(np.float32)
    dims=[28,48,14,14,48,14,14,14,9];mass=[.85,1.,.46,.46,.53,.42,.42,.35,.48]
    assert sum(dims)==len(mu)
    weights=np.repeat(np.array(mass)/np.sqrt(dims),dims).astype(np.float32)
    return mu,sd,weights
def norm(x,par):
    mu,sd,w=par;q=np.clip((x-mu)/sd,-4.,4.)*w
    return (q/np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-8)).astype(np.float16)
def load_train(n,par):
    ks=[];vs=[];sources=[];orig=[];ncase=0
    for j in range(n):
        d=scale.readbank(H/(f'train-{j:02d}.npz'));f,o=feat(d.x,d.offsets)
        ks.append(norm(f,par));orig.append(o.astype(np.float16));vs.append(d.y)
        sources.append(np.repeat(np.arange(ncase,ncase+len(d.case),dtype=np.int32),np.diff(d.offsets)));ncase+=len(d.case)
    return np.concatenate(ks),np.concatenate(vs),np.concatenate(sources),np.concatenate(orig)
def build_index(keys,values,cases,ref):
    rng=np.random.default_rng(932071)
    K=torch.as_tensor(keys,device='cuda');size=min(1024,max(160,int(np.sqrt(len(keys))*1.5)))
    cent=K[rng.choice(len(K),size,replace=False)].float()
    # One deterministic assignment and one arithmetic centroid update form the index.
    def assign(c):
        return torch.cat([(v@c.T).argmax(1).to(torch.int32) for v in K.split(2048)]).cpu().numpy()
    first=assign(cent.half());tot=np.zeros((size,keys.shape[1]),np.float32);cnt=np.bincount(first,minlength=size)
    for ix in np.array_split(np.arange(len(keys)),max(1,len(keys)//18000)):np.add.at(tot,first[ix],keys[ix].astype(np.float32))
    ok=cnt>0;cent[ok]=torch.as_tensor(tot[ok]/np.maximum(np.linalg.norm(tot[ok],axis=1,keepdims=True),1e-7),device='cuda')
    cent=cent.half();bucket=assign(cent);order=np.argsort(bucket,kind='stable');ends=np.r_[0,np.cumsum(np.bincount(bucket,minlength=size))]
    return dict(keys=K[torch.as_tensor(order,device='cuda')],cent=cent,ends=ends,values=values[order],cases=cases[order],ref=ref[order],shuffle=values[rng.permutation(len(values))][order],size=size)
def retrieve(index,queries):
    q=torch.as_tensor(queries,device='cuda');near=torch.cat([(rr@index['cent'].T).topk(5,dim=1).indices for rr in q.split(2048)]).cpu().numpy()
    count=40;score=np.full((len(q),near.shape[1],count),-2.,np.float16);inds=np.zeros(score.shape,np.int32)
    for cell,(a,b) in enumerate(zip(index['ends'],index['ends'][1:])):
        if a==b:continue
        rows,rank=np.where(near==cell);kk=min(count,b-a)
        for start in range(0,len(rows),256):
            loc=rows[start:start+256];sel=rank[start:start+256];qq=q[torch.as_tensor(loc,device='cuda')]
            sim=qq@index['keys'][a:b].T;v,ix=sim.topk(kk,dim=1)
            score[loc,sel,:kk]=v.cpu().numpy();inds[loc,sel,:kk]=ix.cpu().numpy()+a
    s=score.reshape(len(q),-1);z=inds.reshape(len(q),-1);rank=np.argsort(-s,axis=1,kind='stable')[:,:72]
    s=np.take_along_axis(s,rank,axis=1);z=np.take_along_axis(z,rank,axis=1)
    # Keep the strongest neighbor from each training judgment so long repeated text cannot dominate.
    c=index['cases'][z];valid=np.ones(c.shape,bool)
    for i in range(1,c.shape[1]):valid[:,i]=np.all(c[:,:i]!=c[:,i,None],axis=1)
    return s.astype(np.float32),z,valid
def vote(index,nn,k,tau,sh=False):
    sim,slots,valid=nn;keep=valid&(np.cumsum(valid,axis=1)<=k)
    weight=np.exp((sim-sim[:,0:1])/tau)*keep;weight/=np.maximum(weight.sum(1,keepdims=True),1e-9)
    vals=index['shuffle' if sh else 'values'][slots]
    post=np.einsum('nk,nkc->nc',weight,E[vals]);old=np.einsum('nk,nkc->nc',weight,index['ref'][slots].astype(np.float32))
    return post,old
def panel(g,par):
    if g=='test':
        xx,off,_,_=fixed(g);f,_=feat(xx,off);del xx
    else:
        v=scale.readbank(H/(g+'-panel.npz'));f,_=feat(v.x,v.offsets)
    old,d=scale.legacy(g)
    novel=scale.maskinfo(g,type('B',(),dict(case=d.case_ids,offsets=d.offsets))()) if g!='test' else np.ones(len(d.gold),bool)
    return type('Panel',(),dict(keys=norm(f,par),old=old,gold=d.gold,case_ids=d.case_ids,offsets=d.offsets,novel=novel))()
def choose(idx,d,nn):
    best=None;grid=[]
    for k in (3,9,24):
        for tau in (.035,.12):
            post,old=vote(idx,nn,k,tau)
            for use in ('residual','mixture'):
                for strength in (0.,.12,.28,.55,.85):
                    p=((d.old+strength*(post-old)) if use=='residual' else (1-strength)*d.old+strength*post).argmax(1)
                    key,m,nov=scale.key(d.gold,p,d.novel);cfg=dict(k=k,tau=tau,mode=use,strength=strength)
                    grid.append([cfg,*key,nov['macro_f1'],m['macro_f1']])
                    if best is None or key>best[0]:best=(key,cfg)
    return best,grid
def report(idx,d,nn,cfg,g):
    result={'context':d.old.argmax(1)}
    for kind,sh in [('selected',False),('shuffled',True)]:
        p,old=vote(idx,nn,cfg['k'],cfg['tau'],sh)
        u=cfg['strength'];scores=(d.old+u*(p-old)) if cfg['mode']=='residual' else (1-u)*d.old+u*p
        result[kind]=scores.argmax(1)
        if not sh:result['memory_only']=p.argmax(1)
    met={k:lib.metric(d.gold,p) for k,p in result.items()}
    out=dict(scores=met,novel={k:lib.metric(d.gold[d.novel],p[d.novel]) for k,p in result.items()},cases=len(d.case_ids),rows=len(d.gold))
    if g!='test':out['ci']=lib.delta_ci(d,result['selected'],result['context'])
    return result,out
def run(n,official):
    check();fit=scale.readbank(H/'fit-panel.npz');f,_=feat(fit.x,fit.offsets);par=normfit(f);del f,fit
    print('index input',n,flush=True);ks,v,src,ref=load_train(n,par);idx=build_index(ks,v,src,ref)
    print('index built',n,len(v),idx['size'],flush=True)
    prefix='latent-'+str(n)
    if official:
        cfg=json.loads((H/(prefix+'-freeze.json')).read_text())['config'];groups=('test',)
    else:groups=('select','confirm','fresh')
    allm={}
    for g in groups:
        d=panel(g,par);nn=retrieve(idx,d.keys)
        if g=='select':
            (key,cfg),grid=choose(idx,d,nn)
            scale.dump(H/(prefix+'-freeze.json'),dict(config=cfg,quality=key[0],grid=grid,train_rows=len(v),cells=idx['size'],norm_sha256=hashlib.sha256(b''.join(w.tobytes() for w in par)).hexdigest(),train_data_sha256=sha(H/'protocol.json'),objective='Balanced full and lexical-unmatched selector macro-F1; nearest neighbors contribute at most one vote per training case. Only original train supplies values.'))
        p,met=report(idx,d,nn,cfg,g);allm[g]=met
        np.savez_compressed(H/(prefix+'-'+g+'-pred.npz'),case_ids=d.case_ids,offsets=d.offsets,gold=d.gold,**p)
        print('semantic memory',n,g,cfg,met['scores']['selected']['macro_f1'],met['scores']['selected']['pair_errors'],flush=True)
    label='official' if official else 'dev';scale.dump(H/(prefix+'-'+label+'.json'),allm)
    dest=H.parent/(prefix+'-'+label+'.zip');files=list(H.glob(prefix+'-*'))+[Path(__file__)]
    with ZipFile(dest.with_suffix('.part'),'w',ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.name)
        z.writestr('SHA256SUMS.json',json.dumps({p.name:sha(p) for p in files}))
    dest.with_suffix('.part').replace(dest)
if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('stage',type=int);cli.add_argument('--official',action='store_true');a=cli.parse_args();run(a.stage,a.official)
