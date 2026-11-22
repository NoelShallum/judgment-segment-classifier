"""Incrementally solve class-readout and transition weights on official-train judgments."""
import ctypes,hashlib,json,os,sys
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
import torch
from safetensors.numpy import load_file,save_file
import solve as lib
import legal_solve as legal
from scale_encode import H,L,panels,generate,fixed
from legal_encode import sha,check
E=lib.EYE
def dump(p,obj):
    part=p.with_suffix('.tmp');part.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');part.replace(p)
def readbank(path):
    with np.load(path,allow_pickle=False) as f:return SimpleNamespace(**{k:f[k] for k in f.files})
def design(x,offsets):
    parts=[]
    scores=[];hs=[]
    for sub in (x[:,:62],x[:,62:]):
        scores.extend((lib.probabilities(.65*sub[:,:7]),lib.probabilities(.8*sub[:,7:14])))
        hs.append(sub[:,14:][:,-32:])
    probs=np.column_stack(scores);h=np.column_stack(hs)
    for a,b in zip(offsets,offsets[1:]):
        p=probs[a:b];v=h[a:b];n=len(v);i=np.arange(n);t=((i+.5)/n)[:,None]
        mean=p.mean(0);q=np.vstack((np.zeros((1,p.shape[1])),p.cumsum(0,dtype=np.float64)));vsum=np.vstack((np.zeros((1,v.shape[1])),v.cumsum(0,dtype=np.float64)))
        out=[p,v,lib.coords(n),p*t,p*t*t,np.broadcast_to(mean,(n,len(mean))),mean*p,(p[:,:7,None]*p[:,None,14:21]).reshape(n,-1)]
        for band in (1,3,9,26,65):
            lo=np.maximum(0,i-band);hi=np.minimum(n,i+band+1)
            out.extend(((q[i]-q[lo])/np.maximum(1,i-lo)[:,None],(q[hi]-q[i+1])/np.maximum(1,hi-i-1)[:,None]))
            if band in (3,9,26):
                out.extend(((vsum[i]-vsum[lo])/np.maximum(1,i-lo)[:,None],(vsum[hi]-vsum[i+1])/np.maximum(1,hi-i-1)[:,None]))
        ps=.5*(p[:,:7]+p[:,14:21]);winner=ps.argmax(1)
        for cls in range(7):
            anchors=np.where((winner==cls)&(ps[:,cls]>.68))[0]
            if len(anchors):
                j=np.searchsorted(anchors,i);prev=anchors[np.clip(j-1,0,len(anchors)-1)];nxt=anchors[np.clip(j,0,len(anchors)-1)]
                l=np.where((j>0)&(i>=prev),np.exp(-np.minimum(i-prev,200).clip(min=0)/12),0)
                r=np.where(j<len(anchors),np.exp(-np.minimum(nxt-i,200).clip(min=0)/12),0)
            else:l=np.zeros(n);r=np.zeros(n)
            out.extend((l[:,None],r[:,None]))
        parts.append(np.column_stack(out).astype(np.float32))
    return np.concatenate(parts)
def setup(basebank):
    train=design(basebank.x,basebank.offsets)
    center=train.mean(0,dtype=np.float64).astype(np.float32);scale=1/np.maximum(train.std(0,dtype=np.float64),.03).astype(np.float32)
    rng=np.random.default_rng(2601484)
    params={'center':center,'scale':scale,'rand':(rng.normal(size=(len(center),448))/np.sqrt(len(center))).astype(np.float32),'rbias':rng.uniform(-.5,.5,448).astype(np.float32)}
    save_file(params,str(H/'basis.safetensors'));return params
def decode_lib():
    f=ctypes.CDLL(str(H/'decode.so')).decode
    ptr=np.ctypeslib.ndpointer
    f.argtypes=[ptr(np.float32,flags='C_CONTIGUOUS'),ptr(np.float32,flags='C_CONTIGUOUS'),ptr(np.int64,flags='C_CONTIGUOUS'),ctypes.c_int,ptr(np.int8,flags='C_CONTIGUOUS')];f.restype=None
    return f
def dec(z,t,off,st,decoder):
    if st==0:return z.argmax(1).astype(np.int8)
    p=np.empty(len(z),np.int8);decoder(np.ascontiguousarray(z,np.float32),np.ascontiguousarray(t*st,np.float32),np.ascontiguousarray(off,np.int64),len(off)-1,p);return p
def legacy(g):
    d=legal.fetch_group(g);orig=d['past'];name='doc-semantic';lk=json.loads((L/'freeze.json').read_text());p=legal.get(name)
    edits={k:legal.head_output(v,legal.get(k+'-edited')) for k,v in d.items()};pcs={k:legal.get(k+'-pca') for k in d}
    sc=lib.docbasis(legal.basis(d,edits,pcs,'semantic'),p)@p['coef'];c=lk['cfg'][name]
    assert c['scope']=='all' and c['margin']==0
    sc+=c['anchor']*E[orig.base]
    with np.load(L/(g+'-predictions.npz')) as f:
        np.testing.assert_array_equal(sc.argmax(1),f[name]);np.testing.assert_array_equal(f['gold'],orig.gold)
    return sc,orig
def maskinfo(g,bank):
    f=H/(g+'-mask.npz')
    with np.load(f,allow_pickle=False) as z:
        np.testing.assert_array_equal(z['case_ids'],bank.case);np.testing.assert_array_equal(z['offsets'],bank.offsets)
        return z['novel']
def prepare_evaluation(g,params,need_novel=True):
    x=readbank(H/(g+'-panel.npz'));b=lib.docbasis(design(x.x,x.offsets),params)
    old,d=legacy(g);np.testing.assert_array_equal(d.gold,x.y)
    return SimpleNamespace(x=b,old=old,gold=x.y,offsets=x.offsets,case_ids=x.case,novel=maskinfo(g,x) if need_novel else np.ones(len(x.y),bool))
def transition(count):
    pri=(count.sum(0)+1)/(count.sum()+7)
    vals=count+.5+25*pri[None,:]
    return np.clip(np.log(vals/vals.sum(1,keepdims=True)/pri),-2.5,2.5).astype(np.float32)
def cm(y,p):return np.bincount(y.astype(np.int64)*7+p.astype(np.int64),minlength=49).reshape(7,7)
def key(y,p,novel):
    m=lib.metric(y,p);nm=lib.metric(y[novel],p[novel]);return ((m['macro_f1']+nm['macro_f1'])/2,-m['pair_errors'],m['accuracy']),m,nm
def scorebank(d,coef,tr,c,decoder):
    e=d.x@coef
    mixed=(1-c['mix'])*d.old+c['mix']*e
    return {'context':d.old.argmax(1),'scale_only':e.argmax(1),'no_edges':mixed.argmax(1),'selected':dec(mixed,tr,d.offsets,c['edge'],decoder)}
def fitrun():
    check();torch.set_num_threads(1);panels()
    base=readbank(H/'fit-panel.npz');par=setup(base);decoder=decode_lib()
    ev={g:prepare_evaluation(g,par) for g in ('select','confirm','fresh')}
    labfreq=(np.bincount(base.y,minlength=7)+1).astype(float);labfreq/=labfreq.sum()
    rr=(.0,.40);grams={};rhs={};count=np.zeros((7,7),np.float64)
    v=ev['select'];basekey,_,_=key(v.gold,v.old.argmax(1),v.novel)
    schedule=json.loads((H/'protocol.json').read_text());blocks=(len(schedule['case_ids'])+63)//64;reports={}
    def step(i,path):
        d=readbank(path);x=lib.docbasis(design(d.x,d.offsets),par);y=d.y
        for rare in rr:
            w=(labfreq[y]**(-rare)).astype(np.float32)
            g,r=lib.gram(x,E[y],w)
            if rare not in grams:grams[rare]=g;rhs[rare]=r
            else:grams[rare]+=g;rhs[rare]+=r
        for a,b in zip(d.offsets,d.offsets[1:]):
            if b-a>1:count[:]+=cm(y[a:b-1],y[a+1:b])
        if i+1 not in (8,24,blocks):return
        tr=transition(count);best=None;allc=[]
        for rare in rr:
            for lam in (.035,.20):
                coef=lib.ridge(grams[rare],rhs[rare],lam,True);sp=v.x@coef
                for mix in (.0,.22,.45,.70,1.):
                    for edge in (.0,.012,.04,.09):
                        pred=dec((1-mix)*v.old+mix*sp,tr,v.offsets,edge,decoder)
                        k,m,nm=key(v.gold,pred,v.novel)
                        entry=dict(rare=rare,lam=lam,mix=mix,edge=edge)
                        allc.append(dict(config=entry,quality=k[0],all=m['macro_f1'],novel=nm['macro_f1'],pair=m['pair_errors']))
                        if best is None or k>best[0]:best=(k,entry,coef)
        kval,c,coef=best;prefix='scale-'+str(i+1)
        checkpoint=H/(prefix+'.safetensors');save_file({'coef':coef,'transition':tr},str(checkpoint))
        freeze=dict(config=c,selection_objective=kval[0],baseline_selection=basekey[0],weights_sha256=sha(checkpoint),source=sha(H/'protocol.json'),shards=i+1,
                    cases=min((i+1)*64,len(schedule['case_ids'])),computed_sentences=int(count.sum())+min((i+1)*64,len(schedule['case_ids'])),candidates=allc,
                    note='Choose on 48 validation judgments by the mean of pooled seven-role macro-F1 on all rows and on rows without an eligible lexical-memory match. Confirmations opened only after numerical weights and operating rule of each prespecified stage are frozen.')
        dump(H/(prefix+'-freeze.json'),freeze)
        rep={}
        for g,bank in ev.items():
            pp=scorebank(bank,coef,tr,c,decoder);p=pp['selected']
            rep[g]=dict(cases=len(bank.case_ids),rows=len(p),scores={name:lib.metric(bank.gold,v) for name,v in pp.items()},novel={name:lib.metric(bank.gold[bank.novel],v[bank.novel]) for name,v in pp.items()},paired=lib.delta_ci(bank,p,pp['context']))
            np.savez_compressed(H/(prefix+'-'+g+'-predictions.npz'),case_ids=bank.case_ids,offsets=bank.offsets,gold=bank.gold,**pp)
        dump(H/(prefix+'-development.json'),rep)
        reports[prefix]={'quality':kval[0],'config':c,'checkpoint':sha(checkpoint),'development':sha(H/(prefix+'-development.json'))}
        print('STAGE',prefix,c,{k:(round(v['scores']['selected']['macro_f1']*100,3),round(v['novel']['selected']['macro_f1']*100,3),v['scores']['selected']['pair_errors']) for k,v in rep.items()},flush=True)
        archive(prefix)
    generate(blocks,step)
    choice=max(reports,key=lambda k:reports[k]['quality']);dump(H/'selection.json',dict(chosen=choice,stages=reports,training_cases=len(schedule['case_ids']),basis_sha256=sha(H/'basis.safetensors')))
    archive('final')
def archive(label):
    files=[p for p in H.iterdir() if p.name in ('protocol.json','basis.safetensors','selection.json','scale_encode.py','run_scale.py','decode.cpp') or p.name.startswith('scale-') and p.suffix in ('.npz','.json','.safetensors') or p.name in ('official-results.json','official-predictions.npz')]
    dest=H.parent/('scale-'+label+'.zip');tmp=dest.with_suffix('.part')
    with ZipFile(tmp,'w',ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.name)
        z.writestr('SHA256SUMS.json',json.dumps({p.name:sha(p) for p in files}))
    tmp.replace(dest);print('archive',label,sha(dest),flush=True)
def official():
    info=json.loads((H/'selection.json').read_text());choice=info['chosen'];f=json.loads((H/(choice+'-freeze.json')).read_text())
    assert f['weights_sha256']==sha(H/(choice+'.safetensors'));w=load_file(str(H/(choice+'.safetensors')))
    base,off,y,case=fixed('test');np.savez(H/'test-panel.npz',x=base,offsets=off,y=y,case=case);del base
    data=prepare_evaluation('test',load_file(str(H/'basis.safetensors')),False)
    scores=scorebank(data,w['coef'],w['transition'],f['config'],decode_lib());p=scores['selected']
    # Bootstrap in bounded batches for the 712-document official set.
    def ci(a,b):
        ax=np.stack([cm(data.gold[i:j],a[i:j]) for i,j in zip(off,off[1:])]);bx=np.stack([cm(data.gold[i:j],b[i:j]) for i,j in zip(off,off[1:])])
        rng=np.random.default_rng(2936);results=[]
        for _ in range(40):
            inds=rng.integers(0,len(ax),(100,len(ax)));m=ax[inds].sum(1);n=bx[inds].sum(1)
            calc=lambda x:(2*np.diagonal(x,axis1=1,axis2=2)/np.maximum(x.sum(-1)+x.sum(-2),1)).mean(1)
            results.extend(calc(m)-calc(n))
        return np.quantile(results,[.025,.975]).tolist()
    out=dict(chosen=choice,shards=f['shards'],cases=len(case),rows=len(y),scores={k:lib.metric(y,v) for k,v in scores.items()},paired_f1_delta=ci(p,scores['context']),frozen_sha256=sha(H/'selection.json'))
    dump(H/'official-results.json',out);np.savez_compressed(H/'official-predictions.npz',case_ids=case,offsets=off,gold=y,**scores);archive('official')
if __name__=='__main__':
    (official if len(sys.argv)>1 and sys.argv[1]=='official' else fitrun)()
