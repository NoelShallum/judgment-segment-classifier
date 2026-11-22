"""Program residual and error-margin updates directly into the released document matrix."""
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch
from safetensors.numpy import load_file,save_file
import legal_solve as legal
import run_scale as scale
import solve as lib
from legal_encode import check,sha

H=Path(__file__).resolve().parent
S=scale.H; A=H.parent/'anchor-scale-512'
E=lib.EYE
def features(compact,offsets,anchor,params):
    d={};streams={};pcs={}
    for k,start in [('past',0),('both',62)]:
        v=compact[:,start:start+62]
        d[k]=SimpleNamespace(base=anchor,offsets=offsets,raw=v[:,:7],hidden=v[:,14:])
        streams[k]=v[:,7:14];pcs[k]={'mu':np.zeros(48,np.float32),'proj':np.eye(48,dtype=np.float32)}
    return lib.docbasis(legal.basis(d,streams,pcs,'semantic'),params)
def groups(g,params):
    d=scale.readbank(S/(g+'-panel.npz'))
    a=scale.readbank(A/('validation-'+g+'.npz'))
    for x,y in [(d.case,a.case_ids),(d.offsets,a.offsets),(d.y,a.gold)]:np.testing.assert_array_equal(x,y)
    x=features(d.x,d.offsets,a.reviewed,params)
    old=x@params['coef']+.14*E[a.reviewed]
    np.testing.assert_array_equal(old.argmax(1),a.score.argmax(1))
    assert np.max(np.abs(old-a.score))<1e-4
    return SimpleNamespace(x=x,old=old,prior=a.reviewed,gold=d.y,case_ids=d.case,offsets=d.offsets,novel=a.novel)
def error_outputs(x,y,old):
    base=old.argmax(1);idx=np.arange(len(y))
    margin=np.clip(old[idx,base]-old[idx,y]+.22,0,1.15)
    mistake=(E[y]-E[base])*margin[:,None]/2
    full=E[y]-old
    return np.column_stack((full,mistake))
def read_training(n):
    p=S/(f'train-{n:02d}.npz');d=scale.readbank(p)
    b=scale.readbank(A/'returned'/(f'reviewed-{n:02d}.npz'))
    for x,y in [(d.case,b.case_ids),(d.offsets,b.offsets),(d.y,b.gold)]:np.testing.assert_array_equal(x,y)
    return d,b.reference
def finalscore(d,coef):
    return (d.x@coef+.14*E[d.prior]).argmax(1).astype(np.int8)
def fit(params):
    v=groups('select',params);oldcoef=params['coef'];k0,zero,novel0=scale.key(v.gold,v.old.argmax(1),v.novel)
    grams={};rhs={};resources=[];ys=np.concatenate([read_training(i)[0].y for i in range(8)])
    freq=np.bincount(ys,minlength=7)/len(ys);rng=np.random.default_rng(2607012)
    for i in range(8):
        d,prior=read_training(i);x=features(d.x,d.offsets,prior,params);old=x@oldcoef+.14*E[prior];y=d.y
        t=error_outputs(x,y,old);sh=t[rng.permutation(len(t))];alltargets=np.column_stack((t,sh))
        for rarity in (0.,.45):
            w=(freq[y]**-rarity).astype(np.float32)
            g,r=lib.gram(x,alltargets,w)
            if i==0:grams[rarity]=g;rhs[rarity]=r
            else:grams[rarity]+=g;rhs[rarity]+=r
        resources.append(sha(A/'returned'/(f'reviewed-{i:02d}.npz')))
    best_key=k0;cfg={'gain':0.,'target':'score','rarity':0.,'ridge':.05,'scope':'all'};best=None;grid=[]
    for rarity in grams:
        for ridge in (.03,.18,1.):
            sol=lib.ridge(grams[rarity],rhs[rarity],ridge,True)
            for name,j in [('score',0),('margin',7)]:
                delta=sol[:,j:j+7];neg=sol[:,14+j:21+j]
                for scope in ('all','pair'):
                    if scope=='pair':
                        d=np.zeros_like(delta);s=np.zeros_like(neg)
                        d[:,[0,5]]=delta[:,[0,5]];s[:,[0,5]]=neg[:,[0,5]]
                    else:d=delta;s=neg
                    change=v.x@d
                    for gain in (.12,.35,.65,1.0):
                        new=(v.old+gain*change).argmax(1);k,full,nov=scale.key(v.gold,new,v.novel)
                        good=full['pair_errors']<=zero['pair_errors'] and nov['pair_errors']<=novel0['pair_errors']
                        c=dict(rarity=rarity,ridge=ridge,target=name,scope=scope,gain=gain)
                        grid.append([c,k[0],full['pair_errors'],nov['pair_errors'],bool(good)])
                        if good and k>best_key:best_key,cfg,best=k,c,(d.copy(),s.copy())
    if best is None:best=(np.zeros_like(oldcoef),np.zeros_like(oldcoef))
    d,s=best;s*=np.linalg.norm(d)/max(np.linalg.norm(s),1e-10)
    changed=oldcoef+cfg['gain']*d;shuffled=oldcoef+cfg['gain']*s
    file=H/'coef.safetensors';save_file(dict(coef=changed,shuffled_coef=shuffled),str(file))
    scale.dump(H/'freeze.json',dict(config=cfg,selector=best_key[0],grid=grid,parameters_changed=int(np.sum(changed!=oldcoef)),
               frobenius=float(np.linalg.norm(changed-oldcoef)),base_norm=float(np.linalg.norm(oldcoef)),shape=list(oldcoef.shape),
               parent_sha256=sha(legal.H/'doc-semantic.safetensors'),patch_sha256=sha(file),data_receipts=resources))
    return changed,shuffled
def run():
    check();params=legal.get('doc-semantic');coef,neg=fit(params);out={};files=[H/n for n in ('native.py','protocol.json','freeze.json','coef.safetensors')]
    for g in ('select','confirm','fresh'):
        d=groups(g,params);pp={'context':d.old.argmax(1),'selected':finalscore(d,coef),'shuffled':finalscore(d,neg)}
        scores={k:lib.metric(d.gold,v) for k,v in pp.items()};out[g]=dict(cases=len(d.case_ids),rows=len(d.gold),scores=scores,
            novel={k:lib.metric(d.gold[d.novel],v[d.novel]) for k,v in pp.items()},ci=lib.delta_ci(d,pp['selected'],pp['context']))
        p=H/(g+'-pred.npz');np.savez_compressed(p,case_ids=d.case_ids,offsets=d.offsets,gold=d.gold,**pp);files.append(p)
        print('Decoder matrix',g,[(k,round(v['macro_f1']*100,3),v['pair_errors']) for k,v in scores.items()],flush=True)
    scale.dump(H/'validation.json',out);files.append(H/'validation.json')
    promoted=all(out[g]['scores']['selected']['macro_f1']>out[g]['scores']['context']['macro_f1'] and out[g]['scores']['selected']['pair_errors']<out[g]['scores']['context']['pair_errors'] for g in ('confirm','fresh'))
    lock=H/'official-lock.json';scale.dump(lock,dict(promoted=promoted,validation_sha256=sha(H/'validation.json'),freeze_sha256=sha(H/'freeze.json')))
    files.append(lock)
    if promoted:
        x,off,y,ids=scale.fixed('test');src=legal.read('past','test');prior=src.base;del src
        preds={k:[] for k in ('context','selected','shuffled')}
        for a in range(0,len(ids),16):
            b=min(len(ids),a+16);lo,hi=off[a],off[b]
            design=features(x[lo:hi],off[a:b+1]-lo,prior[lo:hi],params);anchor=.14*E[prior[lo:hi]]
            for k,w in [('context',params['coef']),('selected',coef),('shuffled',neg)]:
                preds[k].append((design@w+anchor).argmax(1).astype(np.int8))
        pp={k:np.concatenate(v) for k,v in preds.items()}
        with np.load(legal.H/'test-predictions.npz') as source:np.testing.assert_array_equal(pp['context'],source['doc-semantic'])
        p=H/'official-predictions.npz';np.savez_compressed(p,case_ids=ids,offsets=off,gold=y,**pp);files.append(p)
        p=H/'official-results.json';scale.dump(p,dict(scores={k:lib.metric(y,v) for k,v in pp.items()},cases=len(ids),rows=len(y)));files.append(p)
    dest=H.parent/'native-decoder-results.zip'
    with ZipFile(dest.with_suffix('.part'),'w',ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.name)
        z.writestr('SHA256SUMS.json',json.dumps({p.name:sha(p) for p in files}))
    dest.with_suffix('.part').replace(dest)
if __name__=='__main__':run()
