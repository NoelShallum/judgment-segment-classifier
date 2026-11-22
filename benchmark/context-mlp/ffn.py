"""Directly edit the last feed-forward projection of the past-context InLegalBERT."""
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch
from safetensors.numpy import load_file, save_file
from transformers import BertTokenizerFast
import legal_encode as enc
import legal_solve as legal
import run_scale as scale
import solve as lib

H=Path(__file__).resolve().parent
KIND='past'

def gput(x):
    return torch.as_tensor(x,device='cuda')

@torch.inference_mode()
def capture(m,tok,group,cols=None,physical_only=False):
    docs=enc.records(group);off=np.r_[0,np.cumsum([len(d['text']) for d in docs])];packs=[]
    for a,d in zip(off,docs):
        ids=tok(d['text'],add_special_tokens=False,return_attention_mask=False,verbose=False)['input_ids']
        for j in range(len(ids)):
            for v,w in enc.windows([ids[t] if t>=0 else [] for t in (j-2,j-1,j)],tok.sep_token_id,tok.cls_token_id):
                packs.append((len(v),a+j,w,v))
    packs.sort(key=lambda v:v[0],reverse=True)
    block=m.bert.encoder.layer[-1].output
    cache={};hooks=[]
    def activation(mod,args):
        x=args[0][:,0]
        cache['a']=(x if cols is None else x[:,cols]).contiguous().cpu().numpy()
    def skip(mod,args):
        cache['s']=args[0][:,0].contiguous().cpu().numpy()
    if not physical_only:
        hooks=[block.dense.register_forward_pre_hook(activation),block.LayerNorm.register_forward_pre_hook(skip)]
    av=[];ss=[];indices=[];weight=[];h=np.zeros((off[-1],768),np.float32)
    i=0
    try:
        while i<len(packs):
            n=max(8,min(160,11264//packs[i][0]));part=packs[i:i+n];i+=len(part)
            ids=np.zeros((len(part),(part[0][0]+7)//8*8),np.int64)
            for j,item in enumerate(part):ids[j,:item[0]]=item[3]
            xt=gput(ids);v=m.bert(xt,attention_mask=xt!=0).pooler_output.float().cpu().numpy()
            ix=np.array([v[1] for v in part]);w=np.array([v[2] for v in part],np.float32)
            np.add.at(h,ix,v*w[:,None]);indices.append(ix);weight.append(w)
            if not physical_only:av.append(cache['a']);ss.append(cache['s'])
    finally:
        for hook in hooks:hook.remove()
    xx=np.concatenate(indices);ww=np.concatenate(weight);np.testing.assert_allclose(np.bincount(xx,ww),1,atol=2e-6)
    print('Captured',group,len(h),len(xx),'physical' if physical_only else 'activations',flush=True)
    result=dict(hidden=h,ix=xx,coverage=ww)
    if not physical_only:result.update(a=np.concatenate(av),s=np.concatenate(ss))
    return SimpleNamespace(**result)

def known_scores(group):
    d=legal.fetch_group(group);new=readout(d)
    if group!='fit':
        stored=scale.legacy(group)[0]
        np.testing.assert_array_equal(new.argmax(1),stored.argmax(1))
    return d,new

def readout(d):
    p=legal.get('doc-semantic');v={k:legal.head_output(x,legal.get(k+'-edited')) for k,x in d.items()}
    pcs={k:legal.get(k+'-pca') for k in d}
    return lib.docbasis(legal.basis(d,v,pcs,'semantic'),p)@p['coef']+.14*lib.EYE[d[KIND].base]

def modified(d,m,h):
    old=d[KIND];w=m.classifier.weight.detach().cpu().numpy();b=m.classifier.bias.detach().cpu().numpy()
    raw=h@w.T+b;dd=dict(d);dd[KIND]=SimpleNamespace(**dict(vars(old),raw=raw,hidden=h.astype(np.float16)))
    return readout(dd),lib.probabilities(raw*.65)-lib.probabilities(old.raw*.65)

def basis(f,y,m):
    # Select final-layer neurons by training-set role contrasts. Capture one target only once even if it spans several windows.
    means=[];sec=np.zeros(f.a.shape[1],np.float64)
    for label in range(7):
        ix=np.flatnonzero(y[f.ix]==label);w=f.coverage[ix]
        mean=(w@f.a[ix].astype(np.float32))/w.sum();means.append(mean)
    for ix in np.array_split(np.arange(len(f.a)),32):
        a=f.a[ix].astype(np.float32);sec+=f.coverage[ix]@(a*a)
    means=np.stack(means);freq=np.bincount(y,minlength=7)/len(y);mu=freq@means
    sd=np.sqrt(np.maximum(sec/len(y)-mu*mu,.0009));cons=(means-mu)/sd
    contrasts=[*cons,(means[0]-means[5])/sd,(means[1]-means[2])/sd,(means[3]-means[4])/sd]
    cols=np.unique(np.concatenate([np.argsort(np.abs(v))[-24:] for v in contrasts])).astype(np.int64)
    # Approximate the average seven-logit response to a displacement of the residual before the last LayerNorm.
    ix=np.arange(0,len(f.s),max(1,len(f.s)//5000));s=f.s[ix].astype(np.float32)
    sigma=np.sqrt(s.var(1)+m.config.layer_norm_eps);normed=(s-s.mean(1,keepdims=True))/sigma[:,None]
    block=m.bert.encoder.layer[-1].output
    gamma=block.LayerNorm.weight.detach().float().cpu().numpy()
    U=m.bert.pooler.dense.weight.detach().float().cpu().numpy();w=m.classifier.weight.detach().cpu().numpy()
    pool=f.hidden[f.ix[ix]];der=(1.-pool**2).mean(0)
    j=(w*der)@U*gamma[None,:];j-=j.mean(1,keepdims=True)
    j=(j*np.mean(1/sigma)-((j@normed.T)*(1/sigma)[None,:])@normed/(len(s)*s.shape[1]))
    j-=j.mean(0,keepdims=True)
    jj=j@j.T;directions=j.T@np.linalg.inv(jj+.07*(np.trace(jj)/7)*np.eye(7))
    return cols,mu[cols].astype(np.float32),sd[cols].astype(np.float32),directions.astype(np.float32)

def patches(f,d,scores,m):
    y=d[KIND].gold.astype(int);cols,mu,sd,directions=basis(f,y,m)
    a=np.zeros((len(y),len(cols)),np.float32)
    np.add.at(a,f.ix,f.a[:,cols].astype(np.float32)*f.coverage[:,None])
    x=np.c_[(a-mu)/sd,np.ones(len(a),np.float32)];out={}
    rng=np.random.default_rng(117466)
    for name,prior in [('donor',lib.probabilities(.65*d[KIND].raw)),('context',lib.probabilities(4*scores))]:
        for sh in (False,True):
            ys=y[rng.permutation(len(y))] if sh else y
            target=(lib.EYE[ys]-prior)*4
            g,r=lib.gram(x,target,lib.role_weights(y,.35));a=lib.ridge(g,r,.18,True)
            lin=(a[:-1]/sd[:,None]);bias=a[-1]-mu@lin
            out[name+('-shuffled' if sh else '')]=dict(weight=(directions@lin.T).astype(np.float32),bias=(directions@bias).astype(np.float32))
        t,shuf=out[name],out[name+'-shuffled']
        n=lambda p:np.sqrt((p['weight']**2).sum()+(p['bias']**2).sum())
        ratio=n(t)/max(n(shuf),1e-10)
        for k in shuf:shuf[k]*=ratio
    return cols,out

@torch.inference_mode()
def replay(cap,m,delta,bias):
    enc_out=m.bert.encoder.layer[-1].output;parts=[]
    for a,b in zip(np.array_split(cap.a,max(1,len(cap.a)//2048)),np.array_split(cap.s,max(1,len(cap.s)//2048))):
        update=gput(a).float()@gput(delta).T+gput(bias)
        z=enc_out.LayerNorm((gput(b).float()+update).half())
        p=torch.tanh(m.bert.pooler.dense(z)).float().cpu().numpy()
        parts.append(p)
    h=np.zeros_like(cap.hidden);np.add.at(h,cap.ix,np.concatenate(parts)*cap.coverage[:,None]);return h

def actual_delta(p,strength,source,bias):
    return ((source+strength*p['weight']).astype(np.float16).astype(np.float32)-source,
            (bias+strength*p['bias']).astype(np.float16).astype(np.float32)-bias)

def rule(c,old,doc,dlog):
    a=c['alpha']
    if c['rule']=='raw_residual':z=old+a*dlog
    else:z=(1-a)*old+a*doc
    if c['scope']=='pair':
        z1=old.copy();z1[:,[0,5]]=z[:,[0,5]];z=z1
    return z.argmax(1).astype(np.int8)

def arraypack(z,models):
    return dict(case_ids=z.case_ids,offsets=z.offsets,gold=z.gold,**models)

@torch.inference_mode()
def main():
    enc.check();tok=BertTokenizerFast.from_pretrained(enc.H/'tokenizer',local_files_only=True);m=enc.get(KIND)
    d,old=known_scores('fit');f=capture(m,tok,'fit');np.testing.assert_array_equal(f.hidden.astype(np.float16),d[KIND].hidden)
    cols,variants=patches(f,d,old,m);del f,d,old
    head=m.bert.encoder.layer[-1].output.dense
    source=head.weight[:,cols].float().cpu().numpy();source_b=head.bias.float().cpu().numpy()
    d,old=known_scores('select');base=d[KIND];cap=capture(m,tok,'select',cols)
    np.testing.assert_array_equal(cap.hidden.astype(np.float16),base.hidden)
    mask=scale.maskinfo('select',SimpleNamespace(case=base.case_ids,offsets=base.offsets))
    kk,bm,bn=scale.key(base.gold,old.argmax(1),mask)
    best=(kk,dict(name='donor',strength=0.,rule='doc',alpha=0.,scope='all'));grid=[]
    for name in ('donor','context'):
        for strength in (.30,.85,1.70):
            delta,db=actual_delta(variants[name],strength,source,source_b);hidden=replay(cap,m,delta,db)
            doc,dlog=modified(d,m,hidden)
            for typ in ('doc','raw_residual'):
                for alpha in (.16,.45,1.0):
                    for scope in ('all','pair'):
                        cfg=dict(name=name,strength=strength,rule=typ,alpha=alpha,scope=scope);p=rule(cfg,old,doc,dlog)
                        key,full,novel=scale.key(base.gold,p,mask)
                        ok=full['pair_errors']<=bm['pair_errors'] and novel['pair_errors']<=bn['pair_errors']
                        grid.append([cfg,key[0],full['pair_errors'],novel['pair_errors'],bool(ok)])
                        if ok and key>best[0]:best=(key,cfg)
    key,c=best
    # Also verify one nonzero native write when the selector selects the identity.
    ref=c if c['strength']>0 else dict(name='context',strength=.85,rule='doc',alpha=1.,scope='all')
    patch={};norms={}
    for typ,keyname in [('edit',ref['name']),('shuffled',ref['name']+'-shuffled')]:
        dw,db=actual_delta(variants[keyname],ref['strength'],source,source_b)
        patch[typ+'-weight']=dw;patch[typ+'-bias']=db
        norms[typ]=dict(delta_weight_norm=float(np.linalg.norm(dw)),delta_bias_norm=float(np.linalg.norm(db)),
                        changed_weights=int(np.count_nonzero(dw)),changed_biases=int(np.count_nonzero(db)))
    patch['cols']=cols;save_file(patch,str(H/'native-patch.safetensors'))
    scale.dump(H/'freeze.json',dict(config=c,physical_challenger=ref,selector=key[0],grid=grid,columns=len(cols),norms=norms,
               tensor='bert.encoder.layer.11.output.dense.weight and bias',source=enc.MODELS[KIND][1],
               patch_sha256=enc.sha(H/'native-patch.safetensors'),protocol_sha256=enc.sha(H/'protocol.json')))
    print('Frozen native patch',c,ref,len(cols),norms,flush=True)
    out={}
    for g in ('select','confirm','fresh'):
        d,old=known_scores(g);base=d[KIND];pp={'context':old.argmax(1)};ev={};masks=scale.maskinfo(g,SimpleNamespace(case=base.case_ids,offsets=base.offsets))
        for mode in ('edit','shuffled'):
            dw=patch[mode+'-weight'];db=patch[mode+'-bias'];head.weight[:,cols]=gput(source+dw).half();head.bias.copy_(gput(source_b+db).half())
            h=capture(m,tok,g,cols,True).hidden;doc,dr=modified(d,m,h)
            pp[mode]=rule(ref,old,doc,dr);pp['standalone-'+mode]=(h@m.classifier.weight.cpu().numpy().T+m.classifier.bias.cpu().numpy()).argmax(1)
            if g=='select':
                approx=replay(cap,m,dw,db);dd,dl=modified(d,m,approx);ev[mode]=dict(
                    replay_native_label_agreement=float((rule(ref,old,dd,dl)==pp[mode]).mean()),max_embedding_diff=float(np.abs(approx-h).max()))
            head.weight[:,cols]=gput(source).half();head.bias.copy_(gput(source_b).half())
        pp['standalone-original']=base.raw.argmax(1)
        pp['selected']=pp['edit'] if c['strength']>0 else pp['context']
        out[g]=dict(cases=len(base.case_ids),rows=len(base.gold),all={k:lib.metric(base.gold,v) for k,v in pp.items()},unmatched={k:lib.metric(base.gold[masks],v[masks]) for k,v in pp.items()},physical_comparison=ev)
        np.savez_compressed(H/(g+'-pred.npz'),**arraypack(base,pp))
        print(g,'actual',[(k,round(v['macro_f1']*100,3),v['pair_errors']) for k,v in out[g]['all'].items()],flush=True)
    scale.dump(H/'results.json',out)
    fs=[H/p for p in ('protocol.json','ffn.py','results.json','freeze.json','native-patch.safetensors','select-pred.npz','confirm-pred.npz','fresh-pred.npz')]
    destination=H.parent/'context-mlp-dev.zip'
    with ZipFile(destination.with_suffix('.part'),'w',ZIP_DEFLATED) as z:
        for p in fs:z.write(p,p.name)
        z.writestr('SHA256SUMS.json',json.dumps({p.name:enc.sha(p) for p in fs}))
    destination.with_suffix('.part').replace(destination)
if __name__=='__main__':main()
