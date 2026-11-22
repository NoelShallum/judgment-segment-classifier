"""All 117,065 labeled sentence vectors fit a gradient-free head; add lexical evidence through a physical donor embedding write."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import scipy.linalg as la
from scipy.sparse import csr_matrix
from safetensors import safe_open
from safetensors.numpy import save_file
from tokenizers import Tokenizer
import torch
from torch import nn
from prepare import ROOT,LABELS,docs,save
from extract import groups
from study import metrics,compare,h
from joint import decode
from vocab import SRC,V,PERM,raw_data,score_tokens,priors,target_logs
from uncertainty import interval

def from_ids(ids):
    lengths=np.fromiter((len(q) for q in ids),dtype=np.int64)
    indptr=np.r_[0,np.cumsum(lengths)]
    w=np.repeat(1./np.sqrt(lengths),lengths).astype(np.float32)
    return csr_matrix((w,np.concatenate(ids).astype(np.int32),indptr),shape=(len(ids),V))


def stat(emb,tok):
    path=ROOT/'sentence-vocab-gram.npz'
    if path.exists():return
    fits=set(json.loads((ROOT/'vocab-counts.json').read_text())['fit_ids'])
    gram=np.zeros((7,768,768));sumx=np.zeros((7,768));freq=np.zeros(7,int)
    count=0;raw=[];ys=[]
    def add(items,ys):
        X=(from_ids(items)@emb).astype(float)
        y=np.array(ys)
        for i in np.unique(y):
            xi=X[y==i];gram[i]+=xi.T@xi;sumx[i]+=xi.sum(0);freq[i]+=len(xi)
    for cid,rs in docs('train.csv'):
        if cid not in fits:continue
        enc=tok.encode_batch([r["text"] for r in rs],add_special_tokens=False)
        raw.extend(np.unique(item.ids) for item in enc);ys.extend(r['y'] for r in rs)
        count+=1
        if len(raw)>=2048:
            add(raw,ys);raw.clear();ys.clear()
        if count%100==0:print('sentence-head cases',count,'accumulated rows',freq.sum(),flush=True)
    if raw:add(raw,ys)
    assert count==len(fits)
    np.savez_compressed(path,gram=gram,sumx=sumx,freq=freq)
    print('stats',count,freq.tolist(),flush=True)


def head(gram,sumx,freq,powr,gain):
    weights=(freq/freq.mean())**(-powr)
    count=(weights*freq).sum()
    wsum=(weights[:,None]*sumx).sum(0)
    mean=wsum/count
    ysum=weights*freq;yt=ysum/count
    A=np.tensordot(weights,gram,axes=1)-np.outer(wsum,mean)
    B=sumx.T*weights-np.outer(wsum,yt)
    lam=gain*np.trace(A)/A.shape[0]
    A.flat[::769]+=lam
    W=la.solve(A,B,assume_a='pos',check_finite=False)
    b=yt-mean@W
    return W,b


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","1")));torch.set_num_interop_threads(1)
    root=ROOT/'sentence-vocab';root.mkdir(exist_ok=True)
    with safe_open(str(ROOT/'model/model.safetensors'),framework='numpy') as ck:emb=ck.get_tensor(SRC)
    tok=Tokenizer.from_file(str(ROOT/'model/tokenizer.json'))
    stat(emb,tok)
    with np.load(ROOT/'sentence-vocab-gram.npz') as z:gram=z['gram'];sums=z['sumx'];freq=z['freq']
    with np.load(ROOT/'vocab-counts.npz') as z:df=z['token_document_frequency'];pp=z['position'];tx=z['log_transition'];np.testing.assert_array_equal(freq,z['class_counts'])
    gs=groups('screen');ds={n:raw_data(gs[n],tok) for n in ['select','confirm']}
    dr=ds['select'];LX=from_ids(dr['ids']);x=LX@emb
    tables={s:target_logs(df,freq,s) for s in [20.,500.,5000.]}
    scores={s:LX@v for s,v in tables.items()}
    pr=priors(dr,pp);y=dr['y'];grid=[];heads={}
    # Uniform, preregistered choices. Fit and select use only official training cases.
    for powr in [0.,.65]:
        for reg in [.03,.3,1.5]:
            w,b=head(gram,sums,freq,powr,reg);heads[powr,reg]=(w,b)
            native=x@w+b
            for shrink,g in [(20.,0.),*[(s,g) for s in [20.,500.,5000.] for g in [.05,.2,.8]]]:
                score=native+g*scores[shrink]
                for pc in [0.,.1,.4]:
                    for tc in [0.,.08,.3]:
                        pred=decode(dr,score+pc*pr,tx,tc)
                        grid.append([metrics(y,pred)['macro_f1_7'],powr,reg,shrink,g,pc,tc])
    find=lambda condition:max((q for q in grid if condition(q)),key=lambda q:q[0])
    plans={"best_edited_text":find(lambda a:a[4]>0 and a[5]==a[6]==0),
           "best_zero_text":find(lambda a:a[4]==0 and a[5]==a[6]==0),
           "best_edited_structured":find(lambda a:a[4]>0),"best_zero_structured":find(lambda a:a[4]==0)}
    primary_name=max(['best_edited_text','best_edited_structured'],key=lambda k:plans[k][0])
    fp=root/'freeze.json'
    freeze={"fit_cases":495,"fit_rows":int(freq.sum()),"sufficient_statistics_sha256":digest(ROOT/'sentence-vocab-gram.npz'),
       "source_train_counts_sha256":digest(ROOT/'vocab-counts.npz'),"selection_ids":[i['id'] for i in gs['select']],"confirmation_ids":[i['id'] for i in gs['confirm']],
       "official_validation_used_for_fitting_or_hyperparameter_choice":False,
       "prospective_scope":"supplemental all-sentence frozen native control and residual embedding write, devised after the preceding, independent lexical model was scored on shared16. A separate extended 32 judgments can give a fresh check.",
       "config_order":["seven-class selection macro-F1","class balance power","ridge penalty","word smoothing","word-logit gain","document prior gain","transition gain"],
       "primary_name":primary_name,"plans":plans,"grid":grid}
    if not fp.exists():save(fp,freeze)
    assert json.loads(fp.read_text())==freeze
    actual={}
    for name in ['best_edited_text','best_edited_structured']:
        _,powr,reg,shrink,g,pc,tc=plans[name]
        W,b=heads[powr,reg]
        role=g*tables[shrink];lift=la.pinv(W,rtol=1e-6,check_finite=False)
        mat=nn.Embedding(V,768)
        with torch.no_grad():mat.weight.copy_(torch.from_numpy(emb))
        digest_value=hashlib.sha256();d2=0.;normo=0.;nnz=0
        effective=np.empty((V,7));baseline=emb@W
        for i in range(0,V,1024):
            orig=emb[i:i+1024]
            val=(orig+(role[i:i+1024]@lift).astype(np.float32)).astype(np.float32)
            with torch.no_grad():mat.weight[i:i+len(val)].copy_(torch.from_numpy(val))
            phys=mat.weight[i:i+len(val)].detach().numpy();digest_value.update(phys.tobytes())
            diff=phys.astype(float)-orig;d2+=np.sum(diff*diff);normo+=np.sum(orig.astype(float)**2);nnz+=np.count_nonzero(diff)
            effective[i:i+len(val)]=phys@W
        # W is centered across seven classes. Its one-dimensional null-space is irrelevant to multiclass decisions.
        residual=effective-baseline
        role_centered=role-role.mean(1,keepdims=True);res_centered=residual-residual.mean(1,keepdims=True)
        q1=la.qr(role,mode='economic')[1];q2=la.qr(lift.T,mode='economic')[1];sing=la.svdvals(q1@q2.T)
        receipt={"name":SRC,"base_sha256":h(emb),"materialized_sha256":digest_value.hexdigest(),"changed_entries":int(nnz),
                 "shape":list(emb.shape),"dtype":"float32","relative_frobenius":float(np.sqrt(d2/normo)),"rank_at_rel_1e-6":int((sing>sing[0]*1e-6).sum()),
                 "max_abs_centered_logit_error":float(abs(role_centered-res_centered).max())}
        save_file({"embedding.role_residual":np.ascontiguousarray(role),"role.to_hidden":np.ascontiguousarray(lift),
                   "head.weight":np.ascontiguousarray(W.T),"head.bias":np.asarray(b),"transition.logp":tx,
                   "position.logp":np.log((pp+.5)/(pp.sum(1,keepdims=True)+3.5))},str(root/f'{name}.safetensors'),
                  metadata={"tensor":SRC,"materialize":"source + float32(role_residual @ role.to_hidden) with FP32 addition"})
        actual[name]=(effective,baseline,residual,b,receipt)
        del mat
    sp=groups('common')['common'];ds['shared16']=raw_data(sp,tok)
    with np.load(ROOT/'vocab-common16-predictions.npz') as f:inc=f['incumbent'];np.testing.assert_array_equal(ds['shared16']['row'],f['source_csv_row'])
    rep={'freeze_sha256':digest(fp),'design':'The head minimizes ridge loss on every token-pooled labeled sentence from the same 495 cases. The full embedding change adds class-specific lexical evidence through the six readable margin directions.',
         "scope":"Input-embedding-only lexical readout. All sentence pooling is on unique tokenizer vocabulary IDs. No contextual encoder forward enters this method.","results":{},'tensors':{}}
    for name in ['best_edited_text','best_edited_structured']:
        _,powr,reg,shrink,g,pc,tc=plans[name]
        n= 'best_zero_text' if name.endswith('_text') else 'best_zero_structured'
        _,p0,r0,s0,g0,pc0,tc0=plans[n]
        w0,b0=heads[p0,r0];zero0=emb@w0
        ef,orig,res,b,detail=actual[name];rep['tensors'][name]=detail
        for n,d in ds.items():
            xp=from_ids(d['ids'])
            pos=priors(d,pp)
            pred=decode(d,xp@ef+b+pc*pos,tx,tc)
            base=decode(d,xp@orig+b+pc*pos,tx,tc)
            shuffle=decode(d,xp@(orig+res[PERM])+b+pc*pos,tx,tc)
            best0=decode(d,xp@zero0+b0+pc0*pos,tx,tc0)
            mm={k:metrics(d['y'],v) for k,v in [('edited',pred),('matched_zero',base),('optimized_zero',best0),('equal_norm_shuffle',shuffle)]}
            mm['paired_vs_optimized_zero']=interval(d['case'],d['y'],best0,pred)
            if n=='shared16':
                mm['incumbent']=metrics(d['y'],inc);mm['paired_vs_incumbent']=interval(d['case'],d['y'],inc,pred)
            rep['results'].setdefault(n,{})[name]=mm
            np.savez_compressed(root/f'{name}-{n}.npz',case=d['case'],ix=d['ix'],row=d['row'],start=d['start'],end=d['end'],gold=d['y'],edited=pred,matched_zero=base,optimized_zero=best0,shuffled=shuffle)
            print(name,n,[round(mm[k]['macro_f1_7'],5) for k in ['edited','optimized_zero','matched_zero','equal_norm_shuffle']],flush=True)
    save(root/'report.json',rep)
if __name__=='__main__':main()
