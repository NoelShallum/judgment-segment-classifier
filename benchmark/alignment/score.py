"""Freeze vocabulary-free retrieval gates and audit seen, unseen and near-duplicate judgment slices."""
import argparse,itertools,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from memory import H,sha,save
def bank(group):
    with np.load(H/(group+'-bank.npz'),allow_pickle=False) as f:return SimpleNamespace(**{k:f[k] for k in f.files})
def cm(y,p):
    a=np.bincount(y.astype(np.int64)*7+p.astype(np.int64),minlength=49).reshape(7,7)
    return a
def f1(a):
    return (2*np.diagonal(a,axis1=-2,axis2=-1)/np.maximum(a.sum(-1)+a.sum(-2),1)).mean(-1)
def metric(a):
    f=2*a.diagonal()/np.maximum(a.sum(0)+a.sum(1),1)
    return dict(rows=int(a.sum()),macro_f1=float(f.mean()),accuracy=float(a.trace()/max(a.sum(),1)),none_to_reason=int(a[0,5]),reason_to_none=int(a[5,0]),pair=int(a[0,5]+a[5,0]),f1=f.tolist(),confusion=a.tolist())
def predict(d,c,mode='targeted'):
    v=d.shuffled if mode=='shuffled' else d.votes;top=v.argmax(1)
    consensus=v.max(1)/np.maximum(v.sum(1),1e-10);mask=(d.quality>=c['quality'])&(consensus>=c['agreement'])&(d.block>=c['block'])
    if c['scope']=='pair':mask&=np.isin(d.base,[0,5])&np.isin(top,[0,5])
    if mode=='exactrows':mask&=d.exact
    if mode=='no_nearduplicate_cases':mask&=d.match_ratio<.8
    return np.where(mask,top,d.base).astype(np.int8),mask
def fit():
    d=bank('select');base=metric(cm(d.gold,d.base));selected={}
    for scope in ('all','pair'):
        candidates=[]
        for quality,agreement,block in itertools.product((.62,.75,.88,.95),(.58,.85,.995),(2,4,8)):
            cfg=dict(quality=quality,agreement=agreement,block=block,scope=scope)
            p,eligible=predict(d,cfg);m=metric(cm(d.gold,p))
            if m['pair']<=base['pair']:candidates.append(((m['macro_f1'],-m['pair'],m['accuracy'],-int(eligible.sum())),cfg,m))
        if candidates:
            _,cfg,metrics=max(candidates,key=lambda x:x[0]);selected[scope]=dict(config=cfg,scores=metrics)
        else:selected[scope]=dict(config=dict(quality=2,agreement=1,block=99,scope=scope),scores=base)
    save('freeze.json',dict(arms=selected,primary='all',training=sha(H/'source.json'),selection_input=sha(H/'select-bank.npz'),selection_baseline=base,protocol='Use the seven-class whole-judgment classifier, plus a train-only one-hot memory value matrix. Declare the all-role arm primary before confirmation. Report unchanged/unseen text, judgments without reliable anchors, near-duplicate judgments and independent label-permutation control.'))
def paired(d,a,b):
    f=lambda p:np.stack([cm(d.gold[l:r],p[l:r]) for l,r in zip(d.offsets,d.offsets[1:])])
    va,vb=f(a),f(b);i=np.random.default_rng(61028).integers(0,len(va),(2500,len(va)))
    # Bound peak memory for the official test bootstrap.
    ds=[];qs=[]
    for ix in np.array_split(i,20):
        v,w=va[ix].sum(1),vb[ix].sum(1)
        ds.extend(f1(v)-f1(w));qs.extend(v[:,0,5]+v[:,5,0]-w[:,0,5]-w[:,5,0])
    return dict(macro_delta=np.quantile(ds,[.025,.975]).tolist(),pair_count_delta=np.quantile(qs,[.025,.975]).tolist())
def score(groups):
    lock=json.loads((H/'freeze.json').read_text());assert lock['training']==sha(H/'source.json')
    allres={}
    for g in groups:
        d=bank(g);cfg=lock['arms']['all']['config']
        pred={'incumbent':d.incumbent,'reviewer':d.base};pred['all'],mask=predict(d,cfg);pred['pair']=predict(d,lock['arms']['pair']['config'])[0]
        for mode in ('shuffled','exactrows','no_nearduplicate_cases'):pred[mode]=predict(d,cfg,mode)[0]
        met={k:metric(cm(d.gold,p)) for k,p in pred.items()}
        if g=='select':assert met['all']['confusion']==lock['arms']['all']['scores']['confusion']
        cuts={'eligible_text':mask,'no_eligible_text':~mask,'no_document_anchor':d.match_ratio==0,'doc_alignment_ge_80pct':d.match_ratio>=.8,
              'doc_alignment_lt_80pct':d.match_ratio<.8,'exact_full_case':d.same_case,'aligned_text_varies':mask&(~d.exact),'aligned_text_equal':mask&d.exact}
        sub={k:dict(rows=int(c.sum()),judgments=int(sum(c[a:b].any() for a,b in zip(d.offsets,d.offsets[1:]))),baseline=metric(cm(d.gold[c],d.base[c])) if c.any() else None,edited=metric(cm(d.gold[c],pred['all'][c])) if c.any() else None) for k,c in cuts.items()}
        p=pred['all'];diff=p!=d.base
        allres[g]=dict(cases=len(d.case_ids),rows=len(d.gold),scores=met,slices=sub,paired_vs_reviewer=paired(d,p,d.base),paired_vs_old=paired(d,p,d.incumbent),
                       changes=int(diff.sum()),fixed=int(((p==d.gold)&(d.base!=d.gold)).sum()),harmed=int(((p!=d.gold)&(d.base==d.gold)).sum()),
                       unchanged_unmatched=int((p[~mask]!=d.base[~mask]).sum()))
        np.savez_compressed(H/(g+'-predictions.npz'),gold=d.gold,offsets=d.offsets,case_ids=d.case_ids,detection=mask,near_case=(d.match_ratio>=.8),same_case=d.same_case,**pred)
        print(g,{k:(round(met[k]['macro_f1']*100,3),met[k]['pair']) for k in ('reviewer','all','pair','exactrows','shuffled')},'coverage',int(mask.sum()),'near',sub['doc_alignment_ge_80pct']['judgments'],flush=True)
    save('official.json' if 'test'in groups else 'evaluation.json',dict(lock_sha256=sha(H/'freeze.json'),groups=allres))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=('fit','score','official'));a=ap.parse_args()
    if a.phase=='fit':fit()
    elif a.phase=='score':score(['select','confirm','fresh'])
    else:score(['test'])
