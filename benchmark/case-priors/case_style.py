"""Estimate judgment-specific role priors from disjoint training judgments using a closed-form kernel readout."""
import argparse,csv,hashlib,itertools,json,re,time
from pathlib import Path
import numpy as np
from scipy.linalg import cho_solve,cho_factor
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from safetensors.numpy import save_file
H=Path(__file__).resolve().parent
R=H.parents[1];LIVE=R/'live-nov22';PLAN=json.loads((LIVE/'reviewer-round2/round2-plan.json').read_text())
ROLES=['None','Facts','Issue','Arguments of Petitioner','Arguments of Respondent','Reasoning','Decision'];EYE=np.eye(7,dtype=np.float32)
BINS=16
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,obj):
    path=H/name;p=path.with_suffix('.tmp');p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');p.replace(path)
def txt(v):
    return re.sub(r'\b\d{4,}\b','number',v.casefold())
def sources(split,need,duplicates=False):
    src=R/'references/official'/(split+'.csv')
    res={}
    with src.open(newline='',encoding='utf-8-sig') as f:
        for key,g in itertools.groupby(csv.DictReader(f),lambda r:r['Index']):
            if need is not None and key not in need:continue
            lines=[];ys=[];fp=hashlib.sha256()
            for row in g:
                s=row['Text'];t=s.strip()
                if not t:continue
                fp.update(' '.join(s.casefold().split()).encode()+b'\0')
                lines.append(s);ys.append(ROLES.index(row['Label']))
            assert key not in res
            n=len(lines);assert n
            pos=(np.arange(n)+.5)/n
            ix=np.minimum((BINS*pos).astype(int),BINS-1)
            count=np.bincount(7*ix+np.array(ys),minlength=BINS*7).reshape(BINS,7)
            # Separate header, footer and 24 evenly spaced text excerpts, all available on a new judgment.
            snippets=[lines[j][:260] for j in np.unique(np.linspace(0,n-1,min(24,n)).astype(int))]
            texts=[txt(' '.join(lines[:7])[:2400]),txt(' '.join(lines[-7:])[-2400:]),txt(' '.join(snippets))]
            lens=np.array([len(t) for t in lines]);joined=' '.join(snippets);nt=max(len(joined),1)
            style=np.r_[np.log1p(n)/6,np.log1p(np.quantile(lens,[.1,.25,.5,.75,.9]))/6,(lens<45).mean(),(lens>275).mean(),
                        sum(x.isupper() for x in joined)/nt,sum(x.isdigit() for x in joined)/nt,
                        [joined.lower().count(v)/max(len(snippets),1) for v in [' vs',' v.','appeal','petition','lordship','counsel','court','section','tribunal','judge',' held','we ',' the ']]]
            res[key]=dict(fp=fp.hexdigest(),views=texts,style=style,count=count,n=n)
    return res
def prep():
    train=sources('train',None);val=sources('val',None)
    valfps={v['fp'] for v in val.values()}; groups=PLAN['groups'];panel=set(sum(groups.values(),[]))
    used=set(valfps)|{train[k]['fp'] for k in groups['fit']};mem=[]
    for k in train:
        if k not in panel and train[k]['fp'] not in used:
            mem.append(k);used.add(train[k]['fp'])
    keys=mem+sum([groups[k] for k in ('fit','select','confirm','fresh')],[])
    docs=[(train[k] if k in train else val[k]) for k in keys]
    M=len(mem);assert len({x['fp'] for x in docs})==len(docs)
    saved={'memory':mem,'groups':groups,'bin_count':BINS,'duplicates':'Exact full-case normalized texts removed against every validation judgment and 384 calibration judgments; no IDs or gold used as predictive features.',
           'stage1':'TF-IDF header/footer/distributed excerpts and textual formatting; two profile estimators use independent train judgments. Separate prior target for 16 positions and 7 roles.',
           'selection':'48 cases. Confirm 72 and 96 by fixed rule after locking. Prior project used the validation cases.',
           'sources':{g:sha(R/'references/official'/(g+'.csv')) for g in ('train','val')}}
    save('protocol.json',saved)
    bank=[]
    for col,weight in enumerate((.38,.19,.43)):
        v=TfidfVectorizer(max_features=36000,ngram_range=(1,2),min_df=2,sublinear_tf=True,dtype=np.float32,strip_accents='unicode')
        text=[d['views'][col] for d in docs];v.fit(text[:M]);bank.append(v.transform(text)*np.sqrt(weight))
    X=hstack(bank,format='csr',dtype=np.float32)
    style=np.stack([d['style'] for d in docs]);std=np.maximum(style[:M].std(0),.012);st=np.clip((style-style[:M].mean(0))/std,-6,6).astype(np.float32)
    count=np.stack([x['count'] for x in docs[:M]]).astype(np.float32)
    prior=(count.sum(0)+1)/(count.sum((0,2))[:,None]+7)
    roll=count.copy()
    roll[:,1:]+=count[:,:-1]*.5;roll[:,:-1]+=count[:,1:]*.5
    target=((roll+4*prior)/(roll.sum(-1,keepdims=True)+4)).reshape(M,-1)
    from scipy.spatial.distance import cdist
    def sim(ia,ib):
        simtext=(X[ia]@X[ib].T).toarray()
        dist=cdist(st[ia],st[ib],'sqeuclidean')/st.shape[1]
        return (.79*simtext+.21*np.exp(-dist*.7)).astype(np.float32)
    rnd=np.random.default_rng(261122);anchors=np.sort(rnd.choice(M,min(900,M),replace=False))
    k=sim(np.arange(len(docs)),anchors);mass=k[:M].mean(0);s=np.maximum(k[:M].std(0),.012)
    x=np.clip((k-mass)/s,-7,7)
    x=np.column_stack((x,st,np.ones(len(x),np.float32)))
    gram=x[:M].T@x[:M];rhs=x[:M].T@target
    reg=np.maximum(np.diag(gram),1)*.3;reg[-1]*=.001
    w=cho_solve(cho_factor(gram.astype(np.float64)+np.diag(reg)),rhs).astype(np.float32)
    perm=rnd.permutation(M);sh=cho_solve(cho_factor(gram.astype(np.float64)+np.diag(reg)),x[:M].T@target[perm]).astype(np.float32)
    # Rank-112 weight table writes case-profile priors directly into 16 × 7 output scores.
    save_file({'matrix':w,'shuffled_matrix':sh,'kernel_center':mass,'kernel_scale':s,'landmarks':anchors,'global_prior':prior},str(H/'case-weight.safetensors'))
    todo=np.arange(M,len(docs));near=sim(todo,np.arange(M))
    ix=np.argpartition(near,-30,axis=1)[:,-30:];a=np.take_along_axis(near,ix,axis=1)
    a=np.exp(11*(a-a.max(1,keepdims=True)));a/=a.sum(1,keepdims=True)
    nprior=np.sum(a[:,:,None]*target[ix],axis=1).reshape(-1,BINS,7)
    def panel(v):
        p=np.maximum(v.reshape(-1,BINS,7),.0015);return p/p.sum(2,keepdims=True)
    out=H/'profiles.npz'
    np.savez_compressed(out,case_ids=np.array(keys[M:]),ridge=panel(x[M:]@w),nearest=nprior,shuffled=panel(x[M:]@sh),global_prior=prior,doc_style=st[M:])
    print('case profiles',M,len(todo),sum(docs[i]['n'] for i in range(M)),sha(out),flush=True)
if __name__=='__main__':prep()
