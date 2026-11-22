"""Closed-form semantic-subspace ridge, norm-matched null, and judgment level scoring."""
import os, json, sys, argparse, hashlib, math
from pathlib import Path
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file,load_file
P=Path(__file__).resolve().parent; PR=json.loads((P/'protocol.json').read_text()); STORE=P/'features';OUT=P/'results';OUT.mkdir(exist_ok=True)
NAMES=('start_boundary_projection','end_boundary_projection','inside_text_projection'); KEYS=['boundary_head.boundary_query_head.'+k+'.weight' for k in NAMES]
REGS=(0.03,0.3,3.0); GAINS=(0.5,1.,2.); MASKS=((0,1,2),(2,),(0,1),(0,),(1,))

def readcases(split,ids):
    res={}
    for k in ids:
        with np.load(STORE/f'{split}-{k}.npz') as f:res[k]={name:f[name] for name in f.files}
    return res

def merge_rows(a,values):
    w=a['weights']; totals=np.bincount(a['obs'],weights=w,minlength=len(a['y']));total=np.zeros((len(totals),7),dtype='f8');np.add.at(total,a['obs'],w[:,None]*values);return total/totals[:,None]

def native(a):return merge_rows(a,a['native'].mean(axis=1))

def confusion(y,p):return np.bincount(7*y.astype('i8')+p.astype('i8'),minlength=49).reshape(7,7)

def scores_cm(c):
    d=np.diag(c); denom=c.sum(0)+c.sum(1);f=np.divide(2*d,denom,out=np.zeros(7),where=denom>0)
    return dict(macro_f1=float(f.mean()),accuracy=float(d.sum()/c.sum()),role_f1=f.tolist())

def runs(y):
    edges=np.flatnonzero(np.r_[True,y[1:]!=y[:-1],True]);return {(int(a),int(b),int(y[a])) for a,b in zip(edges[:-1],edges[1:])}

def span_counts(y,p):
    a,b=runs(y),runs(p);return np.array([len(a&b),len(a),len(b)])

def metric(c,sp):
    r=scores_cm(c); r.update(exact_span_micro_f1=float(2*sp[0]/(sp[1]+sp[2])),exact_matches=int(sp[0]),gold_spans=int(sp[1]),predicted_spans=int(sp[2]),rows=int(c.sum()));return r

def evaluate(cases,pred):
    cms=[];spans=[]; breakdown={}
    for k,a in cases.items():
        y=a['y'];pv=pred[k];assert len(y)==len(pv),(k,len(y),len(pv));c=confusion(y,pv);s=span_counts(y,pv);cms.append(c);spans.append(s);breakdown[k]=metric(c,s)
    tc=np.sum(cms,axis=0);overall=metric(tc,np.sum(spans,axis=0));overall['confusion']=tc.tolist();overall['support']=tc.sum(1).tolist();overall['judgments']=len(cases);return overall,breakdown,np.array(cms),np.array(spans)

def build(cases,shuffle=False):
    # Label permutations are by sentence, preserving a split sentence's common label.
    ys=np.concatenate([a['y'] for a in cases.values()]);ys=ys if not shuffle else np.random.default_rng(901112).permutation(ys)
    count=np.bincount(ys,minlength=7);outputs={reg:[] for reg in REGS}
    for ch in range(3):
        qbar=np.concatenate([a[f'q{ch}'] for a in cases.values()]).mean(0).astype('f8');basis=np.linalg.svd(qbar-qbar.mean(0),full_matrices=False)[2][:6]
        allx=[];targets=[];ws=[];ptr=0
        for a in cases.values():
            ix=a['obs'];qb=a['qb'];y=ys[ptr:ptr+len(a['y'])][ix]; ptr+=len(a['y']);w=a['weights']/np.bincount(ix,weights=a['weights'])[ix];w/=np.sqrt(np.maximum(count[y],1))
            q=a[f'q{ch}'].astype('f8'); A=(q-q.mean(1,keepdims=True))@basis.T/np.sqrt(q.shape[-1]);pinv=np.linalg.pinv(A,rcond=.05)
            pre=a['native'][:,ch].astype('f8');des=(np.eye(7)[y]-1/7)*6-(pre-pre.mean(axis=1,keepdims=True)); z=np.einsum('brd,bd->br',pinv[qb],des)
            allx.append(a[f'x{ch}']);targets.append(z);ws.append(w)
        X=np.concatenate(allx).astype('f8');Y=np.concatenate(targets);w=np.concatenate(ws);gram=(X.T*w)@X;cross=(X.T*w)@Y;scale=np.trace(gram)/len(gram)
        for reg in REGS:
            coef=np.linalg.solve(gram+reg*scale*np.eye(len(gram)),cross); outputs[reg].append(np.asarray(basis.T@coef.T,'f4'))
    return outputs

def extra(a,delta,ch):
    q=a[f'q{ch}'];proj=a[f'x{ch}']@delta.T; return np.einsum('bd,bcd->bc',proj,q[a['qb']])/np.sqrt(q.shape[-1])

def get_pred(cases,ds):
    out={}
    for k,a in cases.items():
        val=a['native'].sum(1).astype('f8');
        for ch,d in enumerate(ds):
            if np.any(d):val+=extra(a,d,ch)
        out[k]=merge_rows(a,val/3).argmax(1)
    return out

def hsh(x):return hashlib.sha256(x.astype('<f4').tobytes()).hexdigest()

def fit():
    from joint_solver import joint
    train=readcases('train',PR['fit']);sel=readcases('train',PR['select']);independent=build(train);joint_model,jdiag=joint(train,REGS); ds={(method,reg):v for method,table in [('local_inverse',independent),('joint_normal_equations',joint_model)] for reg,v in table.items()}
    cached={};native_sel={k:native(a) for k,a in sel.items()};native_pred={k:v.argmax(1) for k,v in native_sel.items()};baseline=evaluate(sel,native_pred)[0]['macro_f1'];opts=[]
    for (method,reg),mats in ds.items():
        for k,a in sel.items():cached[(method,reg,k)]=[merge_rows(a,extra(a,mats[ch],ch))/3 for ch in range(3)]
        for chans in (MASKS if method=='local_inverse' else ((0,1,2),)):
            for gain in GAINS:
                preds={k:(native_sel[k]+gain*sum((cached[(method,reg,k)][ch] for ch in chans))).argmax(1) for k in sel}; r=evaluate(sel,preds)[0];opts.append(dict(method=method,reg=reg,channels=list(chans),gain=gain,macro_f1=r['macro_f1'],accuracy=r['accuracy'],exact_span_micro_f1=r['exact_span_micro_f1']))
    opts.sort(key=lambda s:(-s['macro_f1'],len(s['channels']),s['gain']));winner=opts[0]; wreg=winner['reg']; control=(build(train,shuffle=True)[wreg] if winner['method']=='local_inverse' else joint(train,(wreg,),shuffle=True)[0][wreg]); ds=ds[winner['method'],wreg]; adjusted=[]; null=[]
    for ch,(d,c) in enumerate(zip(ds,control)):
        norm=np.linalg.norm(d)/max(np.linalg.norm(c),1e-10);alpha=winner['gain'] if ch in winner['channels'] and winner['macro_f1']>baseline else 0;adjusted.append(np.asarray(d*alpha,'f4'));null.append(np.asarray(c*(norm*alpha),'f4'))
    with safe_open(P/'model'/'model.safetensors',framework='np') as ck:base=[ck.get_tensor(name) for name in KEYS]; n_params=sum(math.prod(ck.get_slice(k).get_shape()) for k in ck.keys())
    info={}
    for i,(key,d,c) in enumerate(zip(KEYS,adjusted,null)):
        s=np.linalg.svd(d.astype('f8'),compute_uv=False);info[key]=dict(shape=list(d.shape),entries=d.size,changed_entries=int(np.count_nonzero(base[i]+d!=base[i])),frobenius=float(np.linalg.norm(d)),null_frobenius=float(np.linalg.norm(c)),numerical_rank=int((s>max(s[0]*1e-5,1e-7)).sum()),native_sha256=hsh(base[i]),edited_sha256=hsh(base[i]+d),null_sha256=hsh(base[i]+c))
    for label,data in [('edit',adjusted),('shuffled',null)]: save_file(dict(zip(KEYS,data)),str(OUT/(label+'-delta.safetensors')));save_file({k:w+d for k,w,d in zip(KEYS,base,data)},str(OUT/(label+'-replacement.safetensors')))
    bestinfo=dict(training_rows=sum(len(a['y']) for a in train.values()),select_rows=sum(len(a['y']) for a in sel.values()),fit_label_support=np.bincount(np.concatenate([a['y'] for a in train.values()]),minlength=7).tolist(),selector_unchanged=baseline,selected=winner,joint_solver=jdiag,checkpoint_parameters=n_params,tensors=info,all_selection=opts)
    (OUT/'freeze.json').write_text(json.dumps(bestinfo,indent=2))
    files=[P/n for n in ('protocol.json','analytic-specification.json','collect.py','score.py','joint_solver.py','verify_edits.py','classification-sample.json','classifier64.py')];files+=[OUT/n for n in ('freeze.json','edit-delta.safetensors','shuffled-delta.safetensors','edit-replacement.safetensors','shuffled-replacement.safetensors')]
    locked={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files};(OUT/'freeze-sha256.json').write_text(json.dumps(locked,indent=2));print('selection',winner,'baseline',baseline,flush=True)

def bootstrap(stat,ref):
    cs,ss=stat; cr,sr=ref;n=len(cs);rng=np.random.default_rng(87931); draws=np.array([rng.integers(0,n,n) for _ in range(2000)]); out=[]
    for ix in draws:
        a=metric(cs[ix].sum(0),ss[ix].sum(0));b=metric(cr[ix].sum(0),sr[ix].sum(0));out.append([a[k]-b[k] for k in ('macro_f1','accuracy','exact_span_micro_f1')])
    ci=np.quantile(out,[.025,.975],axis=0);return {k:ci[:,i].tolist() for i,k in enumerate(('macro_f1','accuracy','exact_span_micro_f1'))}

def report(part):
    ids=PR[{'check':'check','four':'interim','sixteen':'validation'}[part]];split='train' if part=='check' else 'val';cases=readcases(split,ids); pred={'native':{k:native(a).argmax(1) for k,a in cases.items()}}
    for v in ('edit','shuffled'):
        d=load_file(str(OUT/(v+'-delta.safetensors')));pred[v]=get_pred(cases,[d[key] for key in KEYS]);
    if split=='val':pred['incumbent']={k:np.asarray(json.loads((P.parent/'pooler'/'shared-cases'/(k+'.json')).read_text())['predictions']['unchanged']) for k in cases}
    lock=json.loads((OUT/'freeze-sha256.json').read_text());assert all(hashlib.sha256((P/rel).read_bytes()).hexdigest()==dig for rel,dig in lock.items() if rel.startswith('results/'));summary={};by_case={};stats={}
    for name,vals in pred.items():summary[name],by_case[name],c,sp=evaluate(cases,vals);stats[name]=(c,sp)
    obj=dict(freeze_sha256=lock['results/freeze.json'],split=split,ids=ids,aggregate=summary,by_case=by_case,paired_edit_minus={k:bootstrap(stats['edit'],stats[k]) for k in stats if k!='edit'},case_fragment_diagnostics={k:{'sentences':len(a['y']),'fragments':len(a['obs']),'blocks':len(a['subwords']),'max_block_wordpieces':int(a['subwords'].max()),'native_reconstruction_max_err':float(max(a['parity']))} for k,a in cases.items()}); (OUT/(part+'.json')).write_text(json.dumps(obj,indent=2));(OUT/(part+'-predictions.json')).write_text(json.dumps({name:{k:v.tolist() for k,v in p.items()} for name,p in pred.items()}));print(part,{k:{t:round(x[t],6) for t in ('macro_f1','accuracy','exact_span_micro_f1','rows')} for k,x in summary.items()},flush=True)
if __name__=='__main__':
    if sys.argv[1]=='fit':fit()
    else:
        report(sys.argv[1])
        if sys.argv[1]=='four' and not (OUT/'publisher-classify-64.json').exists():
            import subprocess
            # A secondary probe must never suppress a primary result. Keep any failure explicit.
            proc=subprocess.run([sys.executable,'-u',str(P/'classifier64.py')],cwd=P)
            if proc.returncode:print('publisher-classify-64 encountered exit code',proc.returncode,flush=True)
