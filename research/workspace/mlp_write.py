"""Write prescribed associations into the final BERT feed-forward projection without optimizer steps."""
import itertools,json
from pathlib import Path
import numpy as np
from scipy.linalg import solve
from program_trial import fixture_rows,metric,official_sample,save_json
from role_fixtures import ANCHORS,CHECKS
from seven_paths import ROOT,LABELS,FOLDERS,folder_for
from weight_edit import tensor,patched_checkpoint,file_hash
from encoder import write_config

D=ROOT/'cache/mlp-11-states'
PARENT=ROOT/'checkpoints/associative-direct/model.safetensors'
TARGET=ROOT/'checkpoints/associative-mlp'
LAYER='bert.encoder.layer.11.output.'
def material():
    return [tensor(PARENT,k) for k in (LAYER+'LayerNorm.weight',LAYER+'LayerNorm.bias','bert.pooler.dense.weight','bert.pooler.dense.bias','classifier.weight','classifier.bias')]

def compute(r,mat):
    gamma,beta,pw,pb,w,b=mat
    center=r-r.mean(1,keepdims=True);scale=np.sqrt((center**2).mean(1,keepdims=True)+1e-12)
    unit=center/scale
    pooled=np.tanh((unit*gamma+beta)@pw.T+pb)
    return pooled@w.T+b,pooled,unit,scale

def reduce_windows(z,o,m):
    out=np.zeros((int(o.max())+1,7));div=np.bincount(o,weights=m)
    np.add.at(out,o,m[:,None]*z);return out/div[:,None]

def batch(name):
    with np.load(D/(name+'.npz')) as d:return {k:d[k].copy() for k in ('keys','residual','pooled','owners','mass')}

def main():
    mat=material();gamma,beta,pw,pb,w,b=mat;anchors=batch('keys')
    target=np.array([r[3] for r in fixture_rows(ANCHORS)+fixture_rows(CHECKS)])
    assert np.array_equal(anchors['owners'],np.arange(len(target)))
    k=anchors['keys'].astype(np.float64);mu=k.mean(0);kn=k-mu
    r=anchors['residual'];z,pooled,unit,scale=compute(r,mat)
    parity=float(np.max(abs(pooled-anchors['pooled'])));assert parity<.001
    without=z.copy();without[np.arange(len(target)),target]=-np.inf;other=without.argmax(1)
    gap=z[np.arange(len(target)),target]-z[np.arange(len(target)),other]
    centroids=np.stack([r[target==i].mean(0) for i in range(7)])
    contrast=(w[target]-w[other])*(1-pooled**2)
    reverse=(contrast@pw)*gamma
    sensitivities=(reverse-reverse.mean(1,keepdims=True)-unit*(reverse*unit).mean(1,keepdims=True))/scale
    sq=(sensitivities**2).sum(1,keepdims=True)
    deltas={}
    for method,margin in itertools.product(('centroid','linearized_margin'),(1.,3.)):
        needed=np.maximum(margin-gap,0)[:,None]
        if method=='centroid':v=(centroids[target]-centroids[other])*np.minimum(needed/margin,1.)
        else:v=needed*sensitivities/np.maximum(sq,1e-8)
        limit=.35*np.linalg.norm(r,axis=1,keepdims=True)
        v=v*np.minimum(1.,limit/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-8))
        deltas[method,margin]=v
    gram=kn@kn.T
    spec={};stats=[];best=None
    select=batch('select')
    sr,sw,_=official_sample('select',160);sy=np.array([r[3] for r in sr])
    baseline=reduce_windows(compute(select['residual'],mat)[0],select['owners'],select['mass'])
    control=metric(sy,baseline.argmax(1),sw)
    sk=(select['keys'].astype(float)-mu)@kn.T
    for (method,margin),vals in deltas.items():
        for damping in (.1,.5,1.):
            matrix=gram.copy();matrix.flat[::len(gram)+1]+=damping*np.trace(gram)/len(gram)
            dual=solve(matrix,vals,assume_a='pos');induced=(sk@dual).astype('f4')
            for fraction in (.25,.5,1.):
                zz=compute(select['residual']+fraction*induced,mat)[0]
                pred=reduce_windows(zz,select['owners'],select['mass']).argmax(1);m=metric(sy,pred,sw)
                row=(method,margin,damping,fraction,m['accuracy'],m['macro_f1'])
                stats.append(row)
                if m['macro_f1']+1e-12>=control['macro_f1'] and (best is None or (m['accuracy'],m['macro_f1'])>best[:2]):
                    best=(m['accuracy'],m['macro_f1'],method,margin,damping,fraction,dual.copy())
    if best is None:
        row=max(stats,key=lambda s:(s[5],s[4]));method,margin,damping,fraction=row[:4]
        matrix=gram.copy();matrix.flat[::len(gram)+1]+=damping*np.trace(gram)/len(gram)
        dual=solve(matrix,deltas[method,margin],assume_a='pos')
    else:_,_,method,margin,damping,fraction,dual=best
    delta=(fraction*dual.T@kn).astype('f4');bias_delta=(-delta.astype(float)@mu).astype('f4')
    oldw=tensor(PARENT,LAYER+'dense.weight');oldb=tensor(PARENT,LAYER+'dense.bias')
    nw=(oldw+delta).astype('f4');nb=(oldb+bias_delta).astype('f4')
    dest=TARGET/'model.safetensors';changes=patched_checkpoint(PARENT,dest,{LAYER+'dense.weight':nw,LAYER+'dense.bias':nb})
    write_config(TARGET)
    h=file_hash(dest);control_sha=file_hash(PARENT)
    change={'method':method,'desired_margin':margin,'regularization':damping,'fraction':fraction,'parent_sha256':control_sha,'target_sha256':h,
            'programming_keys':len(target),'rank_bound':len(target)-1,'candidate_count':len(stats),'tensors':changes,'gradient_steps':0,'uses_teacher_outputs':False}
    np.savez_compressed(TARGET/'write.npz',**{LAYER+'dense.weight':nw,LAYER+'dense.bias':nb})
    save_json(TARGET/'edit.json',change)
    out={'selection':change,'candidates':stats,'groups':{},'max_parent_pooled_replay_error':parity,
         'method':'Write an additive, centered low-rank key–value residual into the final feed-forward projection of the frozen BERT encoder. Values use authored role prototypes or an analytic pullback of a prescribed class margin through the fixed pooler and layer normalization. The same 84 fictional probes supply the keys. Validation chooses the edit, independently of the Gemini choices. The classifier head remains the previously frozen associative head.'}
    def forward(d):
        rewritten=d['residual']+d['keys']@(nw-oldw).T+(nb-oldb)
        before,pa,_,_=compute(d['residual'],mat);after=compute(rewritten,mat)[0]
        err=float(np.max(abs(pa-d['pooled'])));assert err<.001
        return reduce_windows(before,d['owners'],d['mass']),reduce_windows(after,d['owners'],d['mass']),err
    for fold in ('select','confirm'):
        d=batch(fold);rows,sw,info=official_sample(fold,160);y=np.array([r[3] for r in rows])
        p,q,err=forward(d);out['groups'][fold]={'sampling':info,'parent':metric(y,p.argmax(1),sw),'programmed':metric(y,q.argmax(1),sw),'max_pooled_replay_error':err}
    save_json(ROOT/'results/internal-mlp-validation.json',out)
    print('Physical MLP write',method,margin,damping,fraction,out['groups']['confirm']['programmed']['accuracy'],flush=True)
    for group in FOLDERS:
        d=batch(group);folder=folder_for(group);rows=json.loads((folder/'approved.json').read_text());p,q,err=forward(d)
        save_json(folder/'mlp.json',{'internal_mlp':{'sha256':h,'predictions':{r['id']:LABELS[i] for r,i in zip(rows,q.argmax(1))}}})
if __name__=='__main__':main()
