"""Matrix-free ridge normal equations for jointly writing three native GLiNER matrices."""
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

def joint(cases, regs, shuffle=False):
    ys=np.concatenate([a['y'] for a in cases.values()]);ys=np.random.default_rng(901112).permutation(ys) if shuffle else ys
    freq=np.bincount(ys,minlength=7); feats=[]; coeff_queries=[]; dirs=[]
    for ch in range(3):
        query_avg=np.concatenate([a[f'q{ch}'] for a in cases.values()]).mean(0).astype('f8');V=np.linalg.svd(query_avg-query_avg.mean(0),full_matrices=False)[2][:6];dirs.append(V)
        X=np.concatenate([a[f'x{ch}'] for a in cases.values()]).astype('f8');feats.append(X)
        A=[]
        for a in cases.values():
            q=a[f'q{ch}'].astype('f8');q-=q.mean(1,keepdims=True);q=q@V.T/np.sqrt(q.shape[-1])/3; A.append(q[a['qb']]);
        coeff_queries.append(np.concatenate(A))
    targets=[];weights=[];pt=0
    for a in cases.values():
        idx=a['obs'];p=ys[pt:pt+len(a['y'])][idx];pt+=len(a['y']);w=a['weights']/np.bincount(idx,weights=a['weights'])[idx]/np.sqrt(np.maximum(freq[p],1));weights.append(w)
        native=a['native'].mean(1).astype('f8');targets.append((np.eye(7)[p]-1/7)*6-(native-native.mean(1,keepdims=True)))
    w=np.concatenate(weights);Y=np.concatenate(targets);shapes=[(6,x.shape[1]) for x in feats];offsets=np.cumsum([0]+[a*b for a,b in shapes]);size=offsets[-1]
    def split(v):return [v[offsets[i]:offsets[i+1]].reshape(shape) for i,shape in enumerate(shapes)]
    def adj(z):
        zz=z*w[:,None];return np.concatenate([(np.einsum('bck,bc->bk',A,zz).T@X).ravel() for A,X in zip(coeff_queries,feats)])
    def direct(v):return sum(np.einsum('bck,bk->bc',A,X@t.T) for t,A,X in zip(split(v),coeff_queries,feats))
    diag=np.concatenate([((np.einsum('bck,bck->bk',A,A)*w[:,None]).T@(X*X)).ravel() for A,X in zip(coeff_queries,feats)])
    scales=np.concatenate([np.repeat(diag[a:b].mean(),b-a) for a,b in zip(offsets[:-1],offsets[1:])]);rhs=adj(Y);out={};diaginfo={}
    for reg in regs:
        pen=reg*scales+1e-10
        def mat(v):return adj(direct(v))+pen*v
        op=LinearOperator((size,size),matvec=mat,dtype='d');pre=LinearOperator((size,size),matvec=lambda v:v/(diag+pen),dtype='d');iterations=[]
        sol,status=cg(op,rhs,M=pre,rtol=5e-4,atol=0,maxiter=400,callback=lambda v:iterations.append(1)); diaginfo[str(reg)]={'iterations':len(iterations),'status':int(status),'relative_normal_equation_residual':float(np.linalg.norm(mat(sol)-rhs)/np.linalg.norm(rhs))}; out[reg]=[np.asarray(V.T@z,'f4') for V,z in zip(dirs,split(sol))]
    return out,diaginfo
