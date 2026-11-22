"""Paired whole-judgment bootstrap for the pooled fixed-seven-class macro-F1."""
import numpy as np

def f1(c):
    d=np.diagonal(c,axis1=-2,axis2=-1);den=c.sum(-1)+c.sum(-2)
    return np.divide(2*d,den,out=np.zeros_like(d,dtype=float),where=den>0).mean(-1)

def interval(cases,y,old,new):
    u=np.unique(cases); c0=[];c1=[]
    for v in u:
        ix=cases==v
        pair=[]
        for p in [old,new]:
            m=np.zeros((7,7),np.int32);np.add.at(m,(y[ix],p[ix]),1);pair.append(m)
        c0.append(pair[0]);c1.append(pair[1])
    a,b=map(np.stack,(c0,c1));rep=np.random.default_rng(882211).integers(len(u),size=(3000,len(u)))
    d=f1(b[rep].sum(1))-f1(a[rep].sum(1))
    return {"unit":"judgment","seed":882211,"case_count":len(u),"resamples":len(rep),"point":float(f1(b.sum(0))-f1(a.sum(0))),
            "percentile95":np.quantile(d,[.025,.975]).tolist()}
