"""Feature extraction for long-range role context within one judgment."""
import numpy as np

K=8


def scaled_mean(q,indices,value):
    finite=np.isfinite(value)
    maximum=np.max(np.where(finite,value,-1.e6),axis=1,keepdims=True)
    weights=np.where(finite,np.exp(8*(np.where(finite,value,maximum)-maximum)),0)
    mass=weights.sum(1)
    x=(q[indices]*weights[:,:,None]).sum(1)
    return np.divide(x,mass[:,None],out=np.zeros_like(x),where=mass[:,None]!=0),mass>0


def remote_indices(h):
    n = len(h)
    z=h-h.mean(0)
    z=np.ascontiguousarray(z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1.e-7),dtype="f4")
    k=min(K,n)
    ids=np.empty((n,k),dtype=np.int32)
    weight=np.empty((n,k),dtype=np.float32)
    positions=np.arange(n)[None,:]
    for start in range(0,n,192):
        end=min(n,start+192)
        i=np.arange(start,end)
        sim=z[start:end]@z.T
        sim[np.abs(i[:,None]-positions)<5]=-np.inf
        j=np.argpartition(sim,-k,axis=1)[:,-k:]
        ids[start:end]=j
        weight[start:end]=np.take_along_axis(sim,j,axis=1)
    return ids,weight


def graph_context(structure,current,local):
    n=len(current)
    q=structure[:,:7]
    labels=np.eye(7,dtype="f4")[current]
    states=structure[:,33:]
    j,w=remote_indices(states)
    semantic,support=scaled_mean(q,j,w)
    votes,_=scaled_mean(labels,j,w)
    walk,_=scaled_mean(semantic,j,w)
    positions=(np.arange(n)+.5)/n
    phase=np.clip((positions*5).astype(int),0,4)
    cumulative=np.cumsum(q,axis=0,dtype=np.float64)
    earlier=(cumulative-q)/np.maximum(np.arange(n),1)[:,None]
    later=(cumulative[-1]-cumulative)/np.maximum(np.arange(n)[::-1],1)[:,None]
    segment=np.empty_like(q)
    for k in range(5):
        here=phase==k
        if np.any(here):
            mass=q[here].sum(0)
            segment[here]=(mass-q[here])/max(int(here.sum())-1,1)
    global_view=np.column_stack((local,semantic,votes,semantic*q,support))
    graph=np.column_stack((global_view,walk,walk*q,earlier,later,segment))
    return global_view.astype("f4"),graph.astype("f4")

