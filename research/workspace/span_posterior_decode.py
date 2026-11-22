"""Exact maximal-run marginals and a span-posterior decoder over proposed paths."""
import numpy as np

K=7
MASK=np.eye(K,dtype=bool)


def posterior(logit, transition, edge, temperature, boundary_probabilities=False):
    emission=np.asarray(logit,dtype=np.float64)/temperature
    trans=(transition[None,3:,3:]+edge)/temperature
    begin=transition[1,3:]/temperature
    end=transition[3:,2]/temperature
    n=len(emission)
    enter=np.empty_like(emission)
    exit_=np.empty_like(emission)
    alpha=np.empty_like(emission)
    beta=np.empty_like(emission)
    enter[0]=begin
    alpha[0]=begin+emission[0]
    for i in range(1,n):
        step=alpha[i-1,:,None]+trans[i-1]
        alpha[i]=emission[i]+np.logaddexp.reduce(step,axis=0)
        enter[i]=np.logaddexp.reduce(np.where(MASK,-np.inf,step),axis=0)
    exit_[-1]=end
    beta[-1]=end
    for i in range(n-2,-1,-1):
        step=trans[i]+(emission[i+1]+beta[i+1])[None,:]
        beta[i]=np.logaddexp.reduce(step,axis=1)
        exit_[i]=np.logaddexp.reduce(np.where(MASK,-np.inf,step),axis=1)
    z=np.logaddexp.reduce(alpha[-1]+end)
    stay=trans.diagonal(axis1=1,axis2=2)
    body=np.concatenate((np.zeros((1,K)),np.cumsum(emission,axis=0)))
    repeats=np.concatenate((np.zeros((1,K)),np.cumsum(stay,axis=0)))
    result=enter,exit_,body,repeats,z,np.exp(alpha+beta-z)
    if boundary_probabilities:
        start=np.exp(np.minimum(enter+emission+beta-z,0))
        end=np.exp(np.minimum(alpha+exit_-z,0))
        return result,start,end
    return result


def spans(labels):
    ends=np.r_[np.flatnonzero(labels[1:]!=labels[:-1])+1,len(labels)]
    starts=np.r_[0,ends[:-1]]
    return np.column_stack((starts,ends,labels[starts]))


def evaluate_spans(probability, candidates, counts):
    enter,exit_,body,repeats,z,marginals=probability
    a,b,k=candidates.T
    score=enter[a,k]+exit_[b-1,k]+body[b,k]-body[a,k]+repeats[b-1,k]-repeats[a,k]
    run_probability=np.exp(np.minimum(score-z,0))
    avg=max(1,counts.mean())
    mass=np.concatenate((np.zeros((1,K)),np.cumsum(marginals*counts[:,None]/avg,axis=0)))
    expected_tokens=mass[b,k]-mass[a,k]
    return run_probability,expected_tokens


def best_path(candidates, values, n):
    # Intervals must tile the complete text and adjacent maximal runs must differ.
    end_scores=np.full((n+1,K),-np.inf)
    back=np.full((n+1,K,2),-1,dtype=int)
    for (a,b,k),value in zip(candidates,values):
        if a:
            before=end_scores[a].copy()
            before[k]=-np.inf
            prev=before.argmax()
            score=before[prev]+value
        else:
            prev=-1
            score=value
        if score>end_scores[b,k]:
            end_scores[b,k]=score
            back[b,k]=[a,prev]
    k=int(end_scores[n].argmax())
    path=np.empty(n,dtype=np.int8)
    while n:
        start,prev=back[n,k]
        if start<0:
            raise ValueError("Candidate runs do not cover document")
        path[start:n]=k
        n,k=start,prev
    return path
