"""Route low-confidence sentence labels using frozen structured posteriors."""
import numpy as np

from hier_model import CRF
from span_posterior_decode import posterior


def uncertainty_evidence(logits, edge, parent, gain, current, lengths, rate):
    cal=logits@parent["emission.weight"]+parent["emission.bias"]
    p=posterior(logits,parent[CRF],gain*edge,1.25)[-1]
    q=posterior(cal,parent[CRF],gain*edge,1.25)[-1]
    mix=.45*p+.55*q
    i=np.arange(len(logits))
    risk=1.-mix[i,current]
    query=np.argsort(-risk,kind="stable")[:max(1,int(np.ceil(rate*len(logits))))]
    hot=np.eye(7)[current]
    left=np.vstack((np.zeros((1,7)),hot[:-1]))
    right=np.vstack((hot[1:],np.zeros((1,7))))
    pos=(i+.5)/len(i)
    structural=np.column_stack((p,q,hot,logits-logits.mean(1,keepdims=True),
                                left,right,pos,np.log1p(lengths))).astype("f4")
    return query,.65*mix+.35*hot,structural


def probability(features,weights):
    value=((features-weights["center"])*weights["scale"])@weights["weight"]+weights["bias"]
    value=np.maximum(value,0.)
    return value/np.maximum(value.sum(1,keepdims=True),1.e-9)


def apply_uncertainty(current,query,prior,estimated,rate,mix,margin):
    n=min(len(query),max(1,int(np.ceil(rate*len(current)))))
    ii=query[:n]
    utility=(1-mix)*prior[ii]+mix*estimated[:n]
    proposed=utility.argmax(1)
    gap=utility[np.arange(n),proposed]-utility[np.arange(n),current[ii]]
    accepted=gap>=margin
    result=current.copy()
    result[ii[accepted]]=proposed[accepted]
    return result


def uncertainty_labels(logits,edge,parent,gain,current,lengths):
    rate,mix,margin=parent["uncertainty.settings"]
    weights={k:parent["uncertainty."+k] for k in ("center","scale","weight","bias")}
    query,prior,structure=uncertainty_evidence(logits,edge,parent,gain,current,lengths,rate)
    return apply_uncertainty(current,query,prior,probability(structure[query],weights),rate,mix,margin)
