"""Portable posterior-utility scoring and structured decoding for supplied sentences."""
import numpy as np

from hier_model import CRF
from role_edge_runtime import decode
from sentence_uncertainty_runtime import apply_uncertainty, probability, uncertainty_evidence


def neighbors(source,case_starts):
    empty=np.zeros_like(source)
    previous=empty.copy()
    following=empty.copy()
    same=~case_starts[1:]
    previous[1:][same]=source[:-1][same]
    following[:-1][same]=source[1:][same]
    return previous,following


def input_arrays(structure,base_p,latents,starts):
    prev,follow=neighbors(base_p,starts)
    structural=np.column_stack((structure,prev,follow))
    h=np.column_stack((structural,latents))
    return structural,h


def distribution(x,old,weights):
    h=(x-weights["center"])*weights["scale"]
    a,b=weights["weight"],weights["bias"]
    if len(a)==1:
        scores=h@a[0]+b[0]
    else:
        scores=np.empty((len(x),7),np.float32)
        for k in range(7):
            rows=old==k
            scores[rows]=h[rows]@a[k]+b[k]
    z=np.maximum(scores,0)
    return z/np.maximum(z.sum(1,keepdims=True),1.e-9)


def utility(marginal,read,current,macro,prior,retention):
    initial=.76*marginal+.24*np.eye(7)[current]
    prob=initial if read is None else .48*initial+.52*read
    denom=prior["denominator"]
    direction=(prob-.5*prior["ref_f1"])*(np.mean(denom)/np.maximum(denom,.04))
    values=(1-macro)*prob+macro*direction
    values[np.arange(len(current)),current]+=retention
    return values


def joint_path(current,rank,values,prior,strength,rate,template):
    matrix=np.zeros_like(template)
    matrix[3:,3:]=strength*prior["transition"]
    if rate<1.:
        allowed=(rank[:,None]<rate)|(np.arange(7)[None,:]==current[:,None])
        values=np.where(allowed,values,-1.e6)
    edges=np.zeros((len(current)-1,7,7),dtype=np.float32)
    return decode(values,matrix,edges,0.)


def joint_inputs(logits,edge,parent,gain,selected,lengths,states):
    base_rate,base_mix,base_margin=parent["uncertainty.settings"]
    base={k:parent["uncertainty."+k] for k in ("center","scale","weight","bias")}
    order,prior,structure=uncertainty_evidence(logits,edge,parent,gain,selected,lengths,1.)
    estimate=probability(structure[order],base)
    current=apply_uncertainty(selected,order,prior,estimate,base_rate,base_mix,base_margin).astype(np.int8)
    marginal=.45*structure[:,:7]+.55*structure[:,7:14]
    odds=1.-marginal[np.arange(len(marginal)),current]
    order=np.argsort(-odds,kind="stable")
    rank=np.empty(len(order))
    rank[order]=np.arange(len(order))/len(order)
    w={k.removeprefix("joint."):v for k,v in parent.items() if k.startswith("joint.")}
    mode=int(w["mode"])
    start=np.arange(len(current))==0
    structural,context=input_arrays(structure,marginal,states,start)
    head=None if mode==0 else distribution(context if mode==2 else structural,current,w)
    macro,retention,rate,strength=w["settings"]
    values=utility(marginal,head,current,macro,w,retention)
    return current,rank,values,w


def joint_labels(logits,edge,parent,gain,selected,lengths,states):
    current,rank,values,w=joint_inputs(logits,edge,parent,gain,selected,lengths,states)
    _,_,rate,strength=w["settings"]
    return joint_path(current,rank,values,w,strength,rate,parent[CRF])
