"""Apply the frozen local-context readout to a judgment's supplied sentences."""
import numpy as np

from hier_model import CRF
from sentence_joint_runtime import joint_inputs,joint_path,neighbors


def normalized_positive(x):
    p=np.maximum(x,0.)
    return p/np.maximum(p.sum(1,keepdims=True),1.e-8)


def calibrated_labels(current,structure,prior,posterior,parent):
    one=np.eye(7)[current]
    x=np.column_stack((structure[:,:33],one,posterior,prior*posterior))
    z=(x-parent["document_calibration.center"])*parent["document_calibration.scale"]
    result=normalized_positive(z@parent["document_calibration.matrix"]+parent["document_calibration.bias"])
    blend,threshold,protect=parent["document_calibration.settings"]
    anchor=.55*posterior+.15*prior+.30*one
    response=(1-blend)*anchor+blend*result
    best=response.argmax(1)
    row=np.arange(len(current))
    accept=response[row,best]-response[row,current]>=threshold
    if protect:
        accept &= current!=5
    return np.where(accept,best,current)


def labels(logits,edge,parent,gain,selected,lengths,states):
    anchor,rank,utility,w=joint_inputs(logits,edge,parent,gain,selected,lengths,states)
    _,_,eligible_rate,strength=w["settings"]
    current=joint_path(anchor,rank,utility,w,strength,eligible_rate,parent[CRF])
    n=len(current)
    position=np.arange(n)
    p=np.exp(4*(utility-utility.max(1,keepdims=True)))
    p/=p.sum(1,keepdims=True)
    one=np.eye(7)[current]
    first=position==0
    preceding,following=neighbors(one,first)
    top=np.sort(utility,axis=1)
    margin=top[:,-1]-top[:,-2]
    structure=np.column_stack((p,one,preceding,following,rank,margin,utility.argmax(1)==current,
                               (position+.5)/n,np.log1p(lengths),states)).astype("f4")
    p32=structure[:,:7]
    left,right=neighbors(p32,first)
    data=np.column_stack((structure,left,right,p32*left,p32*right))
    if "local_readout.two_hop" in parent:
        from sentence_graph_runtime import graph_context
        _,data=graph_context(structure,current,data)
    budget,mix,threshold=parent["local_readout.settings"]
    valid=np.flatnonzero(rank<eligible_rate)
    valid=valid[np.argsort(margin[valid],kind="stable")]
    ordered=valid[np.argsort(p32[valid,current[valid]],kind="stable")]
    q=ordered[:max(1,int(np.ceil(budget*n)))] if budget<1 else position
    calibration="document_calibration.matrix" in parent
    rows=data if calibration else data[q]
    y=((rows-parent["local_readout.center"])*parent["local_readout.scale"])@parent["local_readout.matrix"]
    estimate=normalized_positive(y+parent["local_readout.bias"])
    if calibration:
        prior=.6*p+.4*one
        posterior=(1-mix)*prior+mix*estimate
        value=posterior[q]
    else:
        value=(1-mix)*(.6*p[q]+.4*one[q])+mix*estimate
    best=value.argmax(1)
    row=np.arange(len(q))
    accept=(value[row,best]-value[row,current[q]]>=threshold)&(current[q]!=5)
    current[q[accept]]=best[accept]
    if calibration:
        return calibrated_labels(current,structure,prior,posterior,parent)
    return current
