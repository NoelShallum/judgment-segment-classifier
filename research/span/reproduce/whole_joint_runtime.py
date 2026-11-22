"""Construct and score the raw-judgment lattice with frozen joint proposals."""
import numpy as np

from hier_model import CRF
from segment_span_runtime import learned_scores,span_features
from sentence_joint_runtime import joint_inputs,joint_path
from span_posterior_decode import best_path,spans


def candidates(logits,edge,ext,parents,incumbent,joint,counts,lengths,states,selected):
    frozen=logits["joint"]
    gain=float(joint["edge.gain"])
    current,rank,value,w=joint_inputs(frozen,ext,joint,gain,selected,lengths,states)
    _,_,rate,strength=w["settings"]
    role_path=joint_path(current,rank,value,w,strength,rate,joint[CRF])
    plain=joint_path(current,rank,value,w,0.,rate,joint[CRF])
    proposals=np.concatenate((spans(role_path),spans(plain)))
    arcs,support,x,p,_,novel=span_features(
        logits,edge,parents,counts,incumbent["proposals"],float(incumbent["temperature"]),extra_runs=proposals)
    base=p+float(incumbent["share"])*(learned_scores(x,incumbent)-p)
    a,b,k=arcs.T
    pcol=np.r_[np.zeros((1,7)),np.cumsum(value,axis=0)]
    q=np.arange(len(current))
    v=value-value[q,current][:,None]
    mass=np.r_[np.zeros((1,7)),np.cumsum(v,axis=0)]
    votes={}
    for seq in (role_path,plain):
        for z in map(tuple,spans(seq)):
            votes[z]=votes.get(z,0)+1
    supporter=np.array([votes.get(tuple(z),0) for z in arcs])
    cues=np.column_stack((novel,supporter,np.log1p(b-a),
                          (pcol[b,k]-pcol[a,k])/(b-a),(mass[b,k]-mass[a,k])/(b-a),
                          value[a,k],value[b-1,k],v[a,k],v[b-1,k],
                          role_path[a]==k,role_path[b-1]==k,current[a]==k,current[b-1]==k,base))
    return arcs,base,novel,np.column_stack((x,cues)),role_path


def adjustment(features,weights):
    return np.clip(((features-weights["center"])*weights["scale"])@weights["coef"]+weights["bias"],-.7,.7)


def select_labels(arcs,original_score,novel,size,decode,residual,strength,cost,fee,use_extra):
    use=slice(None) if use_extra else ~novel
    total=original_score[use]+strength*residual[use]-cost-fee*novel[use]
    return decode(arcs[use],total,size)


def transfer_labels(logits,edge,ext,parents,incumbent,joint,counts,lengths,states,selected,weights):
    arcs,base,novel,features,role_path=candidates(
        logits,edge,ext,parents,incumbent,joint,counts,lengths,states,selected)
    strength,cost,fee,wide=weights["settings"]
    return select_labels(arcs,base,novel,len(role_path),best_path,adjustment(features,weights),
                         strength,cost,fee,bool(wide))
