"""Decode legal roles with a separately written context-dependent transition bias."""
import numpy as np
from hier_model import path_labels

CHANGE=(~np.eye(7,dtype=bool)).astype(np.float32)

def boundary_features(logits):
    values=logits-logits.mean(1,keepdims=True)
    score1,score2=values[:-1],values[1:]
    delta=score2-score1
    p=np.sort(score1,axis=1);q=np.sort(score2,axis=1)
    m1=p[:,-1]-p[:,-2];m2=q[:,-1]-q[:,-2]
    distance=np.linalg.norm(delta,axis=1)
    changed=(score1.argmax(1)!=score2.argmax(1)).astype(float)
    return np.column_stack((score1,score2,np.abs(delta),m1,m2,np.abs(m2-m1),distance,changed))


def variable_decode(logits,trans,residual):
    a=np.asarray(logits,dtype=np.float32)
    if len(a)<2 or not np.any(residual):
        return path_labels(a,trans)
    track=np.empty((len(a),7),np.int8)
    score=a[0]+trans[1,3:]
    rows=np.arange(7)
    interior=trans[3:,3:]
    for i,u in enumerate(residual,1):
        layer=score[:,None]+interior+u*CHANGE
        best=layer.argmax(0)
        track[i]=best
        score=layer[best,rows]+a[i]
    k=np.argmax(score+trans[3:,2])
    decoded=np.empty(len(a),np.int8)
    for i in range(len(a)-1,0,-1):
        decoded[i]=k;k=track[i,k]
    decoded[0]=k
    return decoded


