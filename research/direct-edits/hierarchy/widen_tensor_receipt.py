from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from embedding_write import BASE,HERE,digest,dump,file_hash

def ptr(k,a):
    if k=="sent_encoder.emb.weight":
        return np.ix_(np.arange(a.shape[0]),np.arange(100))
    if k=="emitter.hidden2tag.weight":
        return np.ix_(np.arange(10),np.r_[np.arange(100),np.arange(114,214)])
    if k in ("emitter.hidden2tag.bias","crf.transitions"):
        return tuple(slice(None) for _ in a.shape)
    rows=np.concatenate([np.arange(100)+g*114 for g in range(4)])
    if a.ndim==1:return rows
    col=np.arange(100) if a.shape[1]==100 else np.r_[np.arange(100),np.arange(114,214)]
    return np.ix_(rows,col)

def run():
    src=load_file(str(BASE));path=HERE/"widened-model/hier.safetensors";dest=load_file(str(path));receipt={}
    for k,old in src.items():
        now=dest[k];orig=np.zeros_like(now);ix=ptr(k,old);orig[ix]=old
        delta=now-orig
        preexisting=now[ix]!=old
        all_edits=int(np.count_nonzero(delta));mapped=int(np.count_nonzero(preexisting))
        receipt[k]=dict(old_shape=list(old.shape),new_shape=list(now.shape),added_slots=int(now.size-old.size),changed_preexisting_values=mapped,
                       changed_or_added_nonzero=all_edits,added_nonzero=all_edits-mapped,delta_L2_from_zero_padded_reference=float(np.linalg.norm(delta.ravel().astype(float))),
                       original_sha256=digest(old),written_sha256=digest(now))
    dump(HERE/"widened-tensors.json",dict(original_sha256=file_hash(BASE),written_sha256=file_hash(path),original_parameters=sum(v.size for v in src.values()),
          new_parameters=sum(v.size for v in dest.values()),gradient_steps=0,coordinate_mapping="Old gates 0..3 use rows 114*g through 114*g+99. Old forward features use columns 0..99; old reverse features use 114..213. The fourteen additional positions in each direction carry two seven-role dictionaries. CRF is verbatim.",tensors=receipt))
if __name__=="__main__":run()
