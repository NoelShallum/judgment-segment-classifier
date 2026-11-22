"""Pool contiguous bidirectional regions with context on both sides."""
import numpy as np

def contextual(model,x,core):
    halo=core//2
    out=[]
    for start in range(0,len(x),core):
        end=min(start+core,len(x))
        left=max(0,start-halo);right=min(len(x),end+halo)
        out.append(model.contextualize(x[left:right])[start-left:end-left])
    return np.concatenate(out)

