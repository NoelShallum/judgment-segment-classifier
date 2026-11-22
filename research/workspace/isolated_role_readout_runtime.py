"""Predict each supplied sentence independently with the frozen isolated readout."""
import re

import numpy as np

from hier_model import LABELS
from role_cues import cue_features

TOKENS=re.compile(r"(?u)\b\w+\b")
MASK=(1<<32)-1


def hash_bin(text,n):
    raw=text.encode("utf-8")
    value=0

    def rotate(x,s):
        return ((x<<s)|(x>>(32-s)))&MASK

    def mix(x):
        return rotate(x*0xcc9e2d51&MASK,15)*0x1b873593&MASK

    end=(len(raw)//4)*4
    for i in range(0,end,4):
        value^=mix(int.from_bytes(raw[i:i+4],"little"))
        value=(rotate(value,13)*5+0xe6546b64)&MASK
    if end<len(raw):
        value^=mix(int.from_bytes(raw[end:],"little"))
    value^=len(raw)
    value^=value>>16
    value=value*0x85ebca6b&MASK
    value^=value>>13
    value=value*0xc2b2ae35&MASK
    value^=value>>16
    return abs(value if value<1<<31 else value-(1<<32))%n


def lexical_score(text,table):
    tokens=TOKENS.findall(text.lower())
    count=len(table)
    indices={hash_bin(w,count) for w in tokens}
    indices.update(hash_bin(a+" "+b,count) for a,b in zip(tokens,tokens[1:]))
    if not indices:
        return np.zeros(7)
    return table[sorted(indices)].sum(0)/np.sqrt(len(indices))


def label_ids(texts,states,weights):
    lex=np.stack([lexical_score(t,weights["isolated.lexical"]) for t in texts])
    surface=np.column_stack((np.log1p([len(t.split()) for t in texts]),
                             np.log1p([len(t) for t in texts]),cue_features(texts)))
    d=np.column_stack((states,lex,surface))
    y=((d-weights["isolated.center"])*weights["isolated.scale"])@weights["isolated.matrix"]+weights["isolated.bias"]
    return (y+weights["isolated.prior_shift"]).argmax(1)


def infer(sentences,model,weights):
    if not isinstance(sentences,list) or not sentences or any(not isinstance(s,str) or not s.strip() for s in sentences):
        raise ValueError("Supply a nonempty JSON array of nonblank sentence strings")
    texts=[s.strip() for s in sentences]
    states=model.contextualize(model.words(texts),independent=True)
    return {"labels":[LABELS[i] for i in label_ids(texts,states,weights)]}
