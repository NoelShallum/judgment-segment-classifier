"""Pure inference functions used by the benchmark and the portable CPU model."""
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

VEC=HashingVectorizer(n_features=1024,alternate_sign=False,norm=None,
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b",dtype=np.float32)


def features(texts):
    # Everything in x is observable when the user submits a judgment.
    part=[(" ".join(texts[int(len(texts)*j/3):int(len(texts)*(j+1)/3)])) for j in range(3)]
    c=VEC.transform(part).toarray()
    count=c.sum(0)
    support=(count>0).astype(np.float32)
    blocks=[np.log1p(count)]
    # Within-case vocabulary contrast captures quotations and changes of register.
    for j in (0,2):
        freq=(c[j]+.5)/np.maximum(c[j].sum(),1)
        global_freq=(count+1.5)/np.maximum(count.sum(),1)
        contrast=np.clip(np.log(freq/global_freq),-2.5,2.5)*support
        blocks.append(contrast.reshape(256,4).mean(1))
    lens=np.asarray([len(s) for s in texts])
    scalars=[np.log1p(len(lens)),np.log1p(lens.mean()),np.log1p(lens.std()),np.mean(lens<40),
             np.mean([s.isupper() for s in texts]),np.mean(['"' in s or "“" in s for s in texts]),
             np.mean([s[0].isdigit() for s in texts]),np.mean([s.endswith("?") for s in texts])]
    return np.concatenate([*blocks,scalars]).astype(np.float32)


def targets(texts,w):
    x=np.clip((features(texts)-w["center"])*w["scale"],-5,5)
    def sm(logprior):
        v=np.exp(logprior-logprior.max())
        return v/v.sum()
    return {"targeted":sm(x@w["role_prior.weight"]+w["offset"]),
            "shuffled":sm(x@w["shuffled.weight"]+w["offset"]),"constant":w["global_fraction"]}


def adapted_bias(r,arm,strength,donor):
    n=len(r["prediction"])
    base=np.bincount(r["prediction"],minlength=7)
    src=(base+10/7)/(n+10)
    gain=np.clip(np.log(r[arm]/src),-1.75,1.75)
    gain-=gain.mean()
    bias=donor.bias.copy()
    bias[3:]+=strength*gain
    return bias


