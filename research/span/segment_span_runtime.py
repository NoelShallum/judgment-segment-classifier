"""Score and decode seven-role spans over a restricted set of candidate endpoints."""
import numpy as np

from hier_model import CRF
from role_edge_runtime import decode
from span_posterior_decode import best_path, evaluate_spans, posterior, spans

ROLES = 7
SHORT_SETTINGS = ((4,2),(8,2),(16,2),(16,3))


def short_arcs(length, probabilities, width, top):
    """Keep the highest posterior roles for each short start/end pair."""
    enter,leave,body,repeats,_,_=probabilities
    sizes=np.arange(1,width+1)
    a,b=np.nonzero(np.arange(length)[:,None]+sizes<=length)
    b=a+sizes[b]
    score=enter[a]+leave[b-1]+body[b]-body[a]+repeats[b-1]-repeats[a]
    k=np.argsort(-score,axis=1,kind="stable")[:,:top]
    return np.column_stack((np.repeat(a,top),np.repeat(b,top),k.ravel())).astype(np.int32)


def widen(base, inferred, lattice, units, tier):
    chunks=[np.asarray(sorted(base),dtype=np.int32)]
    tiers=[np.zeros(len(chunks[0]),np.int8)]
    if tier>=1:
        chunks.append(np.asarray(inferred,dtype=np.int32).reshape(-1,3))
        tiers.append(np.ones(len(chunks[-1]),np.int8))
    if tier>=2:
        w,k=SHORT_SETTINGS[tier-2]
        raw=short_arcs(units,lattice,w,k)
        step=raw[:,1]-raw[:,0]
        ranks=np.tile(np.arange(k),len(raw)//k)
        first=np.where(ranks>=2,5,np.where(step<=4,2,np.where(step<=8,3,4)))
        chunks.append(raw)
        tiers.append(first.astype(np.int8))
    packed,mapping=np.unique(np.concatenate(chunks),axis=0,return_inverse=True)
    source=np.full(len(packed),5,np.int8)
    np.minimum.at(source,mapping,np.concatenate(tiers))
    order=np.lexsort((packed[:,2],packed[:,0],packed[:,1]))
    return packed[order],source[order]


def posterior_proposals(n, starts, ends, state):
    """Propose runs with likely endpoints even if a Viterbi path omitted them."""
    _, leave, body, repeats, _, _ = state
    finish = leave + body[1:] + repeats
    options = []
    distances = ((1,4),(4,10),(10,26),(26,66),(66,193))
    for k in range(ROLES):
        possible = np.flatnonzero(starts[:,k] >= .12)
        for a in possible:
            for lo,hi in distances:
                l,r = a+lo-1,min(n,a+hi-1)
                if l>=r:
                    break
                parts=finish[l:r,k]+.35*np.log(np.maximum(ends[l:r,k],1e-14))
                b=l+int(parts.argmax())+1
                options.append((a,b,k))
    return options


def span_features(logits, edge, parents, counts, proposals, temperature, posterior_ends=False, max_tier=None,
                  extra_runs=None):
    emission = logits["exact"]
    n = len(emission)
    balanced = logits["balanced"]
    transition = parents["balanced"][CRF]
    difference = parents["exact"][CRF] - transition
    paths = [decode(balanced + float(s) * (emission - balanced),
                    transition + float(s) * difference, edge, float(g)) for s, g in proposals]
    base = decode(emission, parents["exact"][CRF], edge, .2)
    paths[list(map(tuple,proposals)).index((1., .2))] = base
    votes = np.unique(np.concatenate([spans(y) for y in paths], axis=0), axis=0, return_counts=True)
    originals = {tuple(map(int, row)): int(j) for row, j in zip(*votes)}
    boundaries = sorted({0, n} | {a for a,b,k in originals} | {b for a,b,k in originals})
    lookup = {x: i for i, x in enumerate(boundaries)}
    options = set(originals)
    for a,b,k in originals:
        i,j = lookup[a],lookup[b]
        options.update((cut,b,k) for cut in boundaries[max(0,i-1):i+2] if cut < b)
        options.update((a,cut,k) for cut in boundaries[max(0,j-1):j+2] if cut > a)
    if posterior_ends or max_tier is not None:
        model, start_probability, end_probability = posterior(
            emission, parents["exact"][CRF], .2*edge, temperature, boundary_probabilities=True)
        inferred = posterior_proposals(n,start_probability,end_probability,model)
    else:
        model = posterior(emission, parents["exact"][CRF], .2*edge, temperature)
    if max_tier is None:
        if posterior_ends:
            options.update(inferred)
        if extra_runs is not None:
            original=options.copy()
            options.update(map(tuple,extra_runs))
        packed = np.asarray(sorted(options, key=lambda v:(v[1],v[0],v[2])), dtype=np.int32)
    else:
        packed,source = widen(options,inferred,model,n,max_tier)
    a,b,k = packed.T
    support = np.array([originals.get(key,0) for key in map(tuple,packed)])
    p, _ = evaluate_spans(model, packed, counts)
    marginal = model[-1]
    centered = emission - np.logaddexp.reduce(emission,axis=1,keepdims=True)
    accum = lambda x: np.vstack((np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)))
    summed, mass = accum(centered), accum(marginal)
    steps = b-a
    total = np.r_[0,np.cumsum(counts)]
    length = n
    log_steps = np.log1p(steps)
    span_avg = (summed[b,k]-summed[a,k])/steps
    body_avg = (mass[b,k]-mass[a,k])/steps
    votes_at = np.zeros(n+1)
    votes_at[0] = votes_at[-1] = len(paths)
    for y in paths:
        votes_at[1:-1] += y[:-1] != y[1:]
    context = np.column_stack((p, np.sqrt(p), np.maximum(np.log(np.maximum(p,1e-12)),-14),
        support/len(paths), votes_at[a]/len(paths), votes_at[b]/len(paths),
        log_steps, np.log1p(total[b]-total[a]), a/length, b/length, a==0, b==length,
        span_avg.clip(-10,0), body_avg, centered[a,k].clip(-10,0), centered[b-1,k].clip(-10,0),
        marginal[a,k], marginal[b-1,k], marginal[np.maximum(0,a-1),k], marginal[np.minimum(n-1,b),k]))
    one_hot = np.eye(ROLES)[k]
    paired = (one_hot[:,:,None]*context[:,None,[0,3,6,8,12,13]]).reshape(len(k),-1)
    inputs = np.column_stack((context,one_hot,paired))
    if posterior_ends or max_tier is not None:
        inputs = np.column_stack((inputs,start_probability[a,k],end_probability[b-1,k]))
    if max_tier is not None:
        return packed, support, inputs, p, base, source
    if extra_runs is not None:
        added=np.array([tuple(arc) not in original for arc in packed])
        return packed, support, inputs, p, base, added
    return packed, support, inputs, p, base


def learned_scores(features, weights):
    return np.clip(((features-weights["center"])*weights["scale"]) @ weights["weights"]+weights["bias"],0,1)


def density_features(logits, edge, parents, token_counts, incumbent, tier):
    arcs,_,x,prior,_,stage=span_features(
        logits,edge,parents,token_counts,incumbent["proposals"],float(incumbent["temperature"]),max_tier=tier)
    current=prior+float(incumbent["share"])*(learned_scores(x[:,:69],incumbent)-prior)
    p0,p1=x[:,69],x[:,70]
    features=np.column_stack((x,np.minimum(p0,p1),np.sqrt(p0*p1),stage==0,stage==1,current)).astype("f4")
    return arcs,stage,features,current


def density_scores(features, weights):
    y=np.clip(((features-weights["center"])*weights["scale"])@weights["coefficients"]+weights["bias"],0,1)
    return y[:,0],np.sqrt(y[:,1]*y[:,2])


def density_values(current, stage, predictions, settings):
    _,mix,factor,cost,fee=settings
    exact,both=predictions
    estimate=exact+factor*(both-exact)
    value=current+mix*(estimate-current)
    return value-cost-fee*(stage>0)


def density_labels(logits, edge, parents, token_counts, incumbent, weights):
    settings=weights["settings"]
    arcs,stage,features,current=density_features(logits,edge,parents,token_counts,incumbent,int(settings[0]))
    values=density_values(current,stage,density_scores(features,weights),settings)
    return best_path(arcs,values,len(token_counts))


def segment_labels(logits, edge, parents, token_counts, weights):
    spans_, support, features, prior, _ = span_features(
        logits, edge, parents, token_counts, weights["proposals"], float(weights["temperature"]),
        bool(weights.get("posterior_ends",False)))
    score = prior+float(weights["share"])*(learned_scores(features,weights)-prior)
    keep = slice(None) if bool(weights["extended"]) else support>0
    return best_path(spans_[keep],score[keep]-float(weights["span_cost"]),len(token_counts))
