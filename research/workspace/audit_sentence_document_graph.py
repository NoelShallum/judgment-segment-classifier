"""Measure role agreement on long-range sentence edges without fitting on test labels."""
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from embeddinggemma_routed_write import ROOT,document,gold_cases,save_json
from known_sentence_pair_edit import aligned_embeddings
from sentence_document_graph_write import remote_indices,scaled_mean
from seven_paths import LABELS


def uniform_role_probability(y,one):
    n=len(y)
    cutoff=np.arange(n)
    low=np.maximum(cutoff-4,0)
    high=np.minimum(cutoff+5,n)
    prefix=np.vstack((np.zeros((1,7)),np.cumsum(one,axis=0)))
    remaining=prefix[-1]-prefix[high]+prefix[low]
    total=n+low-high
    return remaining[cutoff,y]/np.maximum(total,1)


def geometry_matched_probability(y,indices,similarity,one):
    n=len(y)
    row=np.arange(n)[:,None]
    delta=indices-row
    distance=np.maximum(np.abs(delta),5)
    band=2**np.floor(np.log2(distance)).astype(int)
    near=np.maximum(band,5)
    far=band*2
    after=delta>0
    a=np.where(after,row+near,row-far+1).clip(0,n)
    b=np.where(after,row+far,row-near+1).clip(0,n)
    prefix=np.vstack((np.zeros((1,7)),np.cumsum(one,axis=0)))
    assert np.all((indices>=a)[np.isfinite(similarity)]&(indices<b)[np.isfinite(similarity)])
    common=(prefix[b,y[:,None]]-prefix[a,y[:,None]])/np.maximum(b-a,1)
    support=np.isfinite(similarity)
    pivot=np.max(np.where(support,similarity,-1.e6),axis=1,keepdims=True)
    weights=np.where(support,np.exp(8*(np.where(support,similarity,pivot)-pivot)),0)
    return (weights*common).sum(1)/np.maximum(weights.sum(1),1)


def run(docs):
    counts=np.zeros(7,dtype=np.int64)
    nearest=np.zeros(7,np.float64)
    any_role=np.zeros(7,np.float64)
    matched=np.zeros(7,np.float64)
    for d,h in zip(docs,aligned_embeddings(None,docs,"test")):
        y=np.asarray(d["reference_roles"])
        one=np.eye(7)[y]
        j,w=remote_indices(h)
        q,eligible=scaled_mean(one,j,w)
        i=np.arange(len(y))
        baseline=uniform_role_probability(y,one)
        g=y[eligible]
        counts+=np.bincount(g,minlength=7)
        nearest+=np.bincount(g,weights=q[i,y][eligible],minlength=7)
        any_role+=np.bincount(g,weights=baseline[eligible],minlength=7)
        control=geometry_matched_probability(y,j,w,one)
        matched+=np.bincount(g,weights=control[eligible],minlength=7)
    return counts,nearest,any_role,matched


def main():
    cases=[document(k,v) for k,v in gold_cases().items()]
    with ProcessPoolExecutor(3) as pool:
        parts=list(pool.map(run,[cases[i::3] for i in range(3)]))
    counts,chosen,uniform,geometry=(sum(x[i] for x in parts) for i in range(4))
    assert np.all(chosen<=counts) and np.all(uniform<=counts) and np.all(geometry<=counts)
    output={"cases":len(cases),"sentences_with_remote_edges":int(counts.sum()),"role":{},
            "protocol":"Same-role gold agreement with the eight weighted cosine neighbors beyond four sentences. The control is the exact expected agreement if a remote sentence is chosen uniformly from the same judgment. A second control matches each edge’s direction and power-of-two distance band before sampling uniformly. This audit diagnoses graph structure using human labels after edge construction."}
    for name,total,n,u,g in zip(LABELS,counts,chosen,uniform,geometry):
        output["role"][name]={"sentences":int(total),"semantic_neighbor_agreement":float(n/total),
                              "uniform_remote_agreement":float(u/total),"distance_matched_agreement":float(g/total)}
    output["overall"]={"semantic_neighbor_agreement":float(chosen.sum()/counts.sum()),
                      "uniform_remote_agreement":float(uniform.sum()/counts.sum()),
                      "distance_matched_agreement":float(geometry.sum()/counts.sum())}
    save_json(ROOT/"results/sentence-document-graph-affinity.json",output)
    print(output["cases"],output["sentences_with_remote_edges"],output["overall"],flush=True)


if __name__=="__main__":
    main()
