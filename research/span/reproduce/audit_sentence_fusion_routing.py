"""Compute gold-assisted headroom for fixed low-confidence sentence selectors."""
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import json

import numpy as np
from safetensors.numpy import load_file

from sentence_minilm_fusion import (
    DONOR,EDGE,MAIN,RATES,ROOT,HierModel,aligned_embeddings,case_view,cm,document,gold_cases,
    row_summary,save_json,torch,
)

METHODS=("released-role confidence","entropy","largest alternative margin","decoder disagreement")


def audit(cases):
    torch.set_num_threads(1)
    model=HierModel(DONOR)
    parent=load_file(str(MAIN))
    v=json.loads((EDGE/"vocab.json").read_text())
    count=np.zeros((len(METHODS),len(RATES),7,2),np.int64)
    score=np.zeros((len(METHODS),len(RATES),7,7),np.int64)
    baseline=np.zeros((7,7),np.int64)
    for d,s in zip(cases,aligned_embeddings(model,cases,"test")):
        row=case_view(d,s,model,parent,v,max(RATES))
        b,y=row.base,row.gold
        mask=b!=y
        p=row.structural[:,:7].astype(np.float64)
        q=row.structural[:,7:14].astype(np.float64)
        m=.45*p+.55*q
        i=np.arange(len(b))
        entropy=-(m*np.log(np.maximum(m,1e-12))).sum(1)
        other=m.copy()
        other[i,b]=0.
        metrics=(1.-m[i,b],entropy,other.max(1)-m[i,b],np.abs(p-q).sum(1))
        baseline+=cm(y,b)
        for j,a in enumerate(metrics):
            order=np.argsort(-a,kind="stable")
            for k,rate in enumerate(RATES):
                eligible=order[:max(1,int(np.ceil(len(b)*rate)))]
                found=eligible[mask[eligible]]
                oracle=b.copy()
                oracle[eligible]=y[eligible]
                score[j,k]+=cm(y,oracle)
                count[j,k,:,0]+=np.bincount(y[eligible],minlength=7)
                count[j,k,:,1]+=np.bincount(y[found],minlength=7)
    return baseline,count,score


def main():
    docs=[document(k,r) for k,r in gold_cases().items()]
    with ProcessPoolExecutor(2) as workers:
        chunks=list(workers.map(audit,[docs[i::2] for i in range(2)]))
    before,n,score=(sum(x) for x in zip(*chunks))
    baseline=row_summary(before)
    found=json.loads((ROOT/"results/sentence-label-change-qa.json").read_text())["scores"]
    assert baseline==found
    errors=baseline["rows"]-baseline["correct"]
    methods={}
    for i,method in enumerate(METHODS):
        methods[method]={}
        for k,rate in enumerate(RATES):
            queried,opportunities=n[i,k].sum(0)
            ceiling=row_summary(score[i,k])
            item={"queried":int(queried),"routed_errors":int(opportunities),
                  "error_recall":float(opportunities/errors),"oracle":ceiling,
                  "role_counts":n[i,k].tolist()}
            methods[method][str(rate)]=item
            print(method,rate,item["routed_errors"],ceiling["macro_f1"],flush=True)
    save_json(ROOT/"results/sentence-minilm-routing-audit.json",
        {"cases":len(docs),"baseline":baseline,"methods":methods,
         "protocol":"Use only the frozen hierarchical predictions and posterior evidence to route the highest-risk fraction of each judgment. Apply gold labels only after routing to calculate error recall and an oracle that fixes every routed label."})


if __name__=="__main__":
    main()
