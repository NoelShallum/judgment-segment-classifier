"""Boundary and exact labeled span accounting for given, original CSV sentence boundaries."""
import argparse
from pathlib import Path
import numpy as np
from prepare import ROOT, save

def f1(a,b,c):
    return float(2*a/(b+c)) if b+c else 1.

def spans(values):
    edges=np.flatnonzero(np.diff(values)!=0)+1
    return set(zip([0,*edges],[*edges,len(values)],values[[0,*edges]].tolist())),set(edges.tolist())

def compare(data, names):
    index=next(k for k in ["sentence_index","ix"] if k in data)
    cases=np.unique(data["case"]);gold=data['gold']
    out={}
    for name in names:
        count=np.zeros(6,int)
        for cid in cases:
            loc=np.flatnonzero(data['case']==cid);loc=loc[np.argsort(data[index][loc])]
            if not np.array_equal(data[index][loc],np.arange(len(loc))):raise ValueError("Whole judgments are required for segmentation.")
            (a,aa),(b,bb)=spans(gold[loc]),spans(data[name][loc])
            count+=[len(a&b),len(a),len(b),len(aa&bb),len(aa),len(bb)]
        m,ng,npred,m2,n2,p2=map(int,count)
        out[name]={"exact_labeled_run_micro_f1":f1(m,ng,npred),"matched_spans":m,"gold_spans":ng,"predicted_spans":npred,
                   "sentence_boundary_f1":f1(m2,n2,p2),"matched_boundaries":m2,"gold_boundaries":n2,"pred_boundaries":p2}
    return {"definition":"Within each judgment, combine consecutive original CSV sentences bearing one identical role. An exact labeled run requires both end boundaries plus the class to match. Boundary F1 scores class changes between sentences; literal character boundary detection is unmeasured.",
            "rows":len(gold),"cases":len(cases),"methods":out}
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("npz",type=Path);parser.add_argument("out",type=Path)
    parser.add_argument("--methods",required=True);z=parser.parse_args()
    with np.load(z.npz) as d:out=compare(d,z.methods.split(","))
    save(z.out,out);print(out['cases'],out['rows'],{k:round(v['exact_labeled_run_micro_f1'],6) for k,v in out['methods'].items()})
