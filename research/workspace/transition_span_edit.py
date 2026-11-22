"""Search authored, role-specific edits to the public CRF on legal role spans."""
from itertools import product
import csv
from collections import defaultdict
import random

import numpy as np
from safetensors.numpy import load_file,save_file
import torch

from fragment_trial import passages
from hier_full import CACHE, gold_cases
from hier_model import CRF, HierModel, path_labels
from hier_trial import DONOR, OUT
from program_trial import TO_ID, fold, save_json
from seven_paths import ROOT
from span_eval import counts, document, summarize
from weight_edit import file_hash

DIAG=np.arange(3,10)
SPEAKER=np.array([6,7])


def sample_whole():
    docs=defaultdict(list)
    with (ROOT/"references/official/val.csv").open(newline="",encoding="utf-8-sig") as f:
        for i,r in enumerate(csv.DictReader(f)):
            docs[r["Index"]].append((i,r["Text"],TO_ID[r["Label"]]))
    out={}
    for name,confirmation,seed in (("select",False,6520),("confirm",True,6521)):
        pool=sorted(k for k,v in docs.items() if (fold(k)==0)==confirmation and 40<=len(v)<=750)
        ids=random.Random(seed).sample(pool,72)
        out[name]=[document(k,docs[k]) for k in ids]
    return out


def features(m,docs):
    result=[]
    for d in docs:
        texts=[d["text"][a:b].strip() for a,b in d["units"]]
        result.append(m.logits(m.contextualize(m.words(texts))))
    return result


def write_transition(donor,overall,self_shift,speaker_shift,heading_shift,edge):
    w=donor.copy()
    w[3:,3:] *= overall
    w[DIAG,DIAG]+=self_shift
    w[SPEAKER,SPEAKER]+=speaker_shift
    w[3,3]+=heading_shift
    w[1,3:]*=edge
    w[3:,2]*=edge
    return w


def scores(docs,logits,w):
    return np.stack([counts(d,d["units"],path_labels(e,w)) for d,e in zip(docs,logits)])


def upper_bound(docs):
    exact=gold=change=0
    for d in docs:
        points={a for a,b in d["units"]}|{len(d["text"])}
        spans=d["gold_spans"]
        gold+=len(spans)
        exact+=sum(s["start"] in points and s["end"] in points for s in spans)
        change+=sum(s["start"] in points for s in spans[1:])
    return {"gold_spans":gold,"gold_spans_with_both_candidate_endpoints":exact,"reachable_gold_span_recall":exact/gold,
            "loose_exact_span_f1_ceiling":2*exact/(gold+exact),"reachable_interior_gold_boundaries":change}


def report(docs,raw,t,previous):
    a=scores(docs,raw,t);b=scores(docs,raw,previous)
    rng=np.random.default_rng(47721)
    deltas=[]
    for i in rng.integers(len(a),size=(1600,len(a))):
        x=summarize(a[i].sum(0));y=summarize(b[i].sum(0))
        deltas.append([x["exact_labeled_spans"]["f1"]-y["exact_labeled_spans"]["f1"],x["token_accuracy"]-y["token_accuracy"]])
    return {"candidate":summarize(a.sum(0)),"previous":summarize(b.sum(0)),"cases":len(a),
            "span_and_token_accuracy_gap_95pct":np.quantile(deltas,[.025,.975],axis=0).T.tolist(),
            "boundary_coverage":upper_bound(docs)}


def main():
    torch.set_num_threads(2)
    m=HierModel(DONOR)
    it=sample_whole()
    fragments=passages()
    out={}
    # The incumbent remains an eligible choice.
    for name,docs,wt in (("whole",it,"document"),("fragment",fragments,"fragment")):
        metadata=load_file(str(OUT/(wt+".safetensors")))
        previous=metadata[CRF] if name=="whole" else metadata["transition"]
        if name=="whole":
            docs["test"]=[document(k,v) for k,v in gold_cases().items()]
        print(name,"encoding development",len(docs["select"]),flush=True)
        arrays={k:features(m,d) for k,d in docs.items() if k!="test" or name=="fragment"}
        if name=="whole":
            arrays["test"]=[]
            for d in docs["test"]:
                with np.load(CACHE/(d["id"]+".npz")) as f:arrays["test"].append(f["raw"])
        if name=="fragment" and not np.isclose(metadata["classifier.weight"][:,200:],0).all():
            raise ValueError("Incumbent fragment profile needs isolated features too")
        base=summarize(scores(docs["select"],arrays["select"],previous).sum(0))
        best=(base["exact_labeled_spans"]["f1"],base["token_accuracy"])
        chosen=previous;spec="incumbent";grid=[]
        for a,b,c,d,e in product((0.25,.5,.75),(0.,-.4,.4),(-.45,0.,.45),(-.45,0.,.45),((1.,.0) if name=="whole" else (0.,))):
            edit=write_transition(m.transitions,a,b,c,d,e)
            metric=summarize(scores(docs["select"],arrays["select"],edit).sum(0))
            key=(metric["exact_labeled_spans"]["f1"],metric["token_accuracy"])
            grid.append([a,b,c,d,e,*key])
            if key>best:spec=[a,b,c,d,e];chosen=edit;best=key
        path=OUT/("role-specific-"+name+".safetensors")
        save_file({**load_file(str(OUT/"document.safetensors")),CRF:chosen},str(path))
        out[name]={"order":["scale","self shifts","party-argument diagonal shifts","none diagonal shift","edge scale"],
                   "settings":spec,"checkpoint_sha256":file_hash(path),"grid":grid,"scores":{}}
        for split in docs:
            out[name]["scores"][split]=report(docs[split],arrays[split],chosen,previous)
            s=out[name]["scores"][split]
            print(name,split,s["candidate"]["exact_labeled_spans"]["f1"],s["previous"]["exact_labeled_spans"]["f1"],flush=True)
        save_json(ROOT/"results/role-specific-transitions.json",{"selection":"Exact span F1 and then token accuracy on case-disjoint development. Direct shifts to seven-role CRF. Source splits previously explored, so results are comparative follow-up.","tasks":out})

if __name__=="__main__":main()
