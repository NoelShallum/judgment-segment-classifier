"""Compute exact role spans from frozen labels using the same official sentence boundaries."""
import json
import hashlib
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def runs(y):
    starts=np.flatnonzero(np.r_[True,y[1:]!=y[:-1]])
    return starts,np.r_[starts[1:],len(y)],y[starts]


def tally(gold,pred):
    a,b,y=runs(gold)
    c,d,p=runs(pred)
    _,i,j=np.intersect1d(a,c,return_indices=True,assume_unique=True)
    success=(b[i]==d[j])&(y[i]==p[j])
    return np.asarray((success.sum(),len(c),len(a),len(i)-1,len(c)-1,len(a)-1))


def measures(stats):
    m,n,g,m_edge,n_edge,g_edge=stats.sum(0)
    return {"matched":int(m),"predicted":int(n),"gold":int(g),"precision":float(m/n),"recall":float(m/g),
            "exact_labeled_span_f1":float(2*m/(n+g)),"role_boundary_f1":float(2*m_edge/(n_edge+g_edge))}


def interval(a,b):
    draw=np.random.default_rng(633191).integers(len(a),size=(1800,len(a)))
    def f1(x):
        vs=x[draw].sum(1)
        return 2*vs[:,0]/(vs[:,1]+vs[:,2])
    return np.quantile(f1(a)-f1(b),[.025,.975]).tolist()


def main():
    names={"native document prior":(HERE/"priors/official-targeted-predictions.npz","offsets","targeted"),
           "native donor":(HERE/"priors/official-targeted-predictions.npz","offsets","donor"),
           "native head only":(HERE/"hierarchy/uniform-retention-test-score.npz","ends","head"),
           "native head and retention":(HERE/"hierarchy/uniform-retention-test-score.npz","ends","targeted")}
    comp=HERE/"priors/composition-test-predictions.npz"
    if comp.exists():names["native head, retention and document prior"]=(comp,"offsets","gate and text bias")
    counts={}; files={};keys=gold=off=None
    for name,(path,offset,arm) in names.items():
        with np.load(path,allow_pickle=False) as src:
            if gold is None:keys,off,gold=src["keys"],src[offset],src["gold"]
            else:
                for expected,actual in ((keys,src["keys"]),(off,src[offset]),(gold,src["gold"])):np.testing.assert_array_equal(expected,actual)
            pred=src[arm]
        counts[name]=np.stack([tally(gold[a:b],pred[a:b]) for a,b in zip(off,off[1:])])
        files[name]={"file":str(path.relative_to(HERE)),"sha256":sha(path),"array":arm}
    assert len(keys)==712 and len(gold)==149868
    arch=json.loads((ROOT/"results/sentence-joint-pairwise-span-archive.json").read_text())
    inc=json.loads((ROOT/"results/sentence-wide-bert-benchmark.json").read_text())["splits"]["test"]["comparison"]["routes"]["confidence"]["spans"]
    out={"judgments":len(keys),"sentences":len(gold),"definition":"Maximal contiguous same-role run. Count true positive only if start sentence, end sentence and role match. Sentence cuts are supplied. Freeze captured before deriving these additional metrics.",
         "sources":files,"reference":{"archived author previous-predicted role":arch["public_scores"]["exact_labeled_spans"]["f1"],
          "historical AttoFlow span decoder":arch["edited_scores"]["exact_labeled_spans"]["f1"],"AttoFlow sentence system at 20% review":inc["exact_labeled_span_f1"]},
         "scores":{}}
    for k,v in counts.items():
        record=measures(v)
        assert record["gold"]==inc["gold_spans"]==28879
        record["paired_exact_span_f1_delta_vs_native_donor_ci95"]=interval(v,counts["native donor"])
        out["scores"][k]=record
    path=HERE/"official-role-spans.json"
    temp=path.with_suffix(".part")
    temp.write_text(json.dumps(out,indent=2)+"\n");temp.replace(path)
    print(out["reference"],{k:v["exact_labeled_span_f1"] for k,v in out["scores"].items()})


if __name__=="__main__":main()
