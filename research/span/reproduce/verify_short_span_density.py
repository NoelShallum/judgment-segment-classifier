"""Verify the raw NumPy scorer and attribute recovered spans to proposal sources."""
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np

from role_edge_interpolation import EDGE,DONOR,HierModel,ROOT,bootstrap,document,file_hash,full_score,gold_cases,load_file,save_json,torch
from hier_full import CACHE
from hier_model import LABELS
from role_pair_weight_search import prepare_score,transition_scores
from predict_role_edge import infer_segments
from segment_span_runtime import density_features,density_scores,density_values
from short_span_weight_edit import setup
from span_model import merge
from span_posterior_decode import best_path,spans


def check(batch):
    docs,live,physical=batch
    torch.set_num_threads(1)
    parents,vocab,_,incumbent=setup()
    settings=physical["settings"]
    actual=[];same_weights_core=[];usage=np.zeros((6,2),np.int64)
    for d in docs:
        with np.load(CACHE/(d["id"]+".npz")) as r:
            item=prepare_score(d,r["raw"],parents)
        phi=transition_scores(d["text"],d["units"],vocab,parents["exact"])
        total=item.token_role_counts.sum(1)
        arcs,stage,feat,current=density_features(item.logits,phi,parents,total,incumbent,int(settings[0]))
        values=density_values(current,stage,density_scores(feat,physical),settings)
        result=best_path(arcs,values,len(total))
        core=stage==0
        control=best_path(arcs[core],values[core],len(total))
        actual.append(item.score(result))
        same_weights_core.append(item.score(control))
        n=len(total)
        _,idx,_=np.intersect1d(np.ravel_multi_index(arcs.T,(n,n+1,7)),
            np.ravel_multi_index(spans(result).T,(n,n+1,7)),assume_unique=True,return_indices=True)
        a,b,k=arcs[idx].T
        exact=(item.gold_label[a]==k)&(item.gold_end[a]==item.offsets[b])
        usage[:,0]+=np.bincount(stage[idx],minlength=6)
        usage[:,1]+=np.bincount(stage[idx][exact],minlength=6)
        if d["id"] in live:
            got=infer_segments(d["text"],HierModel(DONOR),parents,vocab,incumbent,physical)
            assert got["spans"]==merge(d["units"],result,LABELS),d["id"]
    return np.stack(actual),np.stack(same_weights_core),usage


def main():
    results=ROOT/"results"
    f=ROOT/"checkpoints/hierarchical/whole-short-span-density.safetensors"
    frozen=results/"short-span-density-frozen.json"
    receipt=json.loads(frozen.read_text())
    assert receipt["sha256"]==file_hash(f)
    choice=load_file(str(f))
    live={c["case"] for c in json.loads((EDGE/"qa.json").read_text())["cases"]}
    docs=[document(k,r) for k,r in gold_cases().items()]
    with ProcessPoolExecutor(4) as pool:
        chunks=list(pool.map(check,[(docs[i::4],live,choice) for i in range(4)]))
    actual=np.concatenate([r[0] for r in chunks])
    control=np.concatenate([r[1] for r in chunks])
    usage=sum(r[2] for r in chunks)
    score=full_score(actual)
    assert usage.sum(0).tolist()==[score["exact_labeled_spans"][k] for k in ("predicted","matched")]
    diagnostic={"cases":len(docs),"live_cases":sorted(live),"sha256":file_hash(f),"freeze_sha256":file_hash(frozen),
          "scores":score,"same_weights_original_lattice":full_score(control),
          "gap_ci95_same_weights_original_lattice":bootstrap(actual,control),
          "chosen_span_sources":[{"tier":i,"predicted":int(a),"exact_matches":int(b)} for i,(a,b) in enumerate(usage)]}
    # The shared export checks this against the independently produced benchmark.
    target=results/"short-span-density-edit.json"
    if target.exists() and "test" in (bench:=json.loads(target.read_text()))["splits"]:
        assert bench["splits"]["test"]["candidate"]==score
    save_json(results/"short-span-density-qa.json",diagnostic)
    print("Dense scorer independently replayed",len(docs),score["exact_labeled_spans"],flush=True)


if __name__=="__main__":
    main()
