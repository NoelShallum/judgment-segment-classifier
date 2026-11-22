"""Compare a document-only weight write with two ways of adding isolated evidence."""
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from itertools import product
import json

import numpy as np
from safetensors.numpy import save_file
from scipy.special import softmax

from confirm_sentence_graph_holdout import gather_validation
from hier_full import CACHE
from isolated_role_readout_write import design, NGRAM
from sentence_document_graph_write import (
    DONOR, EDGE, OUT, PARENT, ROOT, STEM, HierModel, aligned_embeddings, case_inputs,
    cm, document, file_hash, gold_cases, graph_features, load_file, measure, paired,
    route_cases, row_summary, sample_whole, save_json, score, torch,
)
from sentence_joint_runtime import neighbors

TAG = "sentence-readout-fusion"
RESULT = ROOT / "results"
FIT_CASES = 144


def record(name):
    return json.loads((RESULT / (name + ".json")).read_text())


def block(args):
    docs, split = args
    torch.set_num_threads(1)
    model = HierModel(DONOR)
    parent = load_file(str(PARENT))
    vocab = json.loads((EDGE / "vocab.json").read_text())
    g = load_file(str(OUT / (STEM + "-two-hop.safetensors")))
    solo = load_file(str(OUT / "isolated-role-readout.safetensors"))
    table = load_file(str(OUT / (NGRAM + "-memory.safetensors")))["lexical"][0]
    settings = record(STEM + "-frozen")["choices_by_arm"]["two-hop"]
    isolated = record("isolated-role-readout-frozen")["choice"]
    assert isolated[:2] == ["combined", "lda"]
    _, _, _, kind, budget, gain, guard, cut = settings
    assert kind == "posterior"
    coeff = g["scale"][:, None] * g["matrix"]
    bias = g["bias"] - g["center"] @ coeff
    rows = []
    for item, ctx in zip(docs, aligned_embeddings(model, docs, split)):
        c = case_inputs(item, ctx, model, parent, vocab)
        local, sem, graph = graph_features(c)
        z = graph @ coeff + bias
        current = score([(c, None)], [z], route_cases([(c, None)], budget), kind, gain, guard, cut)[0]
        v = np.maximum(z[:, :7], 0.)
        v /= np.maximum(v.sum(1, keepdims=True), 1e-8)
        posterior = (1-gain)*c.prior + gain*v
        if split == "test":
            with np.load(CACHE / (c.key + ".npz")) as data:
                state = data["excerpt"][item["source_mask"]]
        else:
            state = model.contextualize(model.words(c.texts), independent=True)
        x = design(c.texts, state, table)[2]
        logits = ((x - solo["center"])*solo["scale"]) @ solo["weight"]
        logits += solo["bias"] + isolated[-1]*solo["prior"]
        rows.append((c.key, c.gold, current, c.structure[:, :33], c.prior, posterior,
                     logits.astype("f4")))
    return rows


def prepare(docs, split):
    with ProcessPoolExecutor(4) as pool:
        shards = pool.map(block, [(docs[i::4], split) for i in range(4)])
        rows = {row[0]: row for part in shards for row in part}
    return [rows[d["id"]] for d in docs]


def features(row, add_isolated):
    _, y, current, surface, prior, graph, logits = row
    one = np.eye(7)[current]
    base = np.column_stack((surface, one, graph, prior*graph))
    if not add_isolated:
        return base
    isolated = softmax(logits / 6, axis=1)
    left, right = neighbors(isolated, np.arange(len(y)) == 0)
    relative = np.clip(logits-logits.max(1, keepdims=True), -55., 0.) / 10
    return np.column_stack((base, isolated, relative, graph*isolated, left, right))


def learn(rows, features_by_case, rarity):
    x = np.concatenate(features_by_case)
    labels = np.concatenate([row[1] for row in rows])
    target = np.eye(7)[labels]
    center = x.mean(0)
    scale = 1/np.maximum(x.std(0), .025)
    z = np.column_stack(((x-center)*scale, np.ones(len(x))))
    freq = np.maximum(target.mean(0), .015)
    w = freq[labels] ** -rarity
    design = z.T @ (w[:, None]*z)
    rhs = z.T @ (w[:, None]*target)
    diag = np.r_[np.ones(z.shape[1]-1), .02] * design.trace()/len(design)
    for ridge in (.015,.12,.6):
        coeff = np.linalg.solve(design+np.diag(ridge*diag), rhs)
        yield ridge, {"center":center, "scale":scale, "coeff":coeff[:-1].copy(), "bias":coeff[-1].copy()}


def response(xs, weight):
    return [((x-weight["center"])*weight["scale"])@weight["coeff"]+weight["bias"] for x in xs]


def simplex(a):
    p = np.maximum(a, 0.)
    return p/np.maximum(p.sum(1, keepdims=True),1e-8)


def candidates(rows, evidence, gain, threshold, protect):
    result = []
    for (_,_,current,_,prior,post,_), new in zip(rows,evidence):
        anchor = .55*post + .15*prior + .30*np.eye(7)[current]
        scores = (1-gain)*anchor + gain*new
        best = scores.argmax(1)
        i = np.arange(len(current))
        accept = (scores[i,best]-scores[i,current] >= threshold)
        if protect:
            accept &= current != 5
        result.append(np.where(accept,best,current))
    return result


def confusion(rows, outputs):
    return np.stack([cm(row[1], y) for row,y in zip(rows,outputs)])


def main():
    torch.set_num_threads(1)
    old_graph = record(STEM + "-comparison")
    old_isolated = record("isolated-role-readout")
    hold = record("sentence-graph-holdout-frozen")
    val_csv = ROOT/"references/official/val.csv"
    train_csv = ROOT/"references/official/train.csv"
    assert hold["val_csv_sha256"] == file_hash(val_csv)
    sources = {name:file_hash(OUT/name) for name in (PARENT.name,STEM+"-two-hop.safetensors",
               "isolated-role-readout.safetensors",NGRAM+"-memory.safetensors")}
    assert sources[STEM+"-two-hop.safetensors"] == old_graph["comparison_weights_sha256"]["two-hop"]
    assert sources["isolated-role-readout.safetensors"] == record("isolated-role-readout-frozen")["weights_sha256"]
    assert sources[NGRAM+"-memory.safetensors"] == record(NGRAM+"-frozen")["memory_sha256"]
    blocked = set(record(STEM+"-frozen")["train_cases"])
    blocked.update(record("isolated-role-readout-frozen")["training_cases"])
    blocked.update(record(NGRAM+"-frozen")["memory_training_cases"])
    tr = gather_validation(train_csv)
    keys = sorted((k for k in tr if k not in blocked and tr[k][1]),
                  key=lambda k:sha256((TAG+"-v1|"+k).encode()).digest())[:FIT_CASES]
    print("Preparing",len(keys),"previously unused calibration cases",flush=True)
    train = prepare([document(k,tr[k][0]) for k in keys],"fit")
    del tr
    cohorts = sample_whole()
    dev = prepare(cohorts["select"],"select")
    baseline = row_summary(confusion(dev,[r[2] for r in dev]).sum(0))
    assert baseline == old_graph["splits"]["select"]["arms"]["two-hop"]["scores"]
    assert row_summary(confusion(dev,[r[-1].argmax(1) for r in dev]).sum(0)) == old_isolated["splits"]["select"]["candidate"]
    print("Baseline validated",baseline["macro_f1"],"fit rows",sum(len(r[1]) for r in train),flush=True)
    grid = []
    picks = {}
    weights = {}

    def rank(arm, setup, estimates):
        for gain, margin, protect in product((.25,.50,.75,1.),(.02,.075,.14),(False,True)):
            opt = (*setup,gain,margin,protect)
            m,a,*_ = measure(confusion(dev,candidates(dev,estimates,gain,margin,protect)).sum(0))
            grid.append([arm,opt,float(m),float(a)])
            if a >= baseline["accuracy"]-.001:
                pick = (m,a,opt)
                if arm not in picks or pick > picks[arm]:
                    picks[arm] = pick

    temps = (2.,4.,6.,12.)
    for temp in temps:
        rank("probability blend",(temp,),[softmax(r[-1]/temp,axis=1) for r in dev])
    for arm, add in (("document only",False),("document and isolated",True)):
        x = [features(r,add) for r in train]
        d = [features(r,add) for r in dev]
        for rarity in (0.,.22,.45):
            for ridge, w in learn(train,x,rarity):
                weights[arm,rarity,ridge] = w
                rank(arm,(rarity,ridge),[simplex(z) for z in response(d,w)])
    print("Validation selections",picks,flush=True)
    choice = {arm:option for arm,(_,_,option) in picks.items()}
    hashes = {}
    for arm in ("document only","document and isolated"):
        path = OUT/(TAG+"-"+arm.replace(" ","-")+".safetensors")
        w = weights[(arm,*choice[arm][:2])]
        save_file(w,str(path))
        with_weight = load_file(str(path))
        assert all(np.array_equal(v,with_weight[k]) for k,v in w.items())
        weights[(arm,*choice[arm][:2])] = with_weight
        hashes[path.name] = file_hash(path)
    freeze = RESULT/(TAG+"-frozen.json")
    hold_path = RESULT/"sentence-graph-holdout-frozen.json"
    save_json(freeze,{"cases_for_fit":keys,"fit_rows":sum(len(r[1]) for r in train),
                      "select_cases":[r[0] for r in dev],"train_sha256":file_hash(train_csv),
                      "selection_policy":"Highest macro-F1 within 0.10 percentage points of the document baseline accuracy on select only, separately for each arm.",
                      "choices":choice,"source_weights_sha256":sources,"fitted_weights_sha256":hashes,
                      "holdout_sha256":file_hash(hold_path),"grid":grid})
    del train
    val = gather_validation(val_csv)
    cohorts["holdout"] = [document(k,val[k][0]) for k in hold["cases"]]
    cohorts["test"] = []
    for k,v in gold_cases().items():
        d = document(k,v)
        d["source_mask"] = np.asarray([bool(t.strip()) for _,t,_ in v])
        cohorts["test"].append(d)
    assert len(set.intersection(set(hold["cases"]),set(d["id"] for d in cohorts["select"]+cohorts["confirm"]))) == 0
    archive_path = RESULT/"known-sentence-numerical-archives.npz"
    author_meta = record("known-sentence-numerical-archives")
    assert file_hash(archive_path) == author_meta["score_file_sha256"]
    with np.load(archive_path,allow_pickle=False) as a:
        np.testing.assert_array_equal(a["cases"],[d["id"] for d in cohorts["test"]])
        author = a[author_meta["best_by_macro_f1"]]
    out = {"freeze_sha256":file_hash(freeze),"aligned_author_archive":author_meta["best_by_macro_f1"],"splits":{}}
    for group in ("select","confirm","holdout","test"):
        rows = dev if group=="select" else prepare(cohorts[group],group)
        base = confusion(rows,[r[2] for r in rows])
        solo = confusion(rows,[r[-1].argmax(1) for r in rows])
        q = out["splits"][group] = {"cases":len(rows),"base":row_summary(base.sum(0)),
                                    "isolated":row_summary(solo.sum(0)),"arms":{}}
        expected = old_graph["splits"][group]["arms"]["two-hop"] if group!="holdout" else record("sentence-graph-holdout")["arms"]["two-hop"]
        assert expected["scores"] == q["base"]
        assert old_isolated["splits"][group]["candidate"] == q["isolated"]
        controls = {}
        for arm,option in choice.items():
            *args,gain,cut,protect = option
            if arm=="probability blend":
                evidence = [softmax(r[-1]/args[0],axis=1) for r in rows]
            else:
                evidence = [simplex(z) for z in response([features(r,arm=="document and isolated") for r in rows],
                                                           weights[arm,*args])]
            predictions = candidates(rows,evidence,gain,cut,protect)
            corr = controls[arm] = confusion(rows,predictions)
            v = q["arms"][arm] = {"scores":row_summary(corr.sum(0)),"paired_to_base":paired(corr,base),
                       "changed":sum(int((p!=r[2]).sum()) for r,p in zip(rows,predictions))}
            if group=="test":
                v["paired_to_aligned_author"] = paired(corr,author)
        q["paired_isolated_contribution"] = paired(controls["document and isolated"],controls["document only"])
        save_json(RESULT/(TAG+".json"),out)
        print(group,q["base"]["macro_f1"],
              {arm:(round(r["scores"]["macro_f1"],7),round(r["scores"]["accuracy"],7),r["changed"])
               for arm,r in q["arms"].items()},flush=True)


if __name__ == "__main__":
    main()
