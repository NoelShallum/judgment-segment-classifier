"""Write a two-encoder referral readout around the frozen pairwise classifier."""
import argparse
from itertools import product
import json

import numpy as np
from safetensors.numpy import load_file, save_file

import mixedbread_referral_write as small
import predict_pairwise_judgment as runtime
import sentence_pairwise_repair as repair
from sentence_document_graph_write import OUT, cm, file_hash, paired, row_summary, save_json

P = repair.P
B = small.bert
TAG = "mixedbread-referral-overlay"
BERT = "BERT previous two and target"


def path(name):
    return P.RESULT / (name + "-frozen.json")


def sources():
    return {name:file_hash(path(name)) for name in (small.TAG, repair.TAG, B.TAG)}


def embeddings(split):
    rows, vec, proof = B.encoded(split, (BERT,))
    revision = json.loads((small.BASE/"provenance.json").read_text())
    width = revision["sha256"]["model.safetensors"][:12]
    mpath = small.CACHE / (split+"-"+width+".npy")
    vectors = np.load(mpath,allow_pickle=False)
    assert vectors.shape == (sum(len(r[3]) for r in rows),384)
    assert np.isfinite(vectors).all()
    aux = np.concatenate([r[4] for r in rows])
    return rows, np.column_stack((aux, vectors)), vec[BERT], (file_hash(mpath),proof[BERT]["features_sha256"])


def get_views(split, recipe, predictor):
    bank = P.dense.prepare(split)
    lex = P.cached_scores(split)
    baseline, unary, _ = repair.incumbent(bank,lex,recipe)["sequence only"]
    current = predictor(bank,lex,unary,baseline)
    expected = P.dense.read(repair.TAG)["splits"][split]["arms"]["sequence only"]["labels"]
    assert row_summary(cm(bank[2],current)) == expected
    rows, x, bvec, checksum = embeddings(split)
    np.testing.assert_array_equal(bank[0],[r[0] for r in rows])
    np.testing.assert_array_equal(bank[2],np.concatenate([r[1] for r in rows]))
    np.testing.assert_array_equal(bank[3],np.concatenate([r[2] for r in rows]))
    mapping=np.concatenate([q+lo for lo,(_,_,_,q,_,_,_) in zip(bank[1], rows)])
    post = np.concatenate([r[5] for r in rows])
    return bank,current,rows,mapping,post,x,bvec,checksum


def dimensions(rows, x, vec, plan):
    w = OUT / (B.TAG+"-2.safetensors")
    assert file_hash(w)==plan["checkpoints"][BERT]
    cfg = load_file(str(w))
    mode,*_,gain,cut = plan["frozen_modes"][BERT]
    ber = B.response(B.design(rows,vec,mode,cfg),cfg)
    return np.column_stack((x,vec)), ber, (gain, cut)


def mixed(x):
    recipe = P.dense.read(small.TAG+"-frozen")["heads"]["Mixedbread target pooled"]
    weight = OUT / (small.TAG+"-1.safetensors")
    if not recipe["sha256"]:
        return None, None
    assert file_hash(weight)==recipe["sha256"]
    return B.response(x,load_file(str(weight))), tuple(recipe["params"][-2:])


def edit(base, positions, prior, confidence, gain, gap, original=None):
    if confidence is None:
        return base
    loc = np.arange(len(positions))
    old = base[positions]
    anchor = .5*prior + .5*np.eye(7)[old]
    prob = (1.-gain)*anchor + gain*confidence
    new = prob.argmax(1)
    permitted = prob[loc,new] - prob[loc,old] >= gap
    if original is not None:
        permitted &= new != original
    answer = base.copy()
    answer[positions[permitted]] = new[permitted]
    return answer


def boot():
    origin = B.read(B.TAG+"-frozen")
    return repair.load_recipe(), runtime.predictor("direct"), origin


def fit():
    recipe, predict, plan = boot()
    rows0, x0, e0, fit_source = embeddings("fit")
    bank, base, rows, index, prior, x, embedding, dev_source = get_views("select",recipe,predict)
    assert not set(r[0] for r in rows0) & set(bank[0])
    targets = np.concatenate([r[1][r[3]] for r in rows0])
    train = np.column_stack((x0,e0))
    features, bert_score, bert_default = dimensions(rows,x,embedding,plan)
    mixed_score, _ = mixed(x)
    old = np.concatenate([r[2][r[3]] for r in rows])
    initial = row_summary(cm(bank[2],base))
    by_method = {}
    grid = {}

    def choose(name, scores, tags, weight=None):
        best = by_method.setdefault(name, {"stats":[initial["macro_f1"],initial["accuracy"],0],"choice":None})
        for gain, gap, veto in product((.24,.38,.56,.75,1.),(.02,.08,.16),(False,True)):
            yp = edit(base,index,prior,scores,gain,gap,old if veto else None)
            labels = row_summary(cm(bank[2],yp))
            cand = [labels["macro_f1"],labels["accuracy"],-int(np.count_nonzero(yp!=base))]
            grid.setdefault(name,[]).append([tags,(gain,gap,veto),cand])
            if cand > best["stats"] and cand[1] >= initial["accuracy"]-.001:
                by_method[name] = {"stats":cand,"choice":[*tags,gain,gap,veto]}
                if weight is not None:
                    save_file(weight,str(OUT/(TAG+".safetensors")))
                best = by_method[name]

    observed = P.dense.read(B.TAG)
    comparison = np.concatenate(B.apply(rows,bert_score,*bert_default))
    assert row_summary(cm(bank[2],comparison)) == observed["splits"]["select"]["arms"][BERT]["scores"]
    choose("BERT only",bert_score,())
    if mixed_score is not None:
        choose("Mixedbread only",mixed_score,())
        for alpha in (.25,.50,.75):
            choose("probability blend",(1-alpha)*bert_score+alpha*mixed_score,(alpha,))
    for rare in (0.,.26,.52):
        for penalty,coeff in B.fit(train,targets,rare):
            choose("combined readout",B.response(features,coeff),(rare,penalty),coeff)
    if by_method["combined readout"]["choice"]:
        by_method["combined readout"]["checkpoint_sha256"] = file_hash(OUT/(TAG+".safetensors"))
    frozen = {"sources":sources(), "gradient_steps":0, "fit_cases":[r[0] for r in rows0],
              "selection_cases":bank[0].tolist(), "training_features_sha256":fit_source,
              "selection_features_sha256":dev_source, "baseline":initial, "arms":by_method, "grid":grid}
    save_json(path(TAG),frozen)
    print({k:(v["choice"],v["stats"]) for k,v in by_method.items()},flush=True)


def evaluate(splits):
    frozen = json.loads(path(TAG).read_text())
    assert frozen["sources"] == sources()
    recipe, predict, plan = boot()
    repro = P.RESULT/(TAG+".json")
    out = {"freeze_sha256":file_hash(path(TAG)),"splits":{}}
    if repro.is_file():
        old = json.loads(repro.read_text())
        if old["freeze_sha256"] == out["freeze_sha256"]:
            out = old
    for split in splits:
        bank, base, rows, index, prior, x, vec, features_hash = get_views(split,recipe,predict)
        assert not set(frozen["fit_cases"]) & set(bank[0])
        if split!="select":
            assert not set(frozen["selection_cases"]) & set(bank[0])
        else:
            assert frozen["selection_features_sha256"] == list(features_hash)
        features, b, bdefault = dimensions(rows,x,vec,plan)
        original_bert = np.concatenate(B.apply(rows,b,*bdefault))
        expected = P.dense.read(B.TAG)["splits"][split]["arms"][BERT]["scores"]
        bert_cf = P.dense.paired_confusion(bank,original_bert)
        assert expected == row_summary(bert_cf.sum(0))
        m, mdefault = mixed(x)
        small_cf = P.dense.paired_confusion(bank,np.concatenate(B.apply(rows,m,*mdefault))) if m is not None else None
        if small_cf is not None:
            assert row_summary(small_cf.sum(0)) == P.dense.read(small.TAG)["splits"][split]["heads"]["Mixedbread target pooled"]["labels"]
        original = np.concatenate([r[2][r[3]] for r in rows])
        count = P.dense.paired_confusion(bank,base)
        r = out["splits"][split] = {"cases":len(rows),"referred":len(index),"baseline":row_summary(count.sum(0)),
                                  "BERT original":expected, "features_sha256":features_hash, "arms":{}}
        author = None
        if split == "test":
            archives = P.RESULT / "known-sentence-numerical-archives.npz"
            metadata = P.dense.read("known-sentence-numerical-archives")
            assert file_hash(archives) == metadata["score_file_sha256"]
            with np.load(archives,allow_pickle=False) as source:
                np.testing.assert_array_equal(source["cases"],bank[0])
                author = source[metadata["best_by_macro_f1"]]
            np.testing.assert_array_equal(author.sum(2),count.sum(2))
            r["aligned_author"] = row_summary(author.sum(0))
        if small_cf is not None:
            r["Mixedbread original"]=row_summary(small_cf.sum(0))
            r["paired original Mixedbread vs BERT"]=paired(small_cf,bert_cf)
        for name, option in frozen["arms"].items():
            choose = option["choice"]
            pred = base
            if choose:
                *args,gain,gap,veto = choose
                if name=="combined readout":
                    loc = OUT/(TAG+".safetensors")
                    assert file_hash(loc)==option["checkpoint_sha256"]
                    scores = B.response(features,load_file(str(loc)))
                elif name=="probability blend":
                    scores = (1-args[0])*b+args[0]*m
                else:
                    scores = b if name=="BERT only" else m
                pred = edit(base,index,prior,scores,gain,gap,original if veto else None)
            assert set(np.flatnonzero(pred!=base)).issubset(set(index))
            cf = P.dense.paired_confusion(bank,pred)
            q = r["arms"][name] = {"labels":row_summary(cf.sum(0)),"paired_baseline":paired(cf,count),
                                   "paired_original_bert":paired(cf,bert_cf),"changed":int(np.sum(pred!=base))}
            if author is not None:
                q["paired_aligned_author"] = paired(cf,author)
            if split=="select":
                assert [q["labels"]["macro_f1"],q["labels"]["accuracy"],-q["changed"]] == option["stats"]
        save_json(repro,out)
        print(split,{k:(v["labels"]["macro_f1"],v["labels"]["accuracy"],v["changed"])
                     for k,v in r["arms"].items()},flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("fit","evaluate"))
    p.add_argument("splits",nargs="*")
    args=p.parse_args()
    if args.action=="fit":
        fit()
    else:
        evaluate(args.splits or small.EVAL_SPLITS)
