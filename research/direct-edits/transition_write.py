"""Write document-specific native CRF transition probabilities from training counts."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import softmax
from safetensors.numpy import load_file, save_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "transitions"
sys.path[:0] = [str(HERE / "priors"), str(HERE / "hierarchy"), str(ROOT)]
import experiment as prior
from prior_model import features
from scale_program import compiler
from span_comparison import tally, measures
from system import h95


def descriptor(text):
    x = features(text)
    return np.r_[x[:1024].reshape(256, 4).mean(1), x[1024:1536].reshape(128, 4).mean(1), x[1536:]]


def adjacent(labels):
    y = np.asarray(labels, dtype=np.int64)
    return np.bincount(7*y[:-1]+y[1:], minlength=49).reshape(7, 7)


def initialize():
    spec = OUT / "plan.json"
    if spec.exists():
        return json.loads(spec.read_text())
    select = json.loads((HERE/"priors/freeze.json").read_text())["ids"]
    shared = json.loads((HERE/"shared-confirmation.json").read_text())["extended_validation_judgment_ids"]
    blocked = set(select) | set(shared)
    scale = json.loads((HERE/"hierarchy/scale/plan.json").read_text())
    blocked.update(scale["validation"])
    cohort = json.loads((HERE/"priors/composition-freeze.json").read_text())
    blocked.update(cohort["reserved_ids"])
    salt = "attoflow-case-conditioned-transition-v1-20261122"
    candidates = sorted({str(k) for k in range(4984, 6408)}-blocked,
                        key=lambda k:hashlib.sha256((salt+k).encode()).digest())
    plan = dict(select=select, shared=shared, confirm=candidates[:128], salt=salt,
                gain=[0., .125, .25, .5, 1.], shrink=18, ridge_fraction=.9,
                native_target="crf.transitions[3:10,3:10] in the 10 by 10 public LegalSeg tensor; original START and STOP entries are preserved.",
                hypothesis="Each case's text predicts its seven by seven adjacent-role transition distribution. The native donor's own path supplies the baseline transition distribution. Add the centered and clipped log ratio to the native transition scores.",
                selection="Highest native sentence macro-F1 gain on selection with accuracy within 0.2 points of the donor, breaking ties by higher accuracy and smaller gain. Report all controls at the same frozen gain.",
                official_gate="Prospective 128 cases must pass a positive paired interval against the donor and both controls, and the edit must beat the actual current 20% system on the 48 shared cases. Otherwise retain validation only.",
                source_sha256=prior.file_hash(prior.DONOR/"hier.safetensors"), train_sha256=prior.file_hash(ROOT/"references/official/train.csv"))
    prior.write(spec, plan)
    return plan


def fit(plan):
    file = OUT/"case-transition-write.safetensors"
    if (OUT/"fit.json").exists():
        assert prior.file_hash(file) == json.loads((OUT/"fit.json").read_text())["weights_sha256"]
        return load_file(str(file))
    x, pairs, n = [], [], 0
    for _, text, labels in prior.cases("train"):
        x.append(descriptor(text))
        pairs.append(adjacent(labels))
        n += len(text)
    x, pairs = np.stack(x).astype(np.float64), np.stack(pairs)
    assert len(x) == 4984 and n == 1122376 and pairs.sum() == n-len(x)
    corpus = pairs.sum(0)+.5
    corpus /= corpus.sum(-1, keepdims=True)
    center = x.mean(0)
    scale = 1/np.maximum(x.std(0), .075)
    x = np.clip((x-center)*scale, -5, 5)
    local = pairs+plan["shrink"]*corpus
    logratio = np.log((local/local.sum(-1, keepdims=True))/corpus)
    logratio -= logratio.mean(-1, keepdims=True)
    y = logratio.reshape(len(x), 49)
    offset = y.mean(0)
    gram = x.T@x
    gram.flat[::len(gram)+1] += plan["ridge_fraction"]*gram.diagonal().mean()
    factor = cho_factor(gram)
    weights = dict(center=center, scale=scale, offset=offset, global_probability=corpus)
    rng = np.random.default_rng(8876112)
    for name, labels in (("target", y), ("shuffled", y[rng.permutation(len(y))])):
        weights[name] = cho_solve(factor, x.T@(labels-offset))
    save_file({k:np.ascontiguousarray(v) for k,v in weights.items()}, str(file))
    prior.write(OUT/"fit.json",dict(weights_sha256=prior.file_hash(file), training_cases=len(x), training_rows=n,
                                   actual_adjacent_pairs=int(pairs.sum()), features=x.shape[1], gradient_updates=0))
    return weights


def prepare(ids, w, plan, trans):
    bank = {}
    for key, text, gold in prior.cases("val", set(ids)):
        cache = prior.HERE/"cases"/(key+".npz")
        if cache.exists():
            with np.load(cache, allow_pickle=False) as f:
                np.testing.assert_array_equal(gold, f["g"])
                original, scores = f["prediction"], f["emissions"]
        else:
            import torch
            torch.set_num_threads(1)
            if not hasattr(prepare, "donor"):
                prepare.donor = prior.HierModel(prior.DONOR)
            donor = prepare.donor
            states = donor.contextualize(donor.words(text))
            scores = donor.logits(states)
            original = prior.path_labels(scores, trans)
        x = np.clip((descriptor(text)-w["center"])*w["scale"], -5, 5)
        pc = w["global_probability"]
        observed = adjacent(original)+plan["shrink"]*pc
        observed /= observed.sum(-1, keepdims=True)
        ds = {}
        for arm in ("target", "shuffled", "constant"):
            inferred = pc
            if arm != "constant":
                inferred = softmax(np.log(pc)+(x@w[arm]+w["offset"]).reshape(7, 7), axis=-1)
            delta = np.clip(np.log(inferred/observed), -2, 2)
            ds[arm] = delta-delta.mean(-1, keepdims=True)
        bank[key] = (gold, scores, original, ds)
    assert set(ids) == set(bank)
    return [bank[k] for k in ids]


def evaluate(data, trans, runner, gain, arm):
    conf, spans, patches = [], [], []
    for gold, scores, original, deltas in data:
        written = trans.copy()
        written[3:,3:] += gain*deltas[arm]
        pred = runner(scores, [0,len(scores)], written)
        if gain == 0:
            np.testing.assert_array_equal(pred, original)
        conf.append(prior.cm(gold, pred.astype(np.int64)))
        spans.append(tally(gold, pred))
        patches.append(written)
    conf, spans = np.stack(conf), np.stack(spans)
    return {**prior.stat(conf), **measures(spans)}, conf, np.stack(patches)


def main():
    OUT.mkdir(exist_ok=True)
    plan = initialize()
    w = fit(plan)
    trans = load_file(str(prior.DONOR/"hier.safetensors"))["crf.transitions"]
    runner = compiler()
    data = prepare(plan["select"],w,plan,trans)
    grid = {g:evaluate(data,trans,runner,g,"target")[0] for g in plan["gain"]}
    candidates = [g for g in grid if grid[g]["accuracy"] >= grid[0]["accuracy"]-.002]
    gain = max(candidates, key=lambda g:(grid[g]["macro_f1"],grid[g]["accuracy"],-g))
    prior.write(OUT/"freeze.json", dict(plan_sha256=prior.file_hash(OUT/"plan.json"),weights_sha256=prior.file_hash(OUT/"case-transition-write.safetensors"),
                                       gain=gain,selection_grid=grid))
    for name in ("shared", "confirm"):
        ids = plan[name]
        data = prepare(ids,w,plan,trans)
        stats, counts = {}, {}
        for arm in ("native","target","constant","shuffled"):
            stats[arm], counts[arm], patches = evaluate(data,trans,runner,0 if arm=="native" else gain,"target" if arm=="native" else arm)
            if name=="shared" and arm=="target":
                place = OUT/"shared-native-transitions.safetensors"
                save_file({f"judgment_{k}.crf.transitions":np.ascontiguousarray(mat) for k,mat in zip(ids,patches)}, str(place))
        if name=="shared":
            aligned = []
            for key, (gold, _, _, _) in zip(ids, data):
                ref = json.loads((HERE/"pooler/shared-cases"/(key+".json")).read_text())
                aligned.append(prior.cm(gold, np.asarray(ref["predictions"]["unchanged"])))
            stats["production"] = prior.stat(np.stack(aligned))
        deltas = {arm:h95(c,counts["target"]) for arm,c in counts.items() if arm!="target"}
        prior.write(OUT/(name+".json"), dict(ids=ids,cases=len(ids),rows=sum(len(r[0]) for r in data),gain=gain,scores=stats,
                                           macro_f1_target_minus_control_ci95=deltas))
        print(name,gain,{arm:round(p["macro_f1"],5) for arm,p in stats.items()},flush=True)


if __name__ == "__main__":
    main()
