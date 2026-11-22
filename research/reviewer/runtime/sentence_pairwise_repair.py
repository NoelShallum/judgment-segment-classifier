"""Write dispute-specific linear decision boundaries over frozen judgment features."""
import argparse
from itertools import combinations, product
import json

import numpy as np
from safetensors.numpy import load_file, save_file
from scipy.linalg import cho_factor, cho_solve

import sentence_phrase_sequence as seq
import sentence_dense_referral_combine as fusion
from hier_model import LABELS
from sentence_document_graph_write import OUT, cm, file_hash, measure, paired, row_summary, save_json

P = seq.phrase
TAG = "sentence-pairwise-repair"
PAIRS = list(combinations(range(7), 2))
SOURCES = ("sentence-phrase-sequence", "sentence-phrase-sequence-referral", P.TAG, P.dense.TAG, fusion.bert.TAG)
FULL = ("select", "confirm", "holdout", "test")
COHORTS = ("confirm", "holdout", "phrase-prospect", "test")
MODES = ("contextual", "nonlinear")


def receipt(tag):
    return P.RESULT / (tag + "-frozen.json")


def dense_basis(bank, lex):
    return P.design(bank, lex, "contextual phrases", 1)


def expand(x, w, mode):
    z = np.clip((x - w["center"]) * w["scale"], -5., 5.)
    if mode == "nonlinear":
        activation = np.tanh(z @ w["projection"] + w["shift"])
        z = np.column_stack((z, activation))
    return np.column_stack((z, np.ones(len(z), dtype=np.float32)))


def train():
    bank = P.dense.prepare("fit")
    lex = P.cached_scores("fit")
    x = dense_basis(bank, lex)
    y = bank[2]
    posterior = bank[4]
    rng = np.random.default_rng(88574)
    w = {
        "center": x.mean(0, dtype=np.float64).astype(np.float32),
        "scale": (1 / np.maximum(x.std(0, dtype=np.float64), .025)).astype(np.float32),
        "projection": (rng.standard_normal((x.shape[1], 128)) / np.sqrt(x.shape[1])).astype(np.float32),
        "shift": rng.uniform(-.8, .8, 128).astype(np.float32),
    }
    counts = np.bincount(y, minlength=7)
    for mode in MODES:
        basis = expand(x, w, mode)
        coeff = []
        for left, right in PAIRS:
            selected = (y == left) | (y == right)
            z = basis[selected]
            labels = y[selected]
            sign = np.where(labels == right, 1., -1.)
            q = posterior[selected]
            # Prefer examples close to this boundary, preserving a background of
            # easy instances and avoiding synthetic confidence labels.
            pair_mass = q[:, left] + q[:, right]
            gap = np.abs(q[:, left] - q[:, right]) / np.maximum(pair_mass, 1.e-6)
            weight = (.25 + .75 * pair_mass) * (.65 + .35 * (1. - gap))
            weight *= (counts[labels] / max(counts[left], counts[right])) ** -.30
            weighted = z * weight.astype(np.float32)[:, None]
            gram = (z.T @ weighted).astype(np.float64)
            gram = (gram + gram.T) * .5
            penalty = .075 * np.trace(gram) / len(gram)
            gram.flat[::len(gram) + 1] += penalty
            gram[-1, -1] -= .98 * penalty
            target = weighted.T @ sign
            estimator = cho_solve(cho_factor(gram), target)
            np.testing.assert_allclose(gram @ estimator, target, rtol=1.e-7, atol=1.e-6)
            coeff.append(estimator.astype(np.float32))
        w["weight_" + mode] = np.ascontiguousarray(np.stack(coeff, axis=1))
    return w, bank[0].tolist(), counts.tolist()


def score_pairs(x, w, mode):
    z = expand(x, w, mode)
    return z @ w["weight_" + mode]


def incumbent(bank, lex, recipe, pre=None):
    base, emissions = seq.source_unary(bank, lex)
    edge_mode, strength, stay, guard = recipe["rules"]["spans"]
    links = seq.edge_arrays(bank, lex, edge_mode, recipe["weights"])
    engine = recipe["engine"]
    offsets = np.ascontiguousarray(bank[1], dtype=np.int64)
    result = {"sequence only": (
        seq.decode(engine, offsets, emissions, base, links, strength, stay, guard),
        emissions, np.zeros(len(base), dtype=bool))}
    if pre is not None:
        bert, routed = pre
        proposal = fusion.readout(bank, "nonlinear", recipe["dense"]["arms"]["nonlinear"])
        hybrid = np.where(routed, bert, P.dense.preserve_issues(bank[3], proposal))
        unary = emissions.copy()
        i = np.arange(len(base))
        unary[i, hybrid] = np.maximum(unary[i, hybrid], unary.max(1) + .005)
        allowed = (hybrid[:, None] == 2) == (np.arange(7)[None, :] == 2)
        allowed[routed] = (hybrid[:, None] == np.arange(7)[None, :])[routed]
        unary[~allowed] = -1.e8
        pred = seq.decode(engine, offsets, unary, hybrid, links, strength, stay, guard)
        # Use unobstructed phrase margins for pair evidence. Preserve referral
        # ownership, and let selection decide if issue membership can be changed.
        supported = emissions.copy()
        supported[i, pred] = np.maximum(supported[i, pred], supported.max(1))
        result["BERT hybrid"] = (pred, supported, routed)
    return result


def proposals(pair_scores, base, unary, margin_scale, top):
    contest = np.full((len(base), 7), -np.inf, dtype=np.float32)
    winners = np.argsort(-unary, axis=1, kind="stable")
    ranks = np.argsort(winners, axis=1)
    i = np.arange(len(base))
    cost = np.maximum(unary[i, base, None] - unary, 0.)
    for col, (a, b) in enumerate(PAIRS):
        origin_a, origin_b = base == a, base == b
        contest[origin_a, b] = pair_scores[origin_a, col]
        contest[origin_b, a] = -pair_scores[origin_b, col]
    contest -= (cost * margin_scale).astype(np.float32)
    contest[ranks >= top] = -np.inf
    return contest


def decide(scores, base, protected, threshold, issue_guard):
    x = scores
    if issue_guard:
        x = scores.copy()
        x[:, 2] = -np.inf
        x[base == 2] = -np.inf
    best = x.argmax(1)
    take = (x[np.arange(len(base)), best] > threshold) & ~protected
    return np.where(take, best, base).astype(np.int8)


def views(split, recipe):
    if split in FULL:
        bank, _, bert, routed = fusion.views(split)
        pre = (bert, routed)
    else:
        bank = P.dense.prepare(split)
        pre = None
    lex = P.cached_scores(split)
    bases = incumbent(bank, lex, recipe, pre)
    for key, (base, _, _) in bases.items():
        if key == "sequence only":
            reference = P.dense.read(seq.TAG)["splits"][split]["arms"]["spans"]["labels"]
        else:
            reference = P.dense.read(seq.TAG + "-referral")["splits"][split]["arms"]["issue-protected hybrid"]["labels"]
        assert row_summary(cm(bank[2], base)) == reference
    return bank, dense_basis(bank, lex), bases


def load_recipe():
    rule = json.loads(receipt(seq.TAG).read_text())
    path = OUT / (seq.TAG + ".safetensors")
    assert file_hash(path) == rule["checkpoint_sha256"]
    return {**rule, "weights": load_file(str(path)), "engine": seq.load_decoder(),
            "dense": P.dense.read(P.dense.TAG + "-frozen")}


def fit():
    w, fit_ids, role_counts = train()
    model = OUT / (TAG + ".safetensors")
    save_file(w, str(model))
    saved = load_file(str(model))
    assert all(np.array_equal(w[k], val) for k, val in saved.items())
    bank, features, bases = views("select", load_recipe())
    assert not set(fit_ids) & set(bank[0])
    options, baseline, grid = {}, {}, {}
    pair_scores = {a: score_pairs(features, saved, a) for a in MODES}
    for name, (base, unary, protected) in bases.items():
        original = measure(cm(bank[2], base))
        best = (float(original[0]), float(original[1]), 0)
        option = None
        baseline[name] = best[:2]
        grid[name] = []
        for mode, scale, top in product(MODES, (.12, .27, .52, .85), (2, 4, 7)):
            contest = proposals(pair_scores[mode], base, unary, scale, top)
            for threshold, guard in product((.02, .10, .20, .34, .48), (False, True)):
                pred = decide(contest, base, protected, threshold, guard)
                m, a, *_ = measure(cm(bank[2], pred))
                statistic = (float(m), float(a), -int(np.count_nonzero(pred != base)))
                args = (mode, scale, top, threshold, guard)
                grid[name].append([args, statistic])
                if a >= original[1] - .001 and statistic > best:
                    best, option = statistic, args
        options[name] = {"choice": option, "selection": best}
    save_json(receipt(TAG), {"sources": {tag: file_hash(receipt(tag)) for tag in SOURCES},
                            "gradient_steps": 0, "pairs": PAIRS, "role_counts": role_counts,
                            "select_cases": bank[0].tolist(), "fit_cases": fit_ids,
                            "checkpoint_sha256": file_hash(model), "baseline": baseline,
                            "arms": options, "grid": grid})
    print("Frozen", options, "baseline", baseline, flush=True)


def metrics(bank, prediction, base):
    g = bank[2]
    changed = prediction != base
    transitions = []
    for a, b in PAIRS:
        mask = changed & (((base == a) & (prediction == b)) | ((base == b) & (prediction == a)))
        transitions.append({"roles": [LABELS[a], LABELS[b]], "changed": int(mask.sum()),
                            "fixed": int(np.count_nonzero(mask & (prediction == g))),
                            "broken": int(np.count_nonzero(mask & (base == g)))})
    old, new = (P.dense.paired_confusion(bank, v) for v in (base, prediction))
    old_spans, new_spans = (seq.span_tally(bank, v) for v in (base, prediction))
    total = old.sum(0)
    errors = [(int(total[a, b] + total[b, a]), LABELS[a], LABELS[b]) for a, b in PAIRS]
    return {"baseline": row_summary(total), "labels": row_summary(new.sum(0)),
            "paired_labels": paired(new, old), "spans": seq.span_info(new_spans),
            "baseline_spans": seq.span_info(old_spans),
            "paired_spans": seq.span_intervals(new_spans, old_spans),
            "changed": int(changed.sum()), "decisions_by_pair": transitions,
            "baseline_confusions_ranked": sorted(errors, reverse=True)}


def evaluate(splits):
    path = receipt(TAG)
    freeze = json.loads(path.read_text())
    assert freeze["sources"] == {tag: file_hash(receipt(tag)) for tag in SOURCES}
    model = OUT / (TAG + ".safetensors")
    assert freeze["checkpoint_sha256"] == file_hash(model)
    w = load_file(str(model))
    recipe = load_recipe()
    out = {"freeze_sha256": file_hash(path), "splits": {}}
    for split in splits:
        bank, x, bases = views(split, recipe)
        assert not set(freeze["fit_cases"]) & set(bank[0])
        if split != "select":
            assert not set(freeze["select_cases"]) & set(bank[0])
        scores = {}
        record = {"cases": len(bank[0]), "arms": {}}
        for name, (base, unary, protected) in bases.items():
            option = freeze["arms"][name]["choice"]
            prediction = base
            if option:
                mode, scale, top, threshold, guard = option
                if mode not in scores:
                    scores[mode] = score_pairs(x, w, mode)
                contest = proposals(scores[mode], base, unary, scale, top)
                prediction = decide(contest, base, protected, threshold, guard)
                np.testing.assert_array_equal(prediction[protected], base[protected])
                if guard:
                    np.testing.assert_array_equal(prediction == 2, base == 2)
            result = metrics(bank, prediction, base)
            record["arms"][name] = result
            if split == "select":
                assert result["labels"]["macro_f1"] == freeze["arms"][name]["selection"][0]
        out["splits"][split] = record
        save_json(P.RESULT / (TAG + ".json"), out)
        print(split, {k: (r["baseline"]["macro_f1"], r["labels"]["macro_f1"],
                          r["baseline"]["accuracy"], r["labels"]["accuracy"], r["changed"])
                      for k, r in record["arms"].items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("fit", "evaluate"))
    parser.add_argument("splits", nargs="*")
    args = parser.parse_args()
    if args.action == "fit":
        fit()
    else:
        evaluate(args.splits or COHORTS)
