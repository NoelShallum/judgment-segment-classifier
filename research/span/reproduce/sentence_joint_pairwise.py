"""Add written role-pair evidence to the joint judgment decoder."""
import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np

import sentence_pairwise_consensus as consensus
import sentence_pairwise_repair as repair
import sentence_phrase_sequence as seq
from sentence_document_graph_write import cm, file_hash, measure, paired, row_summary, save_json

P = seq.phrase
TAG = "sentence-joint-pairwise"
SPLITS = ("select", "confirm", "holdout", "phrase-prospect", "test")
METHODS = ("direct", "consensus", "direct pooled")


def source():
    plan, weights, h = consensus.inputs()
    recipe = repair.load_recipe()
    assert recipe["rules"]["spans"][0] == "learned boundary"
    return plan, weights, h, recipe


def prepare(bank, lex, weights, recipe):
    independent, emission = seq.source_unary(bank, lex)
    mode, gain, stay, guard = recipe["rules"]["spans"]
    links = seq.edge_arrays(bank, lex, mode, recipe["weights"])
    offsets = np.ascontiguousarray(bank[1], dtype=np.int64)
    base = seq.decode(recipe["engine"], offsets, emission, independent, links, gain, stay, guard)
    feature = repair.dense_basis(bank, lex)
    pair = repair.score_pairs(feature, weights, "nonlinear")
    return offsets, independent, emission, links, base, pair


def pooled(delta, offset, base):
    start, end, _ = seq.runs(offset, base)
    length = end - start
    whole = np.add.reduceat(delta.astype(np.float64), start, axis=0) / length[:, None]
    return .5 * (delta + np.repeat(whole, length, axis=0))


def corrections(state, method, cutoff):
    offset, independent, unary, _, base, pair = state
    if method == "consensus":
        power = consensus.reconcile(pair, unary, "uniform")
        return power - power[np.arange(len(base)), base, None]
    # A zero value means the pair readout does not supply enough evidence to
    # change the score. The incumbent stays available everywhere.
    proposal = repair.proposals(pair, base, unary, .27, 4)
    delta = np.maximum(proposal - cutoff, 0.)
    if method == "direct pooled":
        return pooled(delta, offset, base)
    return delta


def apply(state, recipe, params, correction=None):
    method, cutoff, weight, strength, stay = params
    off, independent, unary, edges, base, _ = state
    if weight:
        if correction is None:
            correction = corrections(state, method, cutoff)
        unary = unary + weight * correction
    return seq.decode(recipe["engine"], off, unary, independent, edges, strength, stay, False)


def metrics(offset, gold, pred):
    m, a, *_ = measure(cm(gold, pred))
    spans = seq.f1(seq.shape_metrics(offset, gold, pred)[:3])
    return float(m), float(a), float(spans)


def fit():
    plan, weights, h, recipe = source()
    bank = P.dense.prepare("select")
    assert bank[0].tolist() == plan["select_cases"]
    lex = P.cached_scores("select")
    state = prepare(bank, lex, weights, recipe)
    offset, _, _, _, baseline, _ = state
    gold = bank[2]
    _, gain, stay, guard = recipe["rules"]["spans"]
    assert not guard
    p0 = ("none", 0., 0., gain, stay)
    np.testing.assert_array_equal(apply(state, recipe, p0), baseline)
    incumbent = metrics(offset, gold, baseline)
    choices = {name: (incumbent, p0) for name in ("roles", "spans", "roles no pair", "spans no pair")}
    history = []

    def update(params, correction):
        p = apply(state, recipe, params, correction)
        m, a, f = metrics(offset, gold, p)
        history.append([params, (m, a, f), int(np.count_nonzero(p != baseline))])
        for name, (old, _) in choices.items():
            if params[2] and "no pair" in name:
                continue
            if name.startswith("roles"):
                win = (m, a, f) > old and a >= incumbent[1] - .001
            else:
                win = ((f, m, a) > (old[2], old[0], old[1]) and
                       a >= incumbent[1] - .001 and m >= incumbent[0] - .001)
            if win:
                choices[name] = ((m, a, f), params)

    transitions = tuple(product((.025, .06, .10, .16), (-.16, -.12, -.06, 0., .07)))
    for strength, stay in transitions:
        update(("none", 0., 0., strength, stay), None)
    for method in METHODS:
        thresholds = (0.,) if method == "consensus" else (.04, .12, .20)
        for cutoff in thresholds:
            correction = corrections(state, method, cutoff)
            for weight, (strength, stay) in product((.08, .18, .32, .55, .9, 1.5), transitions):
                update((method, cutoff, weight, strength, stay), correction)
    out = {"source_repair_freeze_sha256": h, "source_sequence_freeze_sha256": file_hash(repair.receipt(seq.TAG)),
           "selection_cases": bank[0].tolist(), "gradient_steps": 0, "baseline": incumbent,
           "profiles": {name: {"choices": p, "selection": value} for name, (value, p) in choices.items()},
           "search": history}
    save_json(repair.receipt(TAG), out)
    print("Frozen", choices, incumbent, flush=True)


def load():
    plan, weights, h, recipe = source()
    src = repair.receipt(TAG)
    frozen = json.loads(src.read_text())
    assert frozen["source_repair_freeze_sha256"] == h
    assert frozen["source_sequence_freeze_sha256"] == file_hash(repair.receipt(seq.TAG))
    return plan, weights, recipe, frozen, file_hash(src)


def predict(text, raw, profile):
    _, weights, recipe, frozen, _ = load()
    params = frozen["profiles"][profile]["choices"]

    def revise(bank, lex, unary, base):
        state = prepare(bank, lex, weights, recipe)
        np.testing.assert_array_equal(state[4], base)
        return apply(state, recipe, params)

    return seq.run_input(text, raw, "spans", revise)


def evaluate(splits):
    origin, weights, recipe, frozen, h = load()
    out = {"freeze_sha256": h, "splits": {}}
    archive = P.RESULT / "known-sentence-numerical-archives.npz"
    meta = P.dense.read("known-sentence-numerical-archives")
    assert file_hash(archive) == meta["score_file_sha256"]
    for s in splits:
        bank = P.dense.prepare(s)
        assert not set(origin["fit_cases"]) & set(bank[0])
        if s != "select":
            assert not set(frozen["selection_cases"]) & set(bank[0])
        lex = P.cached_scores(s)
        state = prepare(bank, lex, weights, recipe)
        baseline = state[4]
        old = P.dense.paired_confusion(bank, baseline)
        before_span = seq.span_tally(bank, baseline)
        before = row_summary(old.sum(0))
        assert before == P.dense.read(seq.TAG)["splits"][s]["arms"]["spans"]["labels"]
        rec = out["splits"][s] = {"cases": len(bank[0]), "baseline": before, "baseline_span": seq.span_info(before_span), "profiles": {}}
        if s == "test":
            with np.load(archive, allow_pickle=False) as rows:
                np.testing.assert_array_equal(rows["cases"], bank[0])
                author = rows[meta["best_by_macro_f1"]]
            np.testing.assert_array_equal(author.sum(2), old.sum(2))
            rec["aligned_author"] = row_summary(author.sum(0))
        counts, spans = {}, {}
        for name, cfg in frozen["profiles"].items():
            p = apply(state, recipe, cfg["choices"])
            c = counts[name] = P.dense.paired_confusion(bank, p)
            t = spans[name] = seq.span_tally(bank, p)
            report = rec["profiles"][name] = {
                "labels": row_summary(c.sum(0)), "spans": seq.span_info(t), "changed": int(np.count_nonzero(p != baseline)),
                "paired_labels": paired(c, old), "paired_spans": seq.span_intervals(t, before_span)}
            if s == "test":
                report["paired_labels_to_aligned_author"] = paired(c, author)
            if s == "select":
                assert metrics(state[0], bank[2], p) == tuple(cfg["selection"])
        for target in ("roles", "spans"):
            control = target + " no pair"
            rec["profiles"][target]["paired_labels_to_no_pair"] = paired(counts[target], counts[control])
            rec["profiles"][target]["paired_spans_to_no_pair"] = seq.span_intervals(spans[target], spans[control])
        save_json(P.RESULT / (TAG + ".json"), out)
        print(s, {name: (r["labels"]["macro_f1"], r["labels"]["accuracy"], r["spans"]["exact_labeled_span_f1"])
                  for name, r in rec["profiles"].items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("fit", "evaluate", "predict"))
    parser.add_argument("splits", nargs="*")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--profile", choices=("roles", "spans"), default="spans")
    args = parser.parse_args()
    if args.action == "fit":
        fit()
    elif args.action == "predict":
        if args.input is None:
            parser.error("predict requires --input")
        text = args.input.read_bytes().decode("utf-8")
        print(json.dumps(predict(text if args.raw else json.loads(text), args.raw, args.profile), ensure_ascii=False))
    else:
        evaluate(args.splits or SPLITS)
