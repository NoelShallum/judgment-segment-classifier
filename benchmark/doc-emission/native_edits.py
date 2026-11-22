"""Apply one-shot native hierarchy emitter and CRF edits to supplied sentences."""

import argparse
import csv
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
from time import monotonic

import numpy as np
from safetensors.numpy import load_file, save_file
from scipy.special import expit, softmax

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import predict_pairwise_judgment as pair
import sentence_phrase_sequence as sequence
import sentence_referral_utility_write as referral
import discourse_span_edit
import embeddinggemma_routed_write as graph_input
from hier_model import CRF, EMITTER, HierModel, LABELS
from program_trial import TO_ID, save_json
from sentence_document_graph_write import DONOR, OUT, file_hash, torch

REVIEWER = ROOT / "live-nov22/bert-t4"
ROUND2 = ROOT / "live-nov22/reviewer-round2"
ROUND_PRED = ROUND2 / "verified/query-results"
PREVIOUS = ROOT / "live-nov22/hierarchy/scale/words"
CSV = ROOT / "references/official"
PARENT = sequence.phrase.dense.PARENT
PLAN = HERE / "plan.json"
FIT_SHAPES = ("pair-margin", "pair-posterior", "all-role", "pair-shuffled")
RAW = ("none_to_reasoning", "reasoning_to_none")
PAIR = (0, 5)
BASE_CUE = discourse_span_edit.cue_features
BASE_EDGE = graph_input.transition_scores
VECTORIZER = sequence.phrase.dense.VECTORIZER


def read(path):
    return json.loads(path.read_text())


def load_rows(split, ids):
    by_key = defaultdict(list)
    with (CSV / (split + ".csv")).open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = row["Index"]
            if key in ids and row["Text"].strip():
                by_key[key].append((row["Text"].strip(), TO_ID[row["Label"]]))
    return by_key


def text_digest(text):
    h = hashlib.sha256()
    for sentence in text:
        h.update(" ".join(sentence.casefold().split()).encode())
        h.update(b"\0")
    return h.hexdigest()


def recipe():
    old = read(REVIEWER / "protocol.json")
    recent = read(ROUND2 / "round2-plan.json")
    groups = {s: old["groups"][s] for s in ("fit", "select", "confirm")}
    groups["fresh"] = recent["groups"]["fresh"]
    text_sha = old["case_text_sha256"] | recent["case_text_sha256"]
    union = [k for cases in groups.values() for k in cases]
    assert len(set(union)) == len(union)
    assert len({text_sha[k] for k in union}) == len(union)
    source_files = [
        DONOR / "hier.safetensors",
        PARENT,
        OUT / "sentence-pairwise-repair.safetensors",
        OUT / "sentence-phrase-sequence.safetensors",
        OUT / "sentence-phrase-readout-1.safetensors",
        OUT / "sentence-phrase-readout-lexical.safetensors",
        ROOT / "results/sentence-phrase-sequence-frozen.json",
        ROOT / "results/sentence-phrase-readout-frozen.json",
        ROOT / "results/sentence-pairwise-repair-frozen.json",
        REVIEWER / "protocol.json",
        ROUND2 / "round2-plan.json",
    ]
    source_sha = {str(p.relative_to(ROOT)): file_hash(p) for p in source_files}
    for split, expected in old["csv_sha256"].items():
        assert split in ("train", "val")
        actual = file_hash(CSV / (split + ".csv"))
        assert actual == expected
        source_sha[f"references/official/{split}.csv"] = actual
    info = {
        "source_sha256": source_sha,
        "groups": groups,
        "text_sha256": {k: text_sha[k] for k in union},
        "arms": {
            "pair-margin": "Closed-form rank-one emitter update of two role rows. Fit a signed native-margin residual on errors and downweight easy rows while retaining other roles.",
            "pair-posterior": "Closed-form rank-one emitter update of two role rows. Fit signed class posteriors near the incumbent pair boundary.",
            "all-role": "Closed-form seven-row regression toward all incorrect roles, weighting uncommon labels.",
            "pair-shuffled": "Matched-norm label permutation control with fixed position and model predictions.",
            "crf": "Closed-form train label versus incumbent co-occurrence ratio in deployed CRF transition tensor, limited to entries touching None and Reasoning.",
            "joint": "Compose the strongest nonzero two-row edit with the None–Reasoning CRF edit.",
        },
        "fit": "384 whole official training judgments. Fit against native activations and archived pre-review labels.",
        "selection": "48 disjoint validation judgments. Rank pooled seven-role macro-F1, then fewer pair errors, then accuracy on the exact pre-review pipeline. Identity eligible in promotion decision.",
        "confirmation": "72 and 96 case-disjoint validation judgments. Both were observed in previous project experiments. Score native pre-review and incumbent reviewer on its original route with predictions fixed on all of those positions. The edited system does not recalculate the BERT route.",
        "ridge": [.12, .6],
        "emitter_gains": [.55, 1.35],
        "transition_gains": [.3, .75],
        "permutation_seed": 2026112341,
        "gradient_steps": 0,
        "device": "CPU",
        "official_test_accessed_in_this_experiment": False,
    }
    if PLAN.exists():
        assert read(PLAN) == info
    else:
        save_json(PLAN, info)
    return info


def stored_case(key, group):
    folder = ROUND2 / "new-cases" if group == "fresh" else REVIEWER / "cases"
    with np.load(folder / (key + ".npz"), allow_pickle=False) as f:
        return {k: f[k] for k in ("gold", "case_sha256", "base", "positions")}


def stash(path, **data):
    tmp = path.with_suffix(".part")
    with tmp.open("wb") as f:
        np.savez(f, **data)
    tmp.replace(path)


def reuse_text_features(fixed_vocab, fixed_cfg):
    """Cache text-only transforms within a single case during weight comparisons."""
    cue = lru_cache(maxsize=2)(BASE_CUE)

    def memo_cues(texts):
        return cue(tuple(texts))

    @lru_cache(maxsize=1)
    def edge(text, offsets):
        return BASE_EDGE(text, offsets, fixed_vocab, fixed_cfg)

    def memo_edges(text, offsets, vocabulary, cfg):
        assert vocabulary is fixed_vocab and cfg is fixed_cfg
        return edge(text, tuple(offsets))

    class MemoVectorizer:
        @lru_cache(maxsize=1)
        def lookup(self, texts):
            return VECTORIZER.transform(texts)

        def transform(self, texts):
            return self.lookup(tuple(texts))

    discourse_span_edit.cue_features = memo_cues
    sequence.phrase.dense.cue_features = memo_cues
    graph_input.transition_scores = memo_edges
    sequence.phrase.dense.VECTORIZER = MemoVectorizer()


class Pipeline:
    def __init__(self, *, fit=False, memo=False):
        torch.set_num_threads(1)
        self.fit = fit
        self.model, self.cfg, self.vocab, self.memory, self.vectors, self.lex_weights, *_ = (
            sequence.phrase.load_predictor("target phrases")
        )
        self.w0 = self.model.weight.copy()
        self.b0 = self.model.bias.copy()
        self.t0 = self.cfg[CRF].copy()
        self.parent = pair.predictor("direct")
        self.rule = read(ROOT / "results" / (sequence.TAG + "-frozen.json"))["rules"]["spans"]
        self.engine = sequence.load_decoder()
        self.sequence_weights = load_file(str(OUT / (sequence.TAG + ".safetensors")))
        if memo:
            reuse_text_features(self.vocab, self.cfg)

    def words(self, key, tokens):
        words_path = PREVIOUS / (key + ".npy")
        if words_path.exists():
            words = np.load(words_path, allow_pickle=False)
            assert words.shape == (len(tokens), 200)
            return torch.from_numpy(words)
        return self.model.words(tokens)

    def states(self, key, tokens):
        states_path = ROOT / "cache/known-sentence-pair-edit" / (key + ".npy")
        if self.fit and not (PREVIOUS / (key + ".npy")).exists() and states_path.exists():
            return np.load(states_path, allow_pickle=False)
        return self.model.contextualize(self.words(key, tokens))

    def features(self, key, text):
        # The archived reference entrypoint assembles the same known-sentence input.
        item = sequence.phrase.dense.document(key, [(i, t, 0) for i, t in enumerate(text)])
        tokens = [item["text"][a:b].strip() for a, b in item["reference_units"]]
        assert tokens == list(text)
        states = self.states(key, tokens)
        assert states.shape == (len(text), 200) and np.isfinite(states).all()
        lex = sequence.phrase.sparse(self.vectors, [sequence.phrase.canonical(t) for t in tokens]) @ self.lex_weights
        return item, states, lex

    def predict(self, item, h, lex, *, include_original=False):
        k, roles, y, anchor, feat = sequence.phrase.dense.dense_features(
            item, h, self.model, self.cfg, self.vocab, self.memory
        )
        ends = np.asarray([0, len(roles)], dtype=np.int64)
        bank = ([k], ends, roles, y, anchor, feat)
        base, unary = sequence.source_unary(bank, lex)
        mode, gain, stay, guard = self.rule
        if mode != "none":
            edge = sequence.edge_arrays(bank, lex, mode, self.sequence_weights)
            base = sequence.decode(self.engine, ends, unary, base, edge, gain, stay, guard)
        baseline = self.parent(bank, lex, unary, base)
        confidence = softmax(unary, axis=1)
        route = referral.route(ends, 1. - confidence[np.arange(len(base)), baseline], .2)
        return (baseline, route, y) if include_original else (baseline, route)

    def install_delta(self, change):
        weight, bias, tr = change
        self.model.weight[:] = self.w0 + weight
        self.model.bias[:] = self.b0 + bias
        self.cfg[CRF][:] = self.t0 + tr

    def install_absolute(self, change):
        weight, bias, tr = change
        self.model.weight[:] = weight
        self.model.bias[:] = bias
        self.cfg[CRF][:] = tr


def subset_fit(args):
    rows, keys, spec = args
    p = Pipeline(fit=True)
    out = []
    for key in keys:
        archive = stored_case(key, "fit")
        texts, gold = zip(*rows[key])
        assert text_digest(texts) == spec["text_sha256"][key] == archive["case_sha256"].item()
        np.testing.assert_array_equal(gold, archive["gold"])
        states = p.states(key, texts)
        assert states.shape == (len(texts), 200) and np.isfinite(states).all()
        out.append((key, states, archive["gold"], archive["base"]))
    return out


def prepare_fit(plan):
    path = HERE / "fit.npz"
    if not path.exists():
        ids = plan["groups"]["fit"]
        rows = load_rows("train", set(ids))
        args = [({k: rows[k] for k in ids[j::2]}, ids[j::2], plan) for j in range(2)]
        with ProcessPoolExecutor(2) as pool:
            data = dict((k, (h, y, b)) for group in pool.map(subset_fit, args) for k, h, y, b in group)
        values = [data[k] for k in ids]
        stash(path, ids=ids, ends=np.r_[0, np.cumsum([len(x[1]) for x in values])],
              states=np.concatenate([x[0] for x in values]), gold=np.concatenate([x[1] for x in values]),
              base=np.concatenate([x[2] for x in values]), plan=file_hash(PLAN))
    with np.load(path, allow_pickle=False) as f:
        assert f["ids"].tolist() == plan["groups"]["fit"]
        assert f["plan"].item() == file_hash(PLAN)
        return tuple(f[k] for k in ("states", "gold", "base", "ends"))


def weighted_design(states, mask):
    x = states[mask]
    center = x.mean(0, dtype=np.float64)
    scale = 1 / np.maximum(x.std(0, dtype=np.float64), .035)
    z = np.column_stack((((x - center) * scale).astype(np.float32), np.ones(len(x), np.float32)))
    return z, center, scale


def solve(x, y, weight, ridge, center, scale):
    u = x * weight.astype(np.float32)[:, None]
    gram = (u.T @ x).astype(np.float64)
    d = ridge * gram.diagonal().mean()
    gram.flat[::len(gram) + 1] += d
    gram[-1, -1] -= .95 * d
    rhs = (u.T @ y).astype(np.float64)
    q = np.linalg.solve(gram, rhs)
    a = q[:-1] * scale[:, None]
    b = q[-1] - center @ a
    return a.T.astype(np.float32), b.astype(np.float32)


def pair_targets(raw, gold, base, mode):
    # Target the margin on the deployed pre-review error distribution.
    s = np.where(gold == PAIR[1], 1., -1.)
    margin = raw[:, PAIR[1]] - raw[:, PAIR[0]]
    pair_gold = np.isin(gold, PAIR)
    pair_base = np.isin(base, PAIR)
    misses = pair_gold & pair_base & (base != gold)
    protect = pair_gold & pair_base & (base == gold)
    if mode in ("pair-margin", "pair-shuffled"):
        target = np.where(misses, s * np.clip(1. - s * margin, .35, 4.5), 0.)
    else:
        target = np.where(pair_gold & pair_base, (s - (2 * expit(margin / 1.8) - 1.)) * 1.8, 0.)
    support = np.bincount(gold, minlength=7)
    mass = np.where(misses, 3., np.where(protect, 1., .55))
    mass *= (support[gold] / len(gold)) ** -.2
    return target[:, None].astype(np.float32), mass


def transition_residual(y, base, ends, model):
    # Match label co-occurrence after accounting for overall role frequency.
    used = np.ones(len(y) - 1, bool)
    used[ends[1:-1] - 1] = False
    def counts(z):
        obs = np.bincount(7 * z[:-1][used] + z[1:][used], minlength=49).reshape(7, 7) + 24.
        independent = obs.sum(0)[None, :] * obs.sum(1)[:, None] / obs.sum()
        return np.log(obs / independent)
    rt = np.clip(counts(y) - counts(base), -1.25, 1.25)
    interested = np.isin(np.arange(7), PAIR)
    rt[~(interested[:, None] | interested[None, :])] = 0
    delta = np.zeros_like(model.transitions)
    delta[3:, 3:] = rt
    return delta


def build(plan):
    paths = HERE / "all-deltas.safetensors"
    if paths.exists():
        return load_file(str(paths))
    h, y, base, ends = prepare_fit(plan)
    model = HierModel(DONOR)
    logits = model.logits(h)
    keys = np.arange(len(y))
    rng = np.random.default_rng(plan["permutation_seed"])
    shuffled = rng.permutation(y)
    x, center, scale = weighted_design(h, keys)
    result = {
        "identity.weight": np.zeros_like(model.weight),
        "identity.bias": np.zeros_like(model.bias),
        "identity.crf": np.zeros_like(model.transitions),
    }
    for mode, ridge in product(FIT_SHAPES, plan["ridge"]):
        w = np.zeros_like(model.weight)
        b = np.zeros_like(model.bias)
        if mode == "all-role":
            support = np.bincount(y, minlength=7)
            weight = (support[y] / len(y)) ** -.32
            weight *= np.where(base != y, 2., 1.)
            target = ((np.eye(7)[y] - softmax(logits / 2., axis=1)) * 3.).astype(np.float32)
            w[3:], b[3:] = solve(x, target, weight, ridge, center, scale)
        else:
            target, weight = pair_targets(logits, shuffled if mode == "pair-shuffled" else y, base, mode)
            a, c = solve(x, target, weight, ridge, center, scale)
            if mode == "pair-shuffled":
                ref = result[f"pair-margin/{ridge:g}.weight"][3 + PAIR[1]]
                mult = np.linalg.norm(ref) / max(.000001, np.linalg.norm(.5 * a[0]))
                a *= mult
                c *= mult
            for role, sign in zip(PAIR, (-.5, .5)):
                w[3 + role] = sign * a[0]
                b[3 + role] = sign * c[0]
        tag = f"{mode}/{ridge:g}"
        result[tag + ".weight"] = w
        result[tag + ".bias"] = b
    result["transition.crf"] = transition_residual(y.astype(np.int64), base.astype(np.int64), ends, model)
    save_file(result, str(paths))
    print("fit", len(ends) - 1, len(y), file_hash(paths), flush=True)
    return result


def delta(arms, config):
    name, gain, transition = config
    tr = arms["transition.crf"] * transition
    return arms[name + ".weight"] * gain, arms[name + ".bias"] * gain, tr


def shard_score(args):
    keys, rows, spec, group, patches = args
    p = Pipeline(memo=True)
    out = []
    now = monotonic()
    for j, key in enumerate(keys, 1):
        raw = stored_case(key, group)
        texts, gold = zip(*rows[key])
        assert text_digest(texts) == spec["text_sha256"][key] == raw["case_sha256"].item()
        np.testing.assert_array_equal(gold, raw["gold"])
        item, states, lex = p.features(key, texts)
        pred = {}
        for tag, (kind, change) in patches.items():
            if kind == "delta":
                p.install_delta(change)
            else:
                p.install_absolute(change)
            revised, route = p.predict(item, states, lex)
            if tag == "identity":
                np.testing.assert_array_equal(revised, raw["base"], err_msg=f"pre-review parity {key}")
                np.testing.assert_array_equal(route, raw["positions"], err_msg=f"route parity {key}")
            pred[tag] = revised
        out.append((key, raw["gold"], raw["positions"], pred))
        if j == 1 or j % 12 == 0 or j == len(keys):
            print(group, j, "/", len(keys), round(monotonic() - now, 1), flush=True)
    return out


def batch(plan, group, patches, limit=None):
    ids = plan["groups"][group][:limit]
    rows = load_rows("val", set(ids))
    args = [(ids[j::2], {k: rows[k] for k in ids[j::2]}, plan, group, patches) for j in range(2)]
    with ProcessPoolExecutor(2) as pool:
        rows = {key: (y, pos, p) for block in pool.map(shard_score, args) for key, y, pos, p in block}
    ordered = [rows[k] for k in ids]
    ends = np.r_[0, np.cumsum([len(x[0]) for x in ordered])]
    labels = np.concatenate([x[0] for x in ordered])
    position = np.concatenate([a + x[1] for a, x in zip(ends, ordered)])
    return ids, ends, labels, position, {
        tag: np.concatenate([x[2][tag] for x in ordered]) for tag in patches
    }


def matrix(y, p):
    return np.bincount(y.astype(int) * 7 + p, minlength=49).reshape(7, 7)


def statistics(cm):
    hit = np.diagonal(cm, axis1=-2, axis2=-1)
    role = 2 * hit / (cm.sum(-1) + cm.sum(-2)).clip(min=1)
    return role.mean(-1), hit.sum(-1) / cm.sum(axis=(-2, -1)), role


def scores(y, p):
    cm = matrix(y, p)
    f, a, r = statistics(cm)
    n2r, r2n = int(cm[0, 5]), int(cm[5, 0])
    return {
        "macro_f1": float(f), "accuracy": float(a),
        "pair_errors": n2r + r2n, **dict(zip(RAW, (n2r, r2n))),
        "per_role_f1": dict(zip(LABELS, r.tolist())), "confusion": cm.tolist(),
    }


def rank(details):
    return details["macro_f1"], -details["pair_errors"], details["accuracy"]


def tune(plan, arms):
    configs = {"identity": ("identity", 0., 0.)}
    for name, ridge, gain in product(FIT_SHAPES[:-1], plan["ridge"], plan["emitter_gains"]):
        configs[f"{name}/{ridge:g}/{gain:g}"] = (f"{name}/{ridge:g}", gain, 0.)
    for g in plan["transition_gains"]:
        configs[f"crf/{g:g}"] = ("identity", 0., g)
    ids, ends, y, review, predictions = batch(plan, "select", {k: ("delta", delta(arms, v)) for k, v in configs.items()})
    results = {k: scores(y, p) for k, p in predictions.items()}
    chosen = {family: max((k for k in configs if k.startswith(family + "/")), key=lambda k: rank(results[k]))
              for family in (*FIT_SHAPES[:-1], "crf")}
    pair_name = max((chosen[k] for k in FIT_SHAPES[:2]), key=lambda k: rank(results[k]))
    joint = (*configs[pair_name][:2], configs[chosen["crf"]][2])
    ref_margin = configs[chosen["pair-margin"]]
    control = (ref_margin[0].replace("pair-margin", "pair-shuffled"), ref_margin[1], 0.)
    chosen_configs = {"identity": configs["identity"], **{k: configs[v] for k, v in chosen.items()},
                      "matched-shuffled": control, "joint": joint}
    p = Pipeline()
    patch_shas = {}
    for k, cfg in chosen_configs.items():
        w, b, tr = delta(arms, cfg)
        path = HERE / (k + ".safetensors")
        save_file({EMITTER + "weight": p.w0 + w, EMITTER + "bias": p.b0 + b, CRF: p.t0 + tr}, str(path))
        patch_shas[k] = file_hash(path)
    freeze = {
        "plan_sha256": file_hash(PLAN), "deltas_sha256": file_hash(HERE / "all-deltas.safetensors"),
        "selected": {k: list(v) for k, v in chosen_configs.items()}, "parameter_files": patch_shas,
        "selection_metrics": results, "best_vs_identity": max(results, key=lambda k: rank(results[k])),
        "selection_judgments": len(ids), "selection_sentences": len(y),
    }
    save_json(HERE / "freeze.json", freeze)
    print("frozen", {k: [v, round(results[v]["macro_f1"] * 100, 3), results[v]["pair_errors"]] for k, v in chosen.items()},
          "baseline", round(results["identity"]["macro_f1"] * 100, 3), results["identity"]["pair_errors"], flush=True)
    return freeze


def interval(y, ends, base, p):
    old = np.stack([matrix(y[a:b], base[a:b]) for a, b in zip(ends, ends[1:])])
    new = np.stack([matrix(y[a:b], p[a:b]) for a, b in zip(ends, ends[1:])])
    resample = np.random.default_rng(177619).integers(len(old), size=(4000, len(old)))
    fo, ao, _ = statistics(old[resample].sum(1))
    fn, an, _ = statistics(new[resample].sum(1))
    return {"macro_f1": np.quantile(fn - fo, [.025, .975]).tolist(),
            "accuracy": np.quantile(an - ao, [.025, .975]).tolist()}


def report_group(plan, freeze, group):
    state = {}
    for tag, digest in freeze["parameter_files"].items():
        file = HERE / (tag + ".safetensors")
        assert file_hash(file) == digest
        v = load_file(str(file))
        state[tag] = ("absolute", (v[EMITTER + "weight"], v[EMITTER + "bias"], v[CRF]))
    ids, ends, gold, positions, pred = batch(plan, group, state)
    with np.load(ROUND_PRED / (group + "-predictions.npz"), allow_pickle=False) as a:
        assert ids == a["case_ids"].tolist()
        np.testing.assert_array_equal(ends, a["offsets"])
        np.testing.assert_array_equal(gold, a["gold"])
        np.testing.assert_array_equal(positions, a["review"])
        incumbent = a["incumbent"]
    np.testing.assert_array_equal(incumbent[np.setdiff1d(np.arange(len(gold)), positions)],
                                  pred["identity"][np.setdiff1d(np.arange(len(gold)), positions)])
    held_route = {}
    for tag, guess in pred.items():
        z = guess.copy()
        z[positions] = incumbent[positions]
        t = scores(gold, z)
        t.update(interval_vs_incumbent=interval(gold, ends, incumbent, z),
                 fixed_errors=int(np.sum((incumbent != gold) & (z == gold))),
                 new_errors=int(np.sum((incumbent == gold) & (z != gold))),
                 changed=int(np.count_nonzero(z != incumbent)))
        held_route[tag] = t
        assert np.array_equal(z[positions], incumbent[positions])
    stash(HERE / (group + "-predictions.npz"), ids=ids, ends=ends, gold=gold, positions=positions,
          incumbent=incumbent, **pred)
    data = {
        "group": group, "judgments": len(ids), "sentences": len(gold), "fixed_review_rows": len(positions),
        "freeze_sha256": file_hash(HERE / "freeze.json"), "reference_sha256": file_hash(ROUND_PRED / (group + "-predictions.npz")),
        "confirmation_pre_review": {k: scores(gold, p) for k, p in pred.items()}, "fixed_review_route": held_route,
    }
    save_json(HERE / (group + "-results.json"), data)
    print(group, len(ids), len(gold), {k: (round(v["macro_f1"] * 100, 3), v["pair_errors"])
          for k, v in held_route.items()}, flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("op", choices=("prepare", "smoke", "tune", "confirm", "fresh"))
    args = cli.parse_args()
    plan = recipe()
    if args.op == "prepare":
        build(plan)
    elif args.op == "smoke":
        p = Pipeline()
        batch(plan, "select", {"identity": ("absolute", (p.w0, p.b0, p.t0))}, limit=2)
    elif args.op == "tune":
        tune(plan, build(plan))
    else:
        freeze = read(HERE / "freeze.json")
        assert file_hash(PLAN) == freeze["plan_sha256"]
        report_group(plan, freeze, args.op)


if __name__ == "__main__":
    main()
