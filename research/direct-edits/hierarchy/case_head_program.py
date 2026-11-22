"""Program LegalSeg's native linear emitter using whole-judgment context summaries."""

import argparse
import csv
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file
from scipy.linalg import cho_factor, cho_solve
from scipy.special import softmax

from native_memory import HERE, confusion, confusion_docs, decode, dump, file_hash, metric
from scale_audit import collect_rows, interval, measures, tally
from scale_program import Corpus, FOLDER, compiler, initialize, save_np


OUTPUT = HERE / "case-head-program"
VAL_CSV = HERE.parents[1] / "references/official/val.csv"
FEATURES = ("log_length", "court_vs_none", "court_shift", "neighbor_agreement", "uncertainty")
WIDTH = 201
DIM = WIDTH * (1 + len(FEATURES))
METHODS = ("static", "conditioned_bias", "conditioned_head", "shuffled_head")


def register():
    OUTPUT.mkdir(exist_ok=True)
    dest = OUTPUT / "plan.json"
    scale = initialize()
    if not dest.exists():
        exclude = set(scale["validation"])
        for path, key in (
            (FOLDER / "quadratic/plan.json", "cases"),
            (HERE / "chain-program/plan.json", "validation"),
            (HERE / "uniform-retention-prospective.json", "ids"),
            (HERE.parent / "shared-confirmation.json", "extended_validation_judgment_ids"),
        ):
            exclude.update(json.loads(path.read_text())[key])
        with VAL_CSV.open(encoding="utf-8-sig", newline="") as stream:
            available = {row["Index"] for row in csv.DictReader(stream)}
        salt = "attoflow-case-specific-native-head-v1|"
        cases = sorted(available - exclude, key=lambda key: hashlib.sha256((salt + key).encode()).digest())[:96]
        assert len(cases) == 96
        dump(dest, dict(
            hypothesis="A document's role mix, regional court language and evidence continuity can change which native hidden features deserve weight. The same 7-by-200 emitter receives a different small affine write for each judgment.",
            features=list(FEATURES), fit_prefixes=[960, 2400], class_power=.28, ridges=[.03, .30],
            gains=[.25, .60, 1.25], conditional_penalty_multiplier=4.,
            retention=scale["retention_patch"], parent_plan=file_hash(FOLDER / "plan.json"),
            training_roles_source=file_hash(FOLDER / "freeze.json"), code_sha256=file_hash(Path(__file__)),
            source_csv=file_hash(VAL_CSV), validation=cases, gradient_steps=0,
            controls="Unchanged retention-edited head. Identical fitted static head. Case summaries linked only to native intercepts. Equal-capacity head fitted with case summaries shuffled among similarly long judgments while retaining each case's own log length.",
            criterion="Fit on whole training judgments. Select magnitude and ridge on the existing 192 distinct training-calibration cases with at most .003 absolute accuracy loss. At 2400 fit cases, require the full conditioned head to beat all three fitted controls before opening the 96 reserved validation cases. Validation may have been used by unrelated work. This method does not load official test.",
            prefix_batch_size=48, official_test_evaluated=False))
    plan = json.loads(dest.read_text())
    assert plan["parent_plan"] == file_hash(FOLDER / "plan.json")
    assert plan["source_csv"] == file_hash(VAL_CSV)
    assert plan["training_roles_source"] == file_hash(FOLDER / "freeze.json")
    return scale, plan


def summarize_matrix(H, prior):
    logits = H @ prior[:-1] + prior[-1]
    p = softmax(logits / 2.2, axis=1)
    average = p.mean(0)
    court = p[:,5] + p[:,6]
    n = len(H)
    quarter = max(1, n // 4)
    continuity = np.sum(p[:-1] * p[1:], axis=1).mean() if n > 1 else np.sum(p * p)
    entropy = -np.mean(np.log(np.maximum(p, 1.e-12)) * p) * 7 / np.log(7)
    signal = np.array([np.log1p(n) / 3 - 1.8, 3 * (average[5] + average[6] - average[0]),
                       4 * (court[-quarter:].mean() - court[:quarter].mean()),
                       4 * (continuity - .22), 3 * (entropy - .6)])
    return np.r_[1., signal.clip(-2, 2)], p


def add_chunk(result, corpus, ids, class_weight, block):
    gathered = []
    onehot = np.eye(7)
    for case in ids:
        labels, (_, states) = corpus.extract(case)
        summary, posterior = summarize_matrix(states, corpus.prior)
        x = np.column_stack((states, np.ones(len(states))))
        target = (onehot[labels] - posterior) * 4.4
        x_weighted = x * class_weight[labels, None]
        gathered.append((len(labels), summary, x_weighted.T @ x, x_weighted.T @ target))
    by_length = sorted(range(len(gathered)), key=lambda i: (gathered[i][0], ids[i]))
    random = np.random.default_rng(36811 + block)
    partners = {}
    for low in range(0, len(by_length), 12):
        group = np.asarray(by_length[low:low + 12])
        group = random.permutation(group)
        partners.update(zip(group, np.roll(group, 1)))
    for i, (count, signal, gram, rhs) in enumerate(gathered):
        swapped = gathered[partners[i]][1].copy()
        swapped[1] = signal[1]
        result["rows"] += count
        for j, view in enumerate((signal, swapped)):
            result["G"][j] += np.kron(np.outer(view, view), gram)
            result["B"][j] += (view[:, None, None] * rhs).reshape(DIM, 7)


def empty():
    return {"G":np.zeros((2, DIM, DIM)), "B":np.zeros((2, DIM,7)), "rows":np.zeros((), dtype=np.int64)}


def solve_writes(stat, prior, plan, ridge, method):
    start = np.arange(WIDTH)
    if method == "static":
        slots = start
    elif method == "conditioned_bias":
        slots = np.r_[start, np.arange(2 * WIDTH - 1, DIM, WIDTH)]
    else:
        slots = np.arange(DIM)
    j = int(method == "shuffled_head")
    matrix = stat["G"][j][np.ix_(slots, slots)]
    ridge_scale = ridge * np.mean(stat["G"][0].diagonal()[:WIDTH])
    diagonal = np.where(slots >= WIDTH, plan["conditional_penalty_multiplier"], 1.)
    diagonal = np.where(slots % WIDTH == WIDTH - 1, .08 * diagonal, diagonal)
    matrix.flat[::len(matrix) + 1] += ridge_scale * diagonal
    correction = np.zeros((DIM,7))
    correction[slots] = cho_solve(cho_factor(matrix, check_finite=False), stat["B"][j, slots], check_finite=False)
    return correction.reshape(-1, WIDTH, 7)


def replay(correction, gain, prior, states, features, offsets):
    out = np.empty((len(states), 7), dtype=np.float32)
    for summary, a, b in zip(features, offsets, offsets[1:]):
        written = (prior + gain * np.einsum("j,jhk->hk", summary, correction)).astype(np.float32)
        out[a:b] = states[a:b] @ written[:-1] + written[-1]
    return out


def study(stat, prior, plan, X, y, ends, tr):
    run = compiler()
    features = [summarize_matrix(X[a:b], prior)[0] for a,b in zip(ends, ends[1:])]
    original = run(X @ prior[:-1] + prior[-1], ends, tr)
    base = metric(confusion(y, original))
    weights = {}
    settings = {}
    for kind in METHODS:
        top = (base["macro_f1"], base["accuracy"])
        settings[kind] = dict(macro_f1=top[0], accuracy=top[1], ridge=None, gain=0.)
        weights[kind] = np.zeros((len(FEATURES) + 1, WIDTH, 7), np.float32)
        for alpha in plan["ridges"]:
            delta = solve_writes(stat, prior, plan, alpha, kind)
            for gain in plan["gains"]:
                scores = replay(delta, gain, prior, X, features, ends)
                performance = metric(confusion(y, run(scores, ends, tr)))
                actual = performance["macro_f1"], performance["accuracy"]
                if actual > top and actual[1] >= base["accuracy"] - .003:
                    top = actual
                    settings[kind] = dict(macro_f1=actual[0], accuracy=actual[1], ridge=alpha, gain=gain)
                    weights[kind] = (gain * delta).astype(np.float32)
    return base, settings, weights


def fit(scale, plan, corpus):
    with np.load(FOLDER / "train-calibration.npz", allow_pickle=False) as file:
        X, y, ends = (file[k] for k in ("retention", "y", "ends"))
        assert file["keys"].tolist() == scale["calibration"]
    support = np.asarray(json.loads((FOLDER / "freeze.json").read_text())["support"])
    class_weight = (support / support.sum()) ** -plan["class_power"]
    stats = empty()
    width = plan["prefix_batch_size"]
    plan_hash = file_hash(OUTPUT / "plan.json")
    finished = 0
    for high in plan["fit_prefixes"]:
        snapshot = OUTPUT / f"stats-{high}.npz"
        if snapshot.exists():
            with np.load(snapshot, allow_pickle=False) as saved:
                assert saved["program"].item() == plan_hash
                stats = {key:saved[key] for key in stats}
        else:
            for low in range(finished, high, width):
                ids = scale["fit"][low:low + width]
                add_chunk(stats, corpus, ids, class_weight, low)
                if (low + width) % 480 == 0:
                    print("Case-specific readout", low + width, int(stats["rows"]), flush=True)
            save_np(snapshot, program=plan_hash, **stats)
        finished = high
        baseline, settings, weights = study(stats, corpus.prior, plan, X, y, ends, corpus.base.transitions)
        path = OUTPUT / f"train-calibration-{high}.json"
        dump(path, dict(fit_cases=high, fit_rows=int(stats["rows"]), calibration_cases=len(ends) - 1,
                        calibration_rows=len(y), baseline=baseline, settings=settings, gradient_steps=0))
        print("Calibrated", high, {k:round(v["macro_f1"], 5) for k,v in settings.items()}, flush=True)
    if settings["conditioned_head"]["macro_f1"] <= max(v["macro_f1"] for k,v in settings.items() if k != "conditioned_head"):
        dump(OUTPUT / "no-advance.json", dict(reason="No improvement over strongest same-data matched write",
                                             selection=file_hash(path), validation_opened=False))
        return
    path = OUTPUT / "head-generator.safetensors"
    before = corpus.prior.astype(np.float32)
    tmp = path.with_suffix(".part")
    save_file({"emitter_context." + k:v for k,v in weights.items()} | {"emitter_unchanged":before}, str(tmp))
    tmp.replace(path)
    dump(OUTPUT / "freeze.json", dict(
        fit_cases=scale["fit"][:high], rows_fit=int(stats["rows"]), selected=settings,
        layer_shapes={k:list(v.shape) for k,v in weights.items()}, recipe=file_hash(OUTPUT / "plan.json"),
        generator_sha256=file_hash(path), native_tensors=["emitter.hidden2tag.weight", "emitter.hidden2tag.bias"],
        gradient_steps=0))


def evaluate(plan, corpus):
    from safetensors.numpy import load_file

    frozen = json.loads((OUTPUT / "freeze.json").read_text())
    assert frozen["recipe"] == file_hash(OUTPUT / "plan.json")
    delta = OUTPUT / "head-generator.safetensors"
    assert frozen["generator_sha256"] == file_hash(delta)
    params = load_file(str(delta))
    ids = plan["validation"]
    rows = collect_rows(VAL_CSV, ids)
    states = [corpus.other.contextualize(corpus.base.words(rows[k][0])) for k in ids]
    ends = np.r_[0, np.cumsum([len(x) for x in states])]
    y = np.asarray([label for key in ids for label in rows[key][1]])
    context = [summarize_matrix(X, corpus.prior)[0] for X in states]
    t = corpus.other.transitions
    X = np.concatenate(states)
    run = compiler()
    preds = {"identity":run(X @ corpus.prior[:-1] + corpus.prior[-1], ends, t)}
    tensor = {}
    for name in METHODS:
        correction = params["emitter_context." + name]
        logits = replay(correction, 1., corpus.prior, X, context, ends)
        preds[name] = run(logits, ends, t)
        if name == "conditioned_head":
            for key, sig in zip(ids, context):
                physical = (corpus.prior + np.einsum("j,jhk->hk", sig, correction)).astype(np.float32)
                w, b = corpus.other.weight.copy(), corpus.other.bias.copy()
                w[3:] = physical[:-1].T
                b[3:] = physical[-1]
                tensor[key + ".emitter.hidden2tag.weight"] = w
                tensor[key + ".emitter.hidden2tag.bias"] = b
    file = OUTPUT / "reserved-case-native-weights.safetensors"
    tmp = file.with_suffix(".part")
    save_file(tensor, str(tmp))
    tmp.replace(file)
    # The saved per-case native tensors produce the same labels as the generated runtime coefficients.
    installed = load_file(str(file))
    spot = ids[0]
    model = corpus.other
    model.weight[:] = installed[spot + ".emitter.hidden2tag.weight"]
    model.bias[:] = installed[spot + ".emitter.hidden2tag.bias"]
    x = model.contextualize(model.words(rows[spot][0]))
    np.testing.assert_array_equal(decode(model.logits(x), ends[:2], t), preds["conditioned_head"][:ends[1]])

    conf = {name:confusion_docs(y,p,ends) for name,p in preds.items()}
    span = {name:np.stack([tally(y[a:b],p[a:b]) for a,b in zip(ends,ends[1:])]) for name,p in preds.items()}
    draw = np.random.default_rng(18819).integers(len(ids), size=(2500, len(ids)))
    deltas = {}
    primary = "conditioned_head"
    for name in (*METHODS, "identity"):
        if name == primary:
            continue
        gaps = [metric(conf[primary][k].sum(0))["macro_f1"] - metric(conf[name][k].sum(0))["macro_f1"] for k in draw]
        deltas[name] = dict(macro_f1_ci95=np.quantile(gaps, [.025,.975]).tolist(),
                            exact_span_f1_ci95=interval(span[primary],span[name]))
    rank = {name:dict(**metric(c.sum(0)), spans=measures(span[name])) for name,c in conf.items()}
    save_np(OUTPUT / "reserved96-predictions.npz", keys=ids, ends=ends, gold=y, **preds)
    dump(OUTPUT / "reserved96.json", dict(cases=ids, rows=len(y), scores=rank, paired=deltas,
         exact_native_parity_rows=int(ends[1]), case_weight_file_sha256=file_hash(file),
         freeze=file_hash(OUTPUT / "freeze.json"), official_test_evaluated=False))
    print("Case-specific reserved", len(ids), len(y), {k:round(v["macro_f1"],5) for k,v in rank.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("reserve","fit","evaluate","all"))
    args = parser.parse_args()
    scale, plan = register()
    if args.phase == "reserve":
        return
    corpus = Corpus(scale)
    if args.phase in ("fit","all") and not (OUTPUT / "freeze.json").exists():
        fit(scale, plan, corpus)
    if args.phase in ("evaluate","all") and (OUTPUT / "freeze.json").exists():
        evaluate(plan, corpus)


if __name__ == "__main__":
    main()
