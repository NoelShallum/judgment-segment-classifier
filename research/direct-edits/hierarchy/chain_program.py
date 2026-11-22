"""One-shot native LegalSeg readout writes using exact role-chain probabilities."""

import argparse
import csv
import ctypes
import hashlib
from itertools import product
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from safetensors.numpy import load_file, save_file
from scipy.special import softmax

from native_memory import BASE, DONOR, HERE, ROOT, HierModel, TO_ID, confusion, confusion_docs, decode, dump, file_hash, metric, native
from retention_replay import load_arm
from scale_program import Corpus, FOLDER, NAMES, compiler, initialize, save_np, solution

sys.path.insert(0, str(HERE.parent))
from span_comparison import interval as span_interval, measures as span_measures, tally as span_tally

OUTPUT = HERE / "chain-program"
VAL_SOURCE = ROOT / "references/official/val.csv"
MODES = ("local", "chain")


def preregister():
    OUTPUT.mkdir(exist_ok=True)
    target = OUTPUT / "plan.json"
    source = initialize()
    if not target.exists():
        blocked = set(source["validation"])
        for path, key in (
            (FOLDER / "quadratic/plan.json", "cases"),
            (HERE / "uniform-retention-prospective.json", "ids"),
            (HERE.parent / "shared-confirmation.json", "extended_validation_judgment_ids"),
        ):
            blocked.update(json.loads(path.read_text())[key])
        for split in ("select", "confirm", "holdout"):
            with np.load(ROOT / f"cache/sentence-recurrent-inverse/{split}.npz") as cache:
                blocked.update(cache["keys"].tolist())
        with VAL_SOURCE.open(newline="", encoding="utf-8-sig") as stream:
            all_ids = {row["Index"] for row in csv.DictReader(stream)}
        salt = "attoflow-legalseg-structured-residual-v1|"
        cases = sorted(all_ids - blocked, key=lambda key: hashlib.sha256((salt + key).encode()).digest())[:96]
        assert len(cases) == 96
        dump(target, dict(
            source_plan=file_hash(FOLDER / "plan.json"), validation_csv=file_hash(VAL_SOURCE),
            donor=file_hash(BASE), posterior_source=file_hash(HERE / "chain_posterior.cpp"),
            program_source=file_hash(Path(__file__)), salt=salt, validation=cases, prefixes=[960, 4792],
            temperature=2.2, class_powers=[0.0, 0.28, 0.6], ridges=[0.03, 0.3],
            gains=[0.5, 1.25, 2.5], transition_gains=[0.125, 0.35, 0.75],
            transition_pseudocount=20, transition_log_ratio_clip=1.0, max_calibration_accuracy_loss=.003,
            hypothesis="The marginal of each human rhetorical role under the native full-document CRF is the relevant baseline for a one-shot emitter correction. An independently optimized, equally sized write to the same seven native output rows uses isolated softmax residuals as the control.",
            transition_arm="Adjust the native CRF's 49 active role-pair entries by a shrunk reference-to-expected log count ratio. START, END and inactive rows are retained. Compare joint posterior+transition, isolated-softmax+transition and transition-only writes.",
            selection="Rank each family on the existing 192 training-calibration judgments by seven-class macro-F1 then accuracy subject to a .003 maximum accuracy loss from the matching state/CRF baseline. Select the highest posterior arm across native and previously retention-edited states. Identity, independent local write and transition-only remain active comparators.",
            confirmation="After the 4792 prefix, refit the frozen choices on all 4984 if the posterior arm beats its primary matched controls on train calibration. Assess the new 96 untouched-by-method validation judgments with case-paired bootstrap and exact spans at fixed sentence boundaries. Earlier unrelated validation exposure may exist. No official test is loaded.",
            gradient_steps=0))
    plan = json.loads(target.read_text())
    assert plan["source_plan"] == file_hash(FOLDER / "plan.json")
    assert plan["posterior_source"] == file_hash(HERE / "chain_posterior.cpp")
    assert plan["validation_csv"] == file_hash(VAL_SOURCE)
    return source, plan


def load_posterior():
    source = HERE / "chain_posterior.cpp"
    binary = OUTPUT / f"chain-posterior-{file_hash(source)[:12]}.so"
    if not binary.exists():
        tmp = binary.with_suffix(".part")
        subprocess.run(["g++", "-std=c++17", "-O3", "-shared", "-fPIC", str(source), "-o", str(tmp)], check=True)
        tmp.replace(binary)
    fun = ctypes.CDLL(str(binary)).role_chain_posterior
    matrix = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    fun.argtypes = [matrix, matrix, ctypes.c_size_t, ctypes.c_double, matrix, matrix]
    fun.restype = None

    def posterior(logits, transitions, temperature):
        result = np.empty(logits.shape, dtype=np.float64)
        counts = np.zeros((7, 7), dtype=np.float64)
        assert logits.shape[1] == 7 and transitions.shape == (10, 10)
        fun(np.ascontiguousarray(logits, dtype=np.float64), np.ascontiguousarray(transitions, dtype=np.float64),
            len(logits), temperature, result, counts)
        return result, counts

    return posterior


def brute_force(logits, tr, temperature):
    tr = np.asarray(tr, dtype=np.float64)
    paths = np.asarray(list(product(range(7), repeat=len(logits))))
    emissions = logits[np.arange(len(logits)), paths].sum(1)
    edges = tr[paths[:, :-1] + 3, paths[:, 1:] + 3].sum(1)
    mass = softmax((emissions + edges + tr[1, paths[:, 0] + 3] + tr[paths[:, -1] + 3, 2]) / temperature)
    marginal = np.stack([np.bincount(col, weights=mass, minlength=7) for col in paths.T])
    counts = np.bincount((7 * paths[:, :-1] + paths[:, 1:]).ravel(), np.repeat(mass, len(logits) - 1),
                         minlength=49).reshape(7, 7)
    return marginal, counts


def verify(posterior, transitions, temperature):
    rng = np.random.default_rng(286979)
    errs = []
    for n in (1, 2, 3, 4):
        scores = rng.normal(size=(n, 7))
        expected = brute_force(scores, transitions, temperature)
        got = posterior(scores, transitions, temperature)
        for a, b in zip(got, expected):
            np.testing.assert_allclose(a, b, rtol=0, atol=2e-12)
            errs.append(float(np.max(np.abs(a - b))))
    dump(OUTPUT / "posterior-qa.json", dict(max_abs_error=max(errs), case_lengths=[1, 2, 3, 4],
                                           against="Enumeration of every seven-role path, including START and END"))


def empty():
    return dict(row_count=np.zeros(7, np.int64), B=np.zeros((2, 2, 7, 201, 7)),
                observed=np.zeros((7, 7), np.int64), predicted=np.zeros((2, 7, 7)))


def accumulate(corpus, keys, plan, posterior):
    out = empty()
    eye = np.eye(7)
    tr = corpus.base.transitions
    temperature = plan["temperature"]
    for key in keys:
        labels, views = corpus.extract(key)
        out["row_count"] += np.bincount(labels, minlength=7)
        out["observed"] += np.bincount(7 * labels[:-1] + labels[1:], minlength=49).reshape(7, 7)
        for i, states in enumerate(views):
            design = np.column_stack((states, np.ones(len(labels))))
            logits = design @ corpus.prior
            chain, edge_counts = posterior(logits, tr, temperature)
            np.testing.assert_allclose(chain.sum(1), 1, atol=2e-13)
            np.testing.assert_allclose(edge_counts.sum(), len(labels) - 1, atol=1e-8)
            out["predicted"][i] += edge_counts
            for j, probs in enumerate((softmax(logits / temperature, axis=1), chain)):
                target = (eye[labels] - probs) * (2 * temperature)
                for role in np.unique(labels):
                    rows = labels == role
                    out["B"][i, j, role] += design[rows].T @ target[rows]
    return out


def gram_prefix(n):
    g = np.zeros((2, 7, 201, 201))
    count = np.zeros(7, np.int64)
    for path in sorted((FOLDER / "shards").glob("*.npz")):
        end = int(path.stem.split("-")[1])
        if end > n:
            break
        with np.load(path) as part:
            g += part["G"]
            count += part["count"]
    if n == 4984:
        with np.load(FOLDER / "cal-stats.npz") as part:
            g += part["G"]
            count += part["count"]
    return g, count


def transition_write(transitions, stats, state, strength, plan):
    old = transitions.copy()
    smooth = plan["transition_pseudocount"]
    limit = plan["transition_log_ratio_clip"]
    logratio = np.log((stats["observed"] + smooth) / (stats["predicted"][state] + smooth)).clip(-limit, limit)
    old[3:, 3:] += strength * plan["temperature"] * (logratio - logratio.mean())
    return old


def train_candidates(stats, gram, reference, transitions, plan, calibration, run):
    output = {}
    all_scores = {}
    for i, name in enumerate(NAMES):
        states, y, ends = calibration[name], calibration["y"], calibration["ends"]
        baseline = metric(confusion(y, run(states @ reference[:-1] + reference[-1], ends, transitions)))
        guard = baseline["accuracy"] - plan["max_calibration_accuracy_loss"]
        grid = {}
        best = {}
        acts = product(plan["class_powers"], plan["ridges"], plan["gains"])
        identity = {"macro_f1": baseline["macro_f1"], "accuracy": baseline["accuracy"], "head": None,
                    "transition_gain": 0., "metrics": baseline}
        readouts = [(None, reference.astype(np.float32), "identity")]
        for setting in acts:
            for j, mode in enumerate(MODES):
                head = solution(gram[i], stats["B"][i, j], stats["row_count"], *setting, reference)
                readouts.append((setting, head, mode))
        for setting, head, mode in readouts:
            z = states @ head[:-1] + head[-1]
            for strength in (0., *plan["transition_gains"]):
                tr = transition_write(transitions, stats, i, strength, plan)
                score = metric(confusion(y, run(z, ends, tr)))
                tag = mode + ("-plus-transitions" if strength else "")
                if tag not in best:
                    best[tag] = identity
                key = (score["macro_f1"], score["accuracy"])
                grid[f"{mode}/{setting}/{strength}"] = dict(macro_f1=key[0], accuracy=key[1])
                have = best[tag]
                if key > (have["macro_f1"], have["accuracy"]) and key[1] >= guard:
                    best[tag] = dict(macro_f1=key[0], accuracy=key[1], head=setting,
                                     transition_gain=strength, metrics=score)
        output[name] = best
        all_scores[name] = grid
    return output, all_scores


def winner(choices):
    ranked = []
    for state, found in choices.items():
        controls = max(found[key]["macro_f1"] for key in ("identity", "identity-plus-transitions",
                                                          "local", "local-plus-transitions"))
        for method in ("chain", "chain-plus-transitions"):
            result = found[method]
            if result["head"] is not None and result["macro_f1"] > controls:
                ranked.append((result["macro_f1"], result["accuracy"], state, method))
    return max(ranked)[-2:] if ranked else None


def install(corpus, stats, gram, plan, staged, choice):
    count = stats["row_count"]
    state, primary = choice
    position = NAMES.index(state)
    model = corpus.base if state == "donor" else corpus.other
    comparisons = staged[state]
    made = {}
    for method in (primary, "identity", "local", "local-plus-transitions", "identity-plus-transitions"):
        config = comparisons[method]
        if config["head"] is None:
            actual = corpus.prior.astype(np.float32)
        else:
            mode = method.split("-")[0]
            actual = solution(gram[position], stats["B"][position, MODES.index(mode)], count,
                              *config["head"], corpus.prior)
        weight, bias = model.weight.copy(), model.bias.copy()
        weight[3:] = actual[:-1].T
        bias[3:] = actual[-1]
        crf = transition_write(model.transitions, stats, position, config["transition_gain"], plan)
        copies = {"emitter.hidden2tag.weight": weight, "emitter.hidden2tag.bias": bias, "crf.transitions": crf}
        if state == "retention":
            copies.update(load_file(str(HERE / "uniform-retention.safetensors")))
        dest = OUTPUT / f"{method}.safetensors"
        tmp = dest.with_suffix(".part")
        save_file(copies, str(tmp))
        tmp.replace(dest)
        made[method] = {"file": dest.name, "sha256": file_hash(dest), "settings": config}
        donor = native(corpus.base)
        changes = {}
        for key, arr in copies.items():
            before = donor[key].numpy()
            difference = arr.astype(np.float64) - before
            changes[key] = {"shape": list(arr.shape), "changed_values": int(np.count_nonzero(difference)),
                            "delta_l2": float(np.linalg.norm(difference)), "final_dtype": str(arr.dtype)}
        made[method]["changes"] = changes
    dump(OUTPUT / "freeze.json", dict(chosen=primary, state_space=state, cases_fit=4984, rows_fit=int(count.sum()),
                                     plan_sha256=file_hash(OUTPUT / "plan.json"),
                                     program_source=file_hash(Path(__file__)), checkpoints=made, gradient_steps=0))


def train(source, plan, corpus):
    posterior = load_posterior()
    verify(posterior, corpus.base.transitions, plan["temperature"])
    with np.load(FOLDER / "train-calibration.npz", allow_pickle=False) as file:
        calibration = {key: file[key] for key in ("keys", "ends", "y", *NAMES)}
        assert calibration["keys"].tolist() == source["calibration"]
    run = compiler()
    stats = empty()
    for lo in range(0, len(source["fit"]), 96):
        hi = min(lo + 96, len(source["fit"]))
        keys = source["fit"][lo:hi]
        target = OUTPUT / f"stats-{lo:04d}-{hi:04d}.npz"
        if not target.exists():
            block = accumulate(corpus, keys, plan, posterior)
            save_np(target, keys=keys, plan_sha256=file_hash(OUTPUT / "plan.json"), **block)
        with np.load(target, allow_pickle=False) as block:
            assert block["keys"].tolist() == keys
            assert block["plan_sha256"].item() == file_hash(OUTPUT / "plan.json")
            for key in stats:
                stats[key] += block[key]
        if hi % 480 == 0 or hi == len(source["fit"]):
            print("Structured sufficient statistics", hi, "judgments", int(stats["row_count"].sum()), "sentences", flush=True)
        if hi in plan["prefixes"]:
            gram, count = gram_prefix(hi)
            np.testing.assert_array_equal(stats["row_count"], count)
            chosen, grid = train_candidates(stats, gram, corpus.prior, corpus.base.transitions, plan, calibration, run)
            take = winner(chosen)
            dump(OUTPUT / f"train-calibration-{hi}.json", dict(cases_fit=hi, rows_fit=int(count.sum()),
                cases_calibration=len(calibration["keys"]), rows_calibration=len(calibration["y"]),
                arms=chosen, grid=grid, advancing=take, official_test_evaluated=False))
            print("Training comparison", hi, take, {k: {m: round(v["macro_f1"], 5) for m,v in methods.items()}
                                                    for k,methods in chosen.items()}, flush=True)
    if take is None:
        dump(OUTPUT / "no-advance.json", dict(reason="Neither chain arm improved its strongest matched training-calibration control.",
                                             plan_sha256=file_hash(OUTPUT / "plan.json"),
                                             program_source=file_hash(Path(__file__)), validation_opened=False))
        return
    part = accumulate(corpus, source["calibration"], plan, posterior)
    for key in stats:
        stats[key] += part[key]
    gram, count = gram_prefix(4984)
    np.testing.assert_array_equal(stats["row_count"], count)
    install(corpus, stats, gram, plan, chosen, take)


def evaluate(plan, corpus):
    freeze = json.loads((OUTPUT / "freeze.json").read_text())
    assert freeze["plan_sha256"] == file_hash(OUTPUT / "plan.json")
    ids = plan["validation"]
    rows = {key: ([], []) for key in ids}
    with VAL_SOURCE.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            text = row["Text"].strip()
            if text and row["Index"] in rows:
                rows[row["Index"]][0].append(text)
                rows[row["Index"]][1].append(TO_ID[row["Label"]])
    model = corpus.base if freeze["state_space"] == "donor" else corpus.other
    y = np.concatenate([rows[key][1] for key in ids])
    ends = np.r_[0, np.cumsum([len(rows[key][0]) for key in ids])]
    X = np.concatenate([model.contextualize(corpus.base.words(rows[key][0])) for key in ids])
    predictions, confusion_by_doc, span_counts = {}, {}, {}
    run = compiler()
    full = {}
    for name, item in freeze["checkpoints"].items():
        profile = load_arm(f"chain-program/{item['file']}", item["sha256"])
        z = profile.logits(X)
        pred = run(z, ends, profile.transitions)
        # Native-module forwarding and the fast decoder agree on an independent, complete judgment.
        live = profile.logits(profile.contextualize(profile.words(rows[ids[0]][0])))
        np.testing.assert_allclose(live, z[:ends[1]], atol=3e-5)
        np.testing.assert_array_equal(pred[:ends[1]], decode(live, ends[:2], profile.transitions))
        predictions[name] = pred
        confusion_by_doc[name] = confusion_docs(y, pred, ends)
        span_counts[name] = np.stack([span_tally(y[a:b], pred[a:b]) for a,b in zip(ends, ends[1:])])
        full[name] = dict(**metric(confusion_by_doc[name].sum(0)), spans=span_measures(span_counts[name]))
    primary = freeze["chosen"]
    selected = predictions[primary]
    draws = np.random.default_rng(511760).integers(len(ids), size=(3000, len(ids)))
    paired = {}
    for name, pred in predictions.items():
        if name == primary:
            continue
        sampled = []
        for draw in draws:
            a, b = (metric(confusion_by_doc[k][draw].sum(0)) for k in (primary, name))
            sampled.append([a["macro_f1"] - b["macro_f1"], a["accuracy"] - b["accuracy"]])
        changed = selected != pred
        paired[name] = dict(macro_f1_accuracy_ci95=np.quantile(sampled, [.025, .975], axis=0).T.tolist(),
                            exact_span_f1_ci95=span_interval(span_counts[primary], span_counts[name]),
                            corrected=int((changed & (selected == y)).sum()),
                            harmed=int((changed & (pred == y)).sum()))
    save_np(OUTPUT / "reserved96-predictions.npz", keys=ids, ends=ends, gold=y, **predictions)
    dump(OUTPUT / "reserved96.json", dict(cases=ids, rows=len(y), scores=full, paired=paired,
         chosen=primary, freeze_sha256=file_hash(OUTPUT / "freeze.json"), official_test_evaluated=False,
         cpu_native_parity={"judgment": ids[0], "rows": int(ends[1]), "decoded_disagreements": 0}))
    print("Structured reserved", len(ids), len(y), {k:(s["macro_f1"], s["accuracy"]) for k,s in full.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("reserve", "train", "evaluate", "all"))
    args = parser.parse_args()
    source, plan = preregister()
    if args.stage == "reserve":
        return
    corpus = Corpus(source)
    if args.stage in ("train", "all") and not (OUTPUT / "freeze.json").exists():
        train(source, plan, corpus)
    if args.stage in ("evaluate", "all") and (OUTPUT / "freeze.json").exists() and not (OUTPUT / "reserved96.json").exists():
        evaluate(plan, corpus)


if __name__ == "__main__":
    main()
