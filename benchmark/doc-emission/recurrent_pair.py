"""Write a targeted low-rank update to the native document LSTM candidate gates."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np
from safetensors.numpy import load_file, save_file

import native_edits as trial
from sentence_recurrent_inverse import inverse_targets

HERE = trial.HERE / "recurrent-pair"
FORWARD, REVERSE = "", "_reverse"
SUFFIXES = (FORWARD, REVERSE)
PARAMETERS = ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0")
FIT_CASES = 192
CELLS_PER_DIRECTION = 24
FIT_RIDGES = (.14, .60)
GAINS = (.40, 1.)
BASIS = ("input", "input-and-recurrent")


def config():
    HERE.mkdir(exist_ok=True)
    main = trial.recipe()
    keys = main["groups"]["fit"][:FIT_CASES]
    spec = {
        "source_plan_sha256": trial.file_hash(trial.PLAN),
        "fit": keys, "selection": main["groups"]["select"],
        "confirmation": {k: main["groups"][k] for k in ("confirm", "fresh")},
        "selected_cells_per_direction": CELLS_PER_DIRECTION,
        "ridges": list(FIT_RIDGES), "gains": list(GAINS), "gradient_steps": 0,
        "method": "Fit candidate-gate preactivation differences at the frozen baseline. Map each target pair logit margin into a native hidden-state direction orthogonal to the other five role scores and to the midpoint of None and Reasoning. Solve candidate preactivations analytically per case and fit physical gate rows with ridge. Verify full native recurrent forward.",
        "pre_review": "Complete production hierarchy and role-sequence pipeline.",
        "confirmation_route": "Fixed previously deployed BERT predictions on the original 20% route.",
        "device": "CPU", "official_test_opened": False,
    }
    target = HERE / "plan.json"
    if target.exists():
        assert trial.read(target) == spec
    else:
        trial.save_json(target, spec)
    return main, spec


def null_directions(model):
    head = model.weight[3:].astype(np.float64)
    n, r = trial.PAIR
    contrast = head[r] - head[n]
    guard = np.vstack((head[[i for i in range(7) if i not in trial.PAIR]], .5 * (head[n] + head[r])))
    width = model.doc.hidden_size
    directions, kept = [], []
    for direction in range(2):
        sl = slice(direction * width, (direction + 1) * width)
        d = contrast[sl]
        idx = np.argsort(-np.abs(d), kind="stable")[:CELLS_PER_DIRECTION]
        guard_subset = guard[:, sl][:, idx]
        z = d[idx] - np.linalg.pinv(guard_subset, rcond=1.e-7) @ (guard_subset @ d[idx])
        result = np.zeros(width, dtype=np.float64)
        result[idx] = .5 * z / (z @ d[idx])
        np.testing.assert_allclose(guard[:, sl] @ result, 0, atol=1.e-9)
        np.testing.assert_allclose(d @ result, .5, atol=1.e-9)
        directions.append(result)
        kept.append(idx)
    return np.concatenate(directions).astype(np.float32), kept


def gather_shard(task):
    keys, data, assignments, spec = task
    p = trial.Pipeline()
    direction, kept = null_directions(p.model)
    out = []
    maxerr = 0.
    for key in keys:
        texts, gold = zip(*data[key])
        g, baseline = assignments[key]
        np.testing.assert_array_equal(gold, g)
        assert trial.text_digest(texts) == spec["text_sha256"][key]
        words = p.words(key, texts).numpy()
        h = p.model.contextualize(trial.torch.from_numpy(words))
        target, mass = trial.pair_targets(p.model.logits(h), g, baseline, "pair-margin")
        desired = np.clip(target * direction, -.12, .12).astype(np.float32)
        a = [None, None]
        for d in range(2):
            x, y, e = inverse_targets(words, h, desired, p.model, d)
            a[d] = (x, y[:, kept[d]], mass[::-1] if d else mass)
            maxerr = max(maxerr, e)
        out.append((key, a))
    return out, maxerr


def native_params(model):
    return {name + suffix: getattr(model.doc, name + suffix).detach().numpy().copy()
            for suffix, name in product(SUFFIXES, PARAMETERS)}


def fit(main, spec):
    source = HERE / "deltas.safetensors"
    if source.exists():
        return load_file(str(source))
    _, gold, base, ends = trial.prepare_fit(main)
    keys = spec["fit"]
    rows = trial.load_rows("train", set(keys))
    assignments = {k: (gold[a:b], base[a:b]) for k, a, b in zip(keys, ends, ends[1:])}
    jobs = [(keys[i::2], {k: rows[k] for k in keys[i::2]}, {k: assignments[k] for k in keys[i::2]}, main)
            for i in range(2)]
    with ProcessPoolExecutor(2) as pool:
        parts = list(pool.map(gather_shard, jobs))
    by_key = {k: v for pack, _ in parts for k, v in pack}
    trial.torch.set_num_threads(1)
    p = trial.HierModel(trial.DONOR)
    _, kept = null_directions(p)
    old = native_params(p)
    dims = p.doc.input_size
    hidden = p.doc.hidden_size
    targets = {key: {} for key in (*BASIS, "shuffled")}
    for d, suffix in enumerate(SUFFIXES):
        entries = [by_key[k][d] for k in keys]
        x, y, m = (np.concatenate([row[i] for row in entries]) for i in range(3))
        positive = np.flatnonzero(np.any(y != 0, axis=1))
        shuffled = y.copy()
        rng = np.random.default_rng(552763 + d)
        shuffled[positive] = shuffled[rng.permutation(positive)]
        active = kept[d] + 2 * hidden
        for name, ridge in product(targets, FIT_RIDGES):
            use = x if name == BASIS[1] else x[:, :dims]
            q, center, scale = trial.weighted_design(use, np.arange(len(use)))
            t = shuffled if name == "shuffled" else y
            v, b = trial.solve(q, t, m, ridge, center, scale)
            if name == "shuffled":
                ref = targets[BASIS[0]][f"{ridge:g}.weight_ih_l0{suffix}"][active]
                factor = np.linalg.norm(ref) / max(.000001, np.linalg.norm(v[:, :dims]))
                v *= factor
                b *= factor
            for param in PARAMETERS:
                tag = param + suffix
                delta = np.zeros_like(old[tag])
                if param == "weight_ih_l0":
                    delta[active] = v[:, :dims]
                elif param == "weight_hh_l0" and name == BASIS[1]:
                    delta[active] = v[:, dims:]
                elif param == "bias_ih_l0":
                    delta[active] = b
                targets[name][f"{ridge:g}.{tag}"] = delta
    patch = {f"{name}/{k}": v for name, bank in targets.items() for k, v in bank.items()}
    save_file(patch, str(source))
    trial.save_json(HERE / "fit-receipt.json", {
        "plan_sha256": trial.file_hash(HERE / "plan.json"),
        "deltas_sha256": trial.file_hash(source),
        "fit_judgments": len(keys), "fit_rows": int(ends[len(keys)]),
        "gate_rows": {s or "forward": (kept[i] + 2 * hidden).tolist() for i, s in enumerate(SUFFIXES)},
        "maximum_reconstructed_hidden_error": max(e for _, e in parts),
    })
    print("gate fit", len(keys), int(ends[len(keys)]), flush=True)
    return patch


def install(model, tensors):
    with trial.torch.no_grad():
        for name, wt in tensors.items():
            getattr(model.doc, name).copy_(trial.torch.from_numpy(wt))


def shard(job):
    keys, rows, main, group, variants = job
    p = trial.Pipeline(memo=True)
    out = []
    for key in keys:
        archive = trial.stored_case(key, group)
        words, gold = zip(*rows[key])
        assert trial.text_digest(words) == main["text_sha256"][key] == archive["case_sha256"].item()
        np.testing.assert_array_equal(gold, archive["gold"])
        item, _, lex = p.features(key, words)
        source = p.words(key, words)
        predictions = {}
        for tag, patch in variants.items():
            install(p.model, patch)
            h = p.model.contextualize(source)
            pred, route = p.predict(item, h, lex)
            if tag == "identity":
                np.testing.assert_array_equal(pred, archive["base"])
                np.testing.assert_array_equal(route, archive["positions"])
            predictions[tag] = pred
        out.append((key, archive["gold"], archive["positions"], predictions))
    return out


def replay(main, group, variants):
    keys = main["groups"][group]
    rows = trial.load_rows("val", set(keys))
    jobs = [(keys[i::2], {k: rows[k] for k in keys[i::2]}, main, group, variants) for i in range(2)]
    with ProcessPoolExecutor(2) as pool:
        out = {k: (y, review, p) for block in pool.map(shard, jobs) for k, y, review, p in block}
    sizes = np.r_[0, np.cumsum([len(out[k][0]) for k in keys])]
    gold = np.concatenate([out[k][0] for k in keys])
    review = np.concatenate([a + out[k][1] for a, k in zip(sizes, keys)])
    predictions = {name: np.concatenate([out[k][2][name] for k in keys]) for name in variants}
    return keys, sizes, gold, review, predictions


def tune(main, spec):
    patches = fit(main, spec)
    p = trial.HierModel(trial.DONOR)
    old = native_params(p)
    candidates = {"identity": old}
    for family, ridge, gain in product(BASIS, FIT_RIDGES, GAINS):
        candidates[f"{family}/{ridge:g}/{gain:g}"] = {
            k: v + gain * patches[f"{family}/{ridge:g}.{k}"] for k, v in old.items()
        }
    ids, _, gold, _, pred = replay(main, "select", candidates)
    scores = {k: trial.scores(gold, z) for k, z in pred.items()}
    selected = {family: max((k for k in candidates if k.startswith(family + "/")),
                            key=lambda k: trial.rank(scores[k])) for family in BASIS}
    _, ridge, gain = selected["input"].split("/")
    control = {k: v + float(gain) * patches[f"shuffled/{ridge}.{k}"] for k, v in old.items()}
    weights = {"identity": old, **{family: candidates[tag] for family, tag in selected.items()}, "shuffled": control}
    meta = {}
    for name, values in weights.items():
        dest = HERE / (name + ".safetensors")
        save_file({"emitter.lstm." + name: tensor for name, tensor in values.items()}, str(dest))
        meta[name] = {"sha256": trial.file_hash(dest),
                      "changed_values": {k: int(np.count_nonzero(v != old[k])) for k, v in values.items()}}
    receipt = {"plan_sha256": trial.file_hash(HERE / "plan.json"), "cases": len(ids), "rows": len(gold),
               "scores": scores, "selected": selected, "patches": meta,
               "best": max(scores, key=lambda k: trial.rank(scores[k]))}
    trial.save_json(HERE / "freeze.json", receipt)
    print("gate selected", selected, {k: round(v["macro_f1"] * 100, 3) for k, v in scores.items()}, flush=True)


def report(main, group):
    frozen = trial.read(HERE / "freeze.json")
    variants = {}
    for name, patch in frozen["patches"].items():
        path = HERE / (name + ".safetensors")
        assert trial.file_hash(path) == patch["sha256"]
        variants[name] = {k.removeprefix("emitter.lstm."): v for k, v in load_file(str(path)).items()}
    ids, ends, gold, review, pred = replay(main, group, variants)
    archive = trial.ROUND_PRED / (group + "-predictions.npz")
    with np.load(archive, allow_pickle=False) as original:
        assert original["case_ids"].tolist() == ids
        for k, v in (("gold", gold), ("offsets", ends), ("review", review)):
            np.testing.assert_array_equal(original[k], v)
        incumbent = original["incumbent"]
    pre = {k: trial.scores(gold, z) for k, z in pred.items()}
    adjusted = {}
    for name, raw in pred.items():
        z = raw.copy()
        z[review] = incumbent[review]
        if name == "identity":
            np.testing.assert_array_equal(incumbent, z)
        adjusted[name] = trial.scores(gold, z)
        adjusted[name]["changes"] = int(np.count_nonzero(z != incumbent))
        adjusted[name]["paired_ci95"] = trial.interval(gold, ends, incumbent, z)
    trial.stash(HERE / (group + "-predictions.npz"), ids=ids, ends=ends, gold=gold, review=review,
                incumbent=incumbent, **pred)
    result = {"group": group, "judgments": len(ids), "sentences": len(gold),
              "archived_review_rows": len(review), "reference_sha256": trial.file_hash(archive),
              "freeze_sha256": trial.file_hash(HERE / "freeze.json"), "pre_review": pre, "fixed_route": adjusted}
    trial.save_json(HERE / (group + "-results.json"), result)
    print(group, {k: (round(v["macro_f1"] * 100, 3), v["pair_errors"]) for k, v in adjusted.items()}, flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("op", choices=("tune", "confirm", "fresh"))
    op = cli.parse_args().op
    plan, spec = config()
    if op == "tune":
        tune(plan, spec)
    else:
        report(plan, op)


if __name__ == "__main__":
    main()
