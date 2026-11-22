"""Pool judgment-level measurements and replay real CPU text inputs."""

import argparse
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

import study as d
from study import a, base, cues

HERE = Path(__file__).resolve().parent


def read_predictions(name):
    path = HERE / (name + "-predictions.npz")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z}


def pooled():
    k = ("confirm", "fresh")
    z = [read_predictions(name) for name in k]
    keys = np.concatenate([v["case_ids"] for v in z])
    assert len(keys) == len(set(keys)) == 168
    starts = np.r_[z[0]["offsets"], z[0]["offsets"][-1] + z[1]["offsets"][1:]]
    y = np.concatenate([v["gold"] for v in z])
    novel = np.concatenate([v["novel"] for v in z])
    incumbent = np.concatenate([v["zero"] for v in z])
    bank = SimpleNamespace(gold=y, ends=starts, novel=novel)
    arm_names = [g for g in z[0] if g.startswith(d.ROLE_MODES) or g == "zero"]
    stats = {}
    for name in arm_names:
        guess = np.concatenate([v[name] for v in z])
        p = base.stats(bank, guess)
        p["paired_all"] = base.interval(bank, incumbent, guess, np.ones(len(y), bool))
        p["paired_novel"] = base.interval(bank, incumbent, guess, novel)
        p["boundary"] = d.span_boundary(y, guess, starts)
        p["changed"] = int(np.count_nonzero(guess != incumbent))
        p["corrected"] = int(np.count_nonzero((guess == y) & (incumbent != y)))
        p["broken"] = int(np.count_nonzero((guess != y) & (incumbent == y)))
        stats[name] = p
    freeze = base.read(d.FREEZE)
    summary = {
        "judgments": len(keys), "sentences": len(y), "no_train_key": int(novel.sum()), "cohorts": k,
        "selected": freeze["selected"], "scores": stats,
        "source_sha256": {name: a.digest(HERE / (name + "-predictions.npz")) for name in k},
        "training_seconds": base.read(HERE / "training.json")["wall_seconds"],
        "fit_and_selection_seconds": freeze["wall_seconds"],
        "validation_wall_seconds": base.read(HERE / "results.json")["wall_seconds"],
        "weights_bytes": {name: (HERE / name).stat().st_size for name in ("association.safetensors", "residual.safetensors")},
    }
    a.write(HERE / "pooled-confirmation.json", summary)
    print("Pooled discourse", {name: (round(v["all"]["macro_f1"] * 100, 3), v["all"]["pair_errors"],
                                      round(v["novel"]["macro_f1"] * 100, 3), v["novel"]["pair_errors"])
                             for name, v in stats.items()}, flush=True)


def replay():
    w, r = d.read_inputs()
    frozen = base.read(d.FREEZE)
    temp = frozen["temperature"]
    selected = frozen["selected"]
    # Audit the chosen configuration and one full candidate from each family with shipped tensors.
    engine = d.decode_loader()
    tests = []
    for group in ("confirm", "fresh"):
        original = base.load_bank(d.ARCHIVE, group)
        bank = read_predictions(group)
        lengths = np.diff(original.ends)
        median = lengths.argsort()[len(lengths) // 2]
        if selected != "zero":
            touched = bank[selected] != bank["zero"]
            altered = np.add.reduceat(touched, original.ends[:-1])
            chosen = np.lexsort((lengths, -altered))[0]
        else:
            chosen = lengths.argsort()[3]
        request = [original.keys[k] for k in dict.fromkeys((median, chosen))]
        pairs = set(request)
        found = {key: text for key, text, _ in a.cases("val") if key in pairs}
        for key in request:
            ix = original.keys.tolist().index(key)
            lo, hi = original.ends[ix:ix + 2]
            row = SimpleNamespace(scores=original.scores[lo:hi], base=original.base[lo:hi],
                                  ends=np.array([0, hi - lo], dtype=np.int64))
            t = perf_counter()
            f = cues.extract(found[key])
            cue_seconds = perf_counter() - t
            s = d.streams(row, f, w, r, temp, "score")
            t = perf_counter()
            labels = {}
            for arm, v in frozen["settings"].items():
                for profile in ("real", "shuffled"):
                    tag = arm if profile == "real" else arm + "-shuffled"
                    guess = d.predict(row, s, temp, v["configuration"], engine, profile)
                    np.testing.assert_array_equal(guess, bank[tag][lo:hi])
                    labels[tag] = int(np.count_nonzero(guess != row.base))
            tests.append({"group": group, "judgment": key, "sentences": int(hi - lo), "label_disagreements": 0,
                          "fixture_seconds": cue_seconds, "decoding_seconds_all_arms": perf_counter() - t,
                          "changed_vs_base": labels})
    a.write(HERE / "native-replay.json", {"device": "CPU", "selected": selected, "cases": tests,
                                         "checksum": {k: a.digest(HERE / k) for k in
                                                      ("association.safetensors", "residual.safetensors", "freeze.json")}})
    print("Replayed native text and score pairs for", len(tests), "judgments", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("pooled", "replay", "all"), nargs="?", default="all")
    phase = p.parse_args().phase
    if phase in ("pooled", "all"):
        pooled()
    if phase in ("replay", "all"):
        replay()
