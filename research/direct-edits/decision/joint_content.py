"""Choose one fixed pooling coefficient in training-only folds and directly write native option rows."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from safetensors.torch import load_file, save_file

from cohort import LABEL_ID, case_rows, incumbent_dir
from compare import scores, stats
from fetch import ROOT, sha
from readout_study import solve, swapped
from weights import arr, savej, source, wtproof

OUT = ROOT / "readout-study/joint-context-cue"
PLAN = ROOT / "content-cue-traincv-plan.json"


def load(split):
    protocol = json.loads((ROOT / "packed-full-plan.json").read_text())
    cids = protocol["fixed_fit_ids"] if split == "train" else protocol["fixed_validation_ids"]
    docs = case_rows(split, set(cids))
    cues, words, y, cases, blocks, rowids = [], [], [], [], [], []
    base = ROOT / "features/ov-GPU-f32/packed-ordered"
    for cid in cids:
        tag = split + "-" + cid
        layout = json.loads((base / "layouts" / (tag + ".json")).read_text())
        where = Path(os.environ.get("GLIDE_CONTENT_ROOT", ROOT / "content-hidden"))
        receipt = json.loads((where / (tag + ".json")).read_text())
        bank = where / (tag + ".npz")
        assert receipt["file"] == sha(bank) and receipt["plan"] == sha(ROOT / "content-plan.json")
        with np.load(base / (tag + ".npz")) as src, np.load(bank) as pooled:
            assert pooled["positions"].tolist() == src["idx"].tolist() == list(range(len(docs[cid])))
            assert layout["sha256_input_ids_cues_targets"] == str(pooled["input_hash"])
            cues.append(src["hidden"])
            words.append(pooled["content_mean"])
        for i, (_, name) in enumerate(docs[cid]):
            y.append(LABEL_ID[name])
            cases.append(cid)
            blocks.append(cid + "-" + str(layout["index_to_slot_group_size_and_index"][str(i)][2]))
            rowids.append(i)
    return np.concatenate(cues), np.concatenate(words), np.asarray(y), np.asarray(cases), np.asarray(blocks), np.asarray(rowids)


def norm_scale(cue, words):
    return float(np.median(np.linalg.norm(cue, axis=1)) / np.median(np.linalg.norm(words, axis=1)))


def pool(cue, words, a, scale):
    return a * cue + (1. - a) * scale * words


def fit():
    cue, words, y, cases, _, _ = load("train")
    source_tensor = source()[0]
    fold_by_case = {c: i % 3 for i, c in enumerate(dict.fromkeys(cases))}
    fold = np.asarray([fold_by_case[c] for c in cases])
    grid, diagnostics = [0., 1., .25, .5, .75], {}
    OUT.mkdir(parents=True, exist_ok=True)
    for a in grid:
        p = np.zeros(len(y), np.int32)
        for k in range(3):
            tr = fold != k
            x = pool(cue, words, a, norm_scale(cue[tr], words[tr]))
            write = arr(solve(x[tr], source_tensor, y[tr]))[:7]
            p[~tr] = (x[~tr] @ write.T).argmax(1)
        diagnostics[str(a)] = stats(y, p, cases)[1]
    a = max(grid, key=lambda k: diagnostics[str(k)]["macro_f1"])
    scale = norm_scale(cue, words)
    x = pool(cue, words, a, scale)
    files = {}
    for name, label in [("roles", y), ("shuffle", np.random.default_rng(221126).permutation(y))]:
        written = solve(x, source_tensor, label)
        path = OUT / (name + "-readout.safetensors")
        save_file({"weight": written}, str(path))
        files[name] = {"sha256": sha(path), "changed": wtproof(source_tensor, written)}
    record = {"plan": sha(PLAN), "fit_rows": len(y), "training_case_fold": fold_by_case,
              "case_disjoint_training_cross_validation": diagnostics,
              "selected_cue_fraction": a, "frozen_content_scale": scale, "files": files}
    savej(OUT / "fit.json", record)
    print("training split OOF", {k: round(v["macro_f1"], 5) for k, v in diagnostics.items()}, "selected", a, flush=True)


def validate():
    record = json.loads((OUT / "fit.json").read_text())
    assert record["plan"] == sha(PLAN)
    cue, words, y, cases, groups, rows = load("val")
    a, scale = record["selected_cue_fraction"], record["frozen_content_scale"]
    x = pool(cue, words, a, scale)
    raw = arr(source()[0])[:7]
    methods = {"native": (x @ raw.T).argmax(1)}
    for name, proof in record["files"].items():
        f = OUT / (name + "-readout.safetensors")
        assert sha(f) == proof["sha256"]
        w = arr(load_file(str(f))["weight"])[:7]
        methods[name] = (x @ w.T).argmax(1)
        if name == "roles":
            permuted = pool(cue, swapped(words, groups), a, scale)
            methods["swap_only_content"] = (permuted @ w.T).argmax(1)
    baseline = json.loads((ROOT / "weights/ov-GPU-f32/packed-ordered/whole16/predictions.json").read_text())
    assert np.array_equal(baseline["cases"], cases) and np.array_equal(baseline["positions"], rows)
    ref = np.asarray(baseline["predictions"]["roles-readout"])
    incumbent = {c: json.loads((incumbent_dir() / (c + ".json")).read_text())["predictions"]["unchanged"] for c in set(cases)}
    inc = np.asarray([incumbent[c][i] for c, i in zip(cases, rows)])
    rng = np.random.default_rng(760230)
    bs = rng.integers(0, len(set(cases)), (4000, len(set(cases))))
    rb = scores(stats(y, ref, cases)[0][bs].sum(1))
    ib = scores(stats(y, inc, cases)[0][bs].sum(1))
    res = {}
    for k, p in methods.items():
        cm, metric = stats(y, p, cases)
        b = scores(cm[bs].sum(1))
        res[k] = {"metrics": metric, "paired_delta_vs_original_cue_95": np.quantile(b - rb, [.025, .975]).tolist(),
                  "paired_delta_vs_incumbent_95": np.quantile(b - ib, [.025, .975]).tolist(),
                  "changes_vs_cue": int((p != ref).sum())}
    savej(OUT / "validation.json", {"training_fit": sha(OUT / "fit.json"), "rows": len(y), "judgments": len(set(cases)), "methods": res})
    print("joint contextual pooling", {k: v["metrics"]["macro_f1"] for k, v in res.items()}, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("mode", choices=["fit", "validation"])
    choice = cli.parse_args()
    (fit if choice.mode == "fit" else validate)()
