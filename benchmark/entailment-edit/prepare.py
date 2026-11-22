"""Fit only the routing scores and prepare public NLI text windows."""

from collections import Counter
import csv
import gzip
from itertools import groupby
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from safetensors.numpy import save_file
from scipy.linalg import solve

from protocol import GROUPS, PAIRS, ROLES, digest

HERE = Path(__file__).resolve().parent
CONT = HERE.parent
sys.path.insert(0, str(CONT / "doc-calibration"))
import calibrate as baseline

ARCHIVE = CONT / "legal-context/calibration-banks.zip"
SOURCE = CONT.parent / "references/official"
ROLE_IDS = {role: num for num, role in enumerate(ROLES)}
PLAN = HERE / "plan.json"


def design(bank, temperature):
    probs = baseline.probabilities(bank.scores, temperature)
    order = np.argsort(-bank.scores, axis=1, kind="stable")
    choices = np.column_stack([np.argmax(bank.experts[:, j:j + 7], axis=1) for j in (0, 7, 22, 29)])
    votes = baseline.EYE[choices].mean(1)
    other = np.concatenate([order[:, :3], choices], axis=1)
    possible = (other[:, :, None] == np.arange(7)).any(1)
    curr = order[:, 0]
    masks = np.column_stack([((curr == a) & possible[:, b]) | ((curr == b) & possible[:, a]) for a, b in PAIRS])
    lens = np.diff(bank.ends)
    position = (np.arange(len(curr)) - np.repeat(bank.ends[:-1], lens)) / np.repeat(lens, lens)
    length = np.repeat(np.log1p(lens), lens)
    margin = np.take_along_axis(probs, order[:, :3], axis=1)
    agrees = (choices == curr[:, None]).astype(np.float32)
    x = np.column_stack([probs, votes, baseline.EYE[curr], baseline.EYE[order[:, 1]],
                         margin[:, 0] - margin[:, 1], margin[:, 1] - margin[:, 2], agrees,
                         masks, position, position ** 2, length]).astype(np.float64)
    return x, masks, possible


def fit(bank):
    temperature = baseline.baseline_scale(bank)
    x, _, possible = design(bank, temperature)
    center, scale = x.mean(0), x.std(0).clip(min=.025)
    x = np.column_stack([(x - center) / scale, np.ones(len(x))])
    lengths = np.diff(bank.ends)
    prior = np.bincount(bank.gold, minlength=7) / len(bank.gold)
    w = np.repeat(1 / np.sqrt(lengths), lengths) * prior[bank.gold] ** -.25
    w /= w.mean()
    recoverable = possible[np.arange(len(x)), bank.gold]
    y = (bank.gold != bank.base) * np.where(recoverable, 1., .4)
    gram = (x.T * w) @ x
    gram.flat[::len(gram) + 1] += len(x) * .08
    coef = solve(gram, (x.T * w) @ y, assume_a="pos")
    return {"center": center, "scale": scale, "coef": coef, "temperature": np.asarray([temperature])}


def route(bank, weights):
    x, masks, _ = design(bank, float(weights["temperature"][0]))
    risk = ((x - weights["center"]) / weights["scale"]) @ weights["coef"][:-1] + weights["coef"][-1]
    chosen = []
    for start, end in zip(bank.ends[:-1], bank.ends[1:]):
        order = np.argsort(-risk[start:end], kind="stable")
        budget = min(24, max(1, int(np.ceil(.075 * len(order)))))
        quotas = (int(.35 * budget), max(1, int(.2 * budget)), max(1, int(.2 * budget)))
        picked = set()
        for j in (1, 2, 0):
            candidates = [int(i) for i in order if masks[start + i, j] and i not in picked]
            picked.update(candidates[:min(quotas[j], budget - len(picked))])
        unfilled = budget - len(picked)
        picked.update([int(i) for i in order if i not in picked][:unfilled])
        chosen.extend(start + i for i in sorted(picked))
    rows = np.asarray(chosen, dtype=np.int32)
    assert np.all(np.diff(rows) > 0)
    return rows, masks


def public_cases(paths, required):
    for path in paths:
        seen = set()
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for key, group in groupby(csv.DictReader(handle), key=lambda row: row["Index"]):
                if key not in required:
                    continue
                if key in seen:
                    raise ValueError(f"Noncontiguous judgment {key}")
                seen.add(key)
                rows = [(r["Text"].strip(), ROLE_IDS[r["Label"]]) for r in group if r["Text"].strip()]
                yield path.stem, key, rows


def clipped(text, length):
    if len(text) <= length:
        return text
    return f"{text[:length // 2]} ... {text[-length // 2:]}"


def prepare():
    started = perf_counter()
    plan = {
        "source": "Pinned anonymous tasksource ModernBERT NLI cross-encoder",
        "route": "Public train-only case-weighted analytic ridge to incumbent residual error. Confident misses and worse class imbalance receive lower weights. Score and donor agreement, probabilities, case positions and hard-pair flags only. Reserve per-case positions for Facts–Issue, between the parties and None–Reasoning before residual risk fill.",
        "budget": "ceil(7.5% of each judgment), minimum 1, maximum 24, with 20/20/35% respective quotas",
        "fit": "192 public official train judgments used upstream as well. No iterative gradient steps.",
        "validation": "Selector 48 complete cases, confirmations 72 and 96 complete cases, previously used by the broader project. Weight prediction controls include zero (current document model), donor unchanged and label-shuffled per-case norm-matched edit.",
        "context": "Source sentence boundaries. One previous and next sentence, explicit target marker, head/tail clipping, seven identical published role definitions for every row. 160 tokens including the hypothesis per input pair.",
        "decision": "Select only on the 48 set, guard accuracy within 0.4pp and None–Reasoning errors on both all sentences and no exact train keys. For official consideration require higher seven-role macro-F1 and fewer pair swaps on each confirm set. Official test excluded from input and offline score processing pending positive confirmation.",
        "case_split": {g: "train" if g == "fit" else "val" for g in GROUPS},
        "constraints": "Run inference only on a single Colab Tesla T4 (all least squares and scoring CPU)",
        "tolerance": .004,
    }
    if PLAN.exists():
        assert baseline.read(PLAN) == plan
    else:
        baseline.write(PLAN, plan)
    banks = {g: baseline.load_bank(ARCHIVE, g) for g in GROUPS}
    key_parts = [set(bank.keys.tolist()) for bank in banks.values()]
    assert sum(map(len, key_parts)) == len(set.union(*key_parts))
    weights = fit(banks["fit"])
    save_file(weights, str(HERE / "router.safetensors"))
    case_group = {str(key): (g, n) for g, bank in banks.items() for n, key in enumerate(bank.keys)}
    routes = {g: route(bank, weights) for g, bank in banks.items()}
    for g, (chosen, _) in routes.items():
        np.save(HERE / f"{g}-route.npy", chosen, allow_pickle=False)
    cases = set()
    counts = Counter()
    pack = HERE / "windows.jsonl.gz"
    stage = pack.with_suffix(".part")
    with stage.open("wb") as target, gzip.GzipFile(fileobj=target, mode="wb", mtime=0, filename="") as out:
        for split, key, values in public_cases([SOURCE / "train.csv", SOURCE / "val.csv"], case_group):
            group, n = case_group[key]
            if split != plan["case_split"][group]:
                raise ValueError(f"Case {key} is in wrong split")
            bank = banks[group]
            first, end = bank.ends[n:n + 2]
            texts, labels = zip(*values)
            np.testing.assert_array_equal(bank.gold[first:end], labels)
            selected = routes[group][0]
            lo, hi = np.searchsorted(selected, [first, end])
            assert hi > lo
            for row in selected[lo:hi]:
                i = row - first
                snippet = [clipped(texts[i - 1], 700) if i else "", clipped(texts[i], 3200),
                           clipped(texts[i + 1], 700) if i + 1 < len(texts) else ""]
                payload = (group, int(row), snippet)
                out.write((json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
            counts[group] += int(hi - lo)
            cases.add(key)
    assert cases == case_group.keys() and all(counts[g] == len(routes[g][0]) for g in GROUPS)
    stage.replace(pack)
    receipt = {
        "sha256": {name: digest(HERE / name) for name in ("plan.json", "router.safetensors", "windows.jsonl.gz")},
        "input_archive_hashes_by_group": {g: bank.signature for g, bank in banks.items()},
        "cohorts": {g: {"cases": len(bank.keys), "total_rows": len(bank.base), "routed_rows": counts[g],
                         "nli_examples": counts[g] * 7, "base_wrong_in_route": int((bank.base[rows] != bank.gold[rows]).sum()),
                         "route_hard_pair_overlap": masks[rows].sum(0).tolist(),
                         "route_gold_role_counts": np.bincount(bank.gold[rows], minlength=7).tolist(),
                         "route_sha256": digest(HERE / f"{g}-route.npy")}
                    for g, bank in banks.items() for rows, masks in (routes[g],)},
        "window_pack_bytes": pack.stat().st_size,
        "wall_seconds": round(perf_counter() - started, 2),
    }
    baseline.write(HERE / "preparation.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    prepare()
