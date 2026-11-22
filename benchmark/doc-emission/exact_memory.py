"""Write a train-only sparse associative layer keyed by exact sentence context."""

import argparse
from collections import defaultdict
import csv
import hashlib
from io import BytesIO
from itertools import groupby, product
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from safetensors.numpy import load_file, save_file

HERE = Path(__file__).resolve().parent / "exact-memory"
ROOT = HERE.parents[2]
CSV = ROOT / "references/official"
REFERENCE = ROOT / "weight-edit-continuation/reviewer"
WEIGHTS = HERE / "memory.safetensors"
SOURCE = HERE / "source.json"
FREEZE = HERE / "freeze.json"
LABELS = ("None", "Facts", "Issue", "Arguments of Petitioner", "Arguments of Respondent", "Reasoning", "Decision")
TO_ID = {k: i for i, k in enumerate(LABELS)}
PAIR = (0, 5)
WIDTHS = (1, 3, 5, 9)
LADDERS = ((1,), (3,), (5,), (9,), (9, 5, 3), (9, 5, 3, 1))
PRED_GROUPS = ("select", "confirm", "fresh")
TYPES = ("all-seven", "pair-only")
EMPTY = bytes(16)
KEY = np.dtype("V16")


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, item):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(item, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text())


def cases(filename):
    with (CSV / (filename + ".csv")).open(newline="", encoding="utf-8-sig") as raw:
        visited = set()
        for index, group in groupby(csv.DictReader(raw), key=lambda row: row["Index"]):
            if index in visited:
                raise ValueError(f"Noncontiguous judgment {index} in {filename}")
            visited.add(index)
            values = [(row["Text"], TO_ID[row["Label"]]) for row in group if row["Text"].strip()]
            if values:
                texts, roles = zip(*values)
                yield index, texts, np.asarray(roles, dtype=np.int8)


def sig(data, scope):
    return hashlib.blake2b(data, digest_size=16, person=scope).digest()


def sentence_signatures(texts):
    return [sig(" ".join(s.casefold().split()).encode(), b"att-role-sent") for s in texts]


def whole(signatures):
    return sig(b"".join(signatures), b"att-role-doc")


def signature_windows(signatures, width):
    if width == 1:
        return np.asarray(signatures, dtype=KEY)
    radius = width // 2
    padded = [EMPTY] * radius + signatures + [EMPTY] * radius
    namespace = f"att-role-win-{width}".encode()
    return np.asarray([sig(b"".join(padded[i:i + width]), namespace) for i in range(len(signatures))],
                      dtype=KEY)


def byte_matrix(x):
    return np.ascontiguousarray(x).view(np.uint8).reshape(-1, 16)


def packed_keys(x):
    assert x.ndim == 2 and x.shape[1] == 16
    return np.ascontiguousarray(x).view(KEY).reshape(-1)


def build_table(ks, ds, role):
    order = np.lexsort((ds, ks))
    sorted_key, sorted_case, sorted_role = ks[order], ds[order], role[order]
    starts = np.r_[0, np.flatnonzero(sorted_key[1:] != sorted_key[:-1]) + 1]
    distinct = np.r_[0, np.flatnonzero((sorted_key[1:] != sorted_key[:-1]) |
                                      (sorted_case[1:] != sorted_case[:-1])) + 1]
    masks = np.left_shift(np.uint8(1), sorted_role.astype(np.uint8))
    seen = np.bitwise_or.reduceat(masks, distinct)
    unambiguous = (seen & (seen - 1)) == 0
    all_roles = np.argmax(seen[:, None] == (1 << np.arange(7))[None, :], axis=1)
    key_slot = np.cumsum(np.r_[True, sorted_key[1:] != sorted_key[:-1]]) - 1
    selected = distinct[unambiguous]
    target = 7 * key_slot[selected] + all_roles[unambiguous]
    counts = np.bincount(target, minlength=7 * len(starts)).reshape(-1, 7)
    support = np.bincount(key_slot[distinct], minlength=len(starts))
    count_max = max(int(counts.max()), int(support.max()))
    assert count_max <= np.iinfo(np.uint16).max
    return {
        "keys": byte_matrix(sorted_key[starts]),
        "weight": counts.astype(np.uint16),
        "source_judgments": support.astype(np.uint16),
        "source_rows": np.diff(np.r_[starts, len(ks)]).astype(np.uint32),
    }


def build():
    HERE.mkdir(exist_ok=True)
    train_path = CSV / "train.csv"
    val_path = CSV / "val.csv"
    val = {whole(sentence_signatures(text)) for _, text, _ in cases("val")}
    accepted, duplicates, validation_duplicates, all_train = [], [], [], set()
    window_keys = defaultdict(list)
    raw_keys, codes, roles = [], [], []
    for i, (index, text, y) in enumerate(cases("train")):
        word = sentence_signatures(text)
        doc = whole(word)
        if doc in all_train:
            duplicates.append(index)
            continue
        all_train.add(doc)
        if doc in val:
            validation_duplicates.append(index)
            continue
        for width in WIDTHS:
            window_keys[width].append(signature_windows(word, width))
        raw_keys.append(np.asarray([sig(x.encode(), b"att-role-raw") for x in text], dtype=KEY))
        codes.append(np.full(len(y), i, dtype=np.uint16))
        roles.append(y)
        accepted.append(index)
        if (i + 1) % 1000 == 0:
            print("Training judgments scanned", i + 1, flush=True)
    ds, p = np.concatenate(codes), np.concatenate(roles)
    tables = {
        "literal.sentence": byte_matrix(np.unique(np.concatenate(raw_keys))),
        "train.complete_judgments": byte_matrix(np.unique(np.asarray(list(all_train), dtype=KEY))),
    }
    summaries = {}
    for width in WIDTHS:
        table = build_table(np.concatenate(window_keys[width]), ds, p)
        keep = table["weight"].sum(1) > 0
        summaries[str(width)] = {
            "keys_with_votes": int(keep.sum()),
            "ambiguous_after_one_vote_per_judgment": int((~keep).sum()),
        }
        tables.update({f"{width}.{k}": v[keep] for k, v in table.items()})
        print(width, summaries[str(width)], flush=True)
    tmp = WEIGHTS.with_suffix(".part")
    save_file(tables, str(tmp))
    tmp.replace(WEIGHTS)
    write(SOURCE, {
        "train_sha256": digest(train_path), "validation_text_pruning_csv_sha256": digest(val_path),
        "weights_sha256": digest(WEIGHTS), "weights_bytes": WEIGHTS.stat().st_size,
        "widths": WIDTHS, "rows_written": len(ds), "train_judgments": len(accepted),
        "duplicate_normalized_training_texts_omitted": duplicates,
        "training_judgments_identical_to_a_validation_judgment_omitted": validation_duplicates,
        "keys": summaries,
        "method": "Sparse key-to-seven-role associative weights written from labels of the official training corpus.",
        "key_definition": "Case-folding and whitespace normalization preserve punctuation, names, and digits. Centered windows stop at judgment boundaries.",
        "literal_definition": "Raw UTF-8 center sentence byte equality including original casing and whitespace.",
        "support_definition": "One unambiguous label vote per distinct normalized training judgment per key.",
        "gradient_steps": 0, "official_test_accessed_during_weight_write": False,
    })


def table():
    source = read(SOURCE)
    assert digest(WEIGHTS) == source["weights_sha256"]
    p = load_file(str(WEIGHTS))
    for width in WIDTHS:
        p[f"{width}.keys"] = packed_keys(p[f"{width}.keys"])
    for k in ("literal.sentence", "train.complete_judgments"):
        p[k] = packed_keys(p[k])
    return p


def search(sorted_keys, query):
    exact = np.searchsorted(sorted_keys, query, side="left")
    position = exact.clip(max=len(sorted_keys) - 1)
    return position, sorted_keys[position] == query


def features(text, p):
    signatures = sentence_signatures(text)
    duplicate = search(p["train.complete_judgments"], np.asarray([whole(signatures)], dtype=KEY))[1][0]
    literal = search(p["literal.sentence"], np.asarray([sig(t.encode(), b"att-role-raw") for t in text], dtype=KEY))[1]
    support, rows, counts = [], [], []
    for width in WIDTHS:
        given = signature_windows(signatures, width)
        at, valid = search(p[f"{width}.keys"], given)
        counts.append(np.where(valid[:, None], p[f"{width}.weight"][at], 0))
        support.append(np.where(valid, p[f"{width}.source_judgments"][at], 0))
        rows.append(np.where(valid, p[f"{width}.source_rows"][at], 0))
    return (np.stack(counts, axis=1), np.stack(support, axis=1), np.stack(rows, axis=1),
            np.full(len(text), duplicate, dtype=bool), literal)


def archive(cohort):
    path = REFERENCE / (cohort + "-predictions.npz")
    with np.load(path, allow_pickle=False) as f:
        keys, end, gold, old = (f[k] for k in ("case_ids", "offsets", "gold", "incumbent"))
        new = f["all_role" if cohort == "official" else "all-role"]
        return path, keys.tolist(), end, gold, {"incumbent": old, "all-role": new}


def legal_predictions(cohort, keys, ends, gold, official_predictions):
    filename = cohort + "-predictions.npz"
    if cohort == "official":
        if official_predictions is None:
            return None
        payload = official_predictions.read_bytes()
    else:
        source = ROOT / "weight-edit-continuation/legal-context/legal-results.zip"
        if not source.is_file():
            return None
        with ZipFile(source) as z:
            expected = json.loads(z.read("SHA256SUMS.json"))[filename]
            payload = z.read(filename)
            assert hashlib.sha256(payload).hexdigest() == expected
    with np.load(BytesIO(payload), allow_pickle=False) as f:
        np.testing.assert_array_equal(keys, f["case_ids"])
        np.testing.assert_array_equal(ends, f["offsets"])
        np.testing.assert_array_equal(gold, f["gold"])
        return f["doc-semantic"], hashlib.sha256(payload).hexdigest()


def on_cohort(p, cohort):
    source, keys, ends, gold, baselines = archive(cohort)
    my = set(keys)
    groups = {}
    for key, text, y in cases("test" if cohort == "official" else "val"):
        if key in my:
            groups[key] = (y, features(text, p))
    assert len(groups) == len(keys)
    np.testing.assert_array_equal(np.concatenate([groups[k][0] for k in keys]), gold)
    np.testing.assert_array_equal(np.diff(ends), [len(groups[k][0]) for k in keys])
    values = [np.concatenate([groups[k][1][i] for k in keys]) for i in range(5)]
    return keys, ends, gold, baselines, values, digest(source)


def propose(base, values, method, setting):
    widths, votes, consensus = setting
    count, source_docs, source_rows, full_duplicate, literal = values
    pred = base.copy()
    matched = np.zeros(len(base), dtype=np.int8)
    supporting_cases = np.zeros(len(base), dtype=np.uint16)
    same_key_rows = np.zeros(len(base), dtype=np.uint32)
    for width in widths:
        j = WIDTHS.index(width)
        current = count[:, j]
        prob_mass = current.sum(1)
        role = current.argmax(1)
        agreement = current[np.arange(len(role)), role] / prob_mass.clip(min=1)
        eligible = ((matched == 0) & (prob_mass >= votes) & (agreement >= consensus) & ~full_duplicate)
        if method == "pair-only":
            eligible &= np.isin(base, PAIR) & np.isin(role, PAIR)
        pred[eligible] = role[eligible]
        matched[eligible] = width
        supporting_cases[eligible] = source_docs[eligible, j]
        same_key_rows[eligible] = source_rows[eligible, j]
    return pred, matched, supporting_cases, same_key_rows


def cm(y, p):
    return np.bincount(y.astype(np.int64) * 7 + p, minlength=49).reshape(7, 7)


def metric(c):
    hit = np.diagonal(c, axis1=-2, axis2=-1)
    f1 = 2 * hit / (c.sum(-1) + c.sum(-2)).clip(min=1)
    return f1.mean(-1), hit.sum(-1) / c.sum(axis=(-2, -1)).clip(min=1), f1


def stats(y, p):
    c = cm(y, p)
    f, a, roles = metric(c)
    return {"macro_f1": float(f), "accuracy": float(a), "none_to_reasoning": int(c[0, 5]),
            "reasoning_to_none": int(c[5, 0]), "pair_errors": int(c[0, 5] + c[5, 0]),
            "per_role_f1": dict(zip(LABELS, roles.tolist())), "confusion": c.tolist()}


def choose():
    p = table()
    keys, ends, y, baselines, values, source_hash = on_cohort(p, "select")
    incumbent = baselines["all-role"]
    live = stats(y, incumbent)
    best = {}
    for method in TYPES:
        grid = []
        for setting in product(LADDERS, (1, 2, 3), (.80, .95, 1.)):
            pred, mask, *_ = propose(incumbent, values, method, setting)
            z = stats(y, pred)
            grid.append((setting, z["macro_f1"], z["accuracy"], z["pair_errors"], int(np.count_nonzero(pred != incumbent))))
        pick = max(grid, key=lambda row: (row[1], -row[3], row[2], -row[4]))
        best[method] = {"parameters": pick[0], "selected_macro_f1": pick[1], "grid": grid}
        print(method, pick, flush=True)
    assert not set(keys) & set(archive("confirm")[1]) and not set(keys) & set(archive("fresh")[1])
    write(FREEZE, {
        "weights_sha256": digest(WEIGHTS), "fit_receipt_sha256": digest(SOURCE), "selection_cases": keys,
        "selection_rows": len(y), "selection_source_sha256": source_hash,
        "selection": "Tune separately for all-seven and pair-only only on the centered reviewer on 48 case-disjoint validation judgments. Rank macro-F1, pair confusions, accuracy, then number of changes. Exact same selected rules are applied to both incumbents.",
        "selection_baseline": live, "arms": best,
        "primary_arm_for_locked_test": "all-seven on all-role",
        "exposure": "The separate confirmation cohorts were used in earlier validation studies and the exact-context availability diagnostic. The gate concerns repeatability in exploratory validation.",
        "test_opened_for_selection": False,
    })


def interval(y, ends, a, b):
    old = np.stack([cm(y[l:r], a[l:r]) for l, r in zip(ends, ends[1:])])
    now = np.stack([cm(y[l:r], b[l:r]) for l, r in zip(ends, ends[1:])])
    sample = np.random.default_rng(17285901).integers(len(old), size=(4000, len(old)))
    m, k = metric(old[sample].sum(1))[:2], metric(now[sample].sum(1))[:2]
    return {name: np.quantile(s - t, [.025, .975]).tolist()
            for name, s, t in zip(("macro_f1", "accuracy"), k, m)}


def outcomes(y, old, pred, masks, values):
    count, _, _, full_duplicate, literal = values
    accepted, source_cases, source_rows = masks
    out = {"all_rows": stats(y, pred), "paired_changed_labels": int(np.count_nonzero(old != pred)),
           "fixed": int(np.count_nonzero((old != y) & (pred == y))),
           "broken": int(np.count_nonzero((old == y) & (pred != y))),
           "changed_wrong_to_other_wrong": int(np.count_nonzero((old != y) & (pred != y) & (old != pred)))}
    available = count.sum((1, 2)) > 0
    matches = accepted > 0
    slices = {
        "no_matching_written_key_at_any_width": ~available,
        "any_matching_written_key": available,
        "unaccepted_by_selected_rule": ~matches,
        "accepted_memory_key": matches,
        "exact_full_judgment_duplicate": full_duplicate,
        "full_judgment_text_disjoint": ~full_duplicate,
        "accepted_and_raw_center_sentence_literal": matches & literal,
        "accepted_but_raw_center_sentence_differs": matches & ~literal,
        "accepted_key_in_one_distinct_train_judgment": matches & (source_cases == 1),
        "accepted_key_in_multiple_train_judgments": matches & (source_cases > 1),
        "accepted_key_seen_on_multiple_train_rows": matches & (source_rows > 1),
    }
    slices.update({f"accepted_context_{w}_sentences": accepted == w for w in WIDTHS})
    def sub(mask):
        return {"rows": int(mask.sum()), "baseline": stats(y[mask], old[mask]),
                "candidate": stats(y[mask], pred[mask]), "changed": int(np.count_nonzero(mask & (old != pred)))}
    out["slices"] = {name: sub(mask) for name, mask in slices.items() if mask.any()}
    np.testing.assert_array_equal(old[~matches], pred[~matches])
    np.testing.assert_array_equal(old[full_duplicate], pred[full_duplicate])
    return out


def evaluate(cohort, official_predictions=None):
    lock = read(FREEZE)
    assert lock["weights_sha256"] == digest(WEIGHTS) and lock["fit_receipt_sha256"] == digest(SOURCE)
    if cohort == "official":
        decision = read(HERE / "official-lock.json")
        assert decision["freeze_sha256"] == digest(FREEZE)
    p = table()
    keys, ends, y, baselines, values, source_sha = on_cohort(p, cohort)
    semantic = legal_predictions(cohort, keys, ends, y, official_predictions)
    if semantic:
        baselines["doc-semantic"] = semantic[0]
    if cohort != "select":
        assert not set(keys) & set(lock["selection_cases"])
    report = {"cohort": cohort, "cases": len(keys), "rows": len(y), "freeze_sha256": digest(FREEZE),
              "source_sha256": source_sha, "semantic_source_sha256": semantic[1] if semantic else None,
              "baseline": {k: stats(y, v) for k, v in baselines.items()}, "arms": {}}
    arrays = dict(case_ids=keys, offsets=ends, gold=y, **baselines)
    for backbone, method in product(baselines, TYPES):
        old = baselines[backbone]
        pred, *m = propose(old, values, method, lock["arms"][method]["parameters"])
        name = backbone + "/" + method
        result = outcomes(y, old, pred, m, values)
        result["paired_ci95"] = interval(y, ends, old, pred)
        report["arms"][name] = result
        arrays[name] = pred
        arrays[name + ".context_width"] = m[0]
        arrays[name + ".distinct_training_cases"] = m[1]
        print(cohort, name, result["all_rows"]["macro_f1"], result["all_rows"]["pair_errors"],
              "fixed", result["fixed"], "broken", result["broken"], flush=True)
    with (HERE / (cohort + "-predictions.npz")).open("wb") as f:
        np.savez_compressed(f, **arrays)
    write(HERE / (cohort + "-results.json"), report)


def lock_official():
    frozen = read(FREEZE)
    judgments, current, edited, ends, labels = [], [], [], [0], []
    records = {}
    for cohort in ("confirm", "fresh"):
        scores = read(HERE / (cohort + "-results.json"))
        assert scores["freeze_sha256"] == digest(FREEZE)
        baseline = scores["baseline"]["all-role"]
        arm = scores["arms"]["all-role/all-seven"]["all_rows"]
        assert arm["macro_f1"] > baseline["macro_f1"] and arm["pair_errors"] < baseline["pair_errors"]
        path = HERE / (cohort + "-predictions.npz")
        records[cohort] = {"report_sha256": digest(HERE / (cohort + "-results.json")),
                           "predictions_sha256": digest(path)}
        with np.load(path, allow_pickle=False) as data:
            assert not set(judgments) & set(data["case_ids"])
            judgments.extend(data["case_ids"])
            ends.extend(ends[-1] + data["offsets"][1:])
            labels.append(data["gold"])
            current.append(data["all-role"])
            edited.append(data["all-role/all-seven"])
    y, a, b = (np.concatenate(parts) for parts in (labels, current, edited))
    paired = interval(y, np.asarray(ends), a, b)
    assert paired["macro_f1"][0] > 0
    cfg = {"freeze_sha256": digest(FREEZE), "weights_sha256": frozen["weights_sha256"],
           "selection": frozen["selection"], "confirmation": records,
           "pooled_macro_f1_delta_ci95": paired["macro_f1"], "judgments": len(judgments), "sentences": len(y),
           "evaluation": "Apply both selection-frozen associative writes unchanged. Report all rows and train-passage matching slices for old and centered reviewer baselines, and independently locked whole-sentence model predictions when available.",
           "test_used_to_select": False}
    write(HERE / "official-lock.json", cfg)
    print("Locked", len(judgments), len(y), paired["macro_f1"], flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("operation", choices=("build", "choose", "select", "confirm", "fresh", "lock-official", "official"))
    ap.add_argument("--legal-official-predictions", type=Path)
    args = ap.parse_args()
    op = args.operation
    if op == "build":
        build()
    elif op == "choose":
        choose()
    elif op == "lock-official":
        lock_official()
    else:
        evaluate(op, args.legal_official_predictions)


if __name__ == "__main__":
    main()
