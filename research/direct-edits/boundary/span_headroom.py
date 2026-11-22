"""Measure whether frozen GLiNER scores can repair existing LegalSeg spans."""
import argparse
import hashlib
import json
import os
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from safetensors.numpy import load_file

from score import KEYS, evaluate, extra, merge_rows, native, readcases, runs


HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
SHORT_LENGTH = 3
ROLE_SCORE_SEED = 906215


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def short(spans):
    return {span for span in spans if span[1] - span[0] <= SHORT_LENGTH}


def counts(gold, offered):
    return np.array([len(gold & offered), len(gold), len(offered)])


def ratio(num, den):
    return np.divide(num, den, where=den > 0, out=np.zeros_like(num, dtype=float))


def pooled(count_arrays):
    hit, gold, predicted = np.sum(count_arrays, axis=0)
    return dict(matches=int(hit), gold=int(gold), predicted=int(predicted),
                recall=float(ratio(hit, gold)), precision=float(ratio(hit, predicted)),
                f1=float(ratio(2 * hit, gold + predicted)))


def span_stats(y, pred):
    truth, found = runs(y), runs(pred)
    return np.stack([
        counts(truth, found),
        counts(short(truth), short(found)),
        counts({s[0] for s in truth if s[0] > 0}, {s[0] for s in found if s[0] > 0}),
    ])


def interior(base, candidate):
    return {span for span in short(runs(candidate))
            if span[0] > 0 and span[1] < len(base)
            and base[span[0]] != span[2]
            and np.all(base[span[0] - 1:span[1] + 1] == base[span[0]])}


def insert(base, additions):
    updated = base.copy()
    for start, stop, label in sorted(additions):
        updated[start:stop] = label
    return updated


def shift_window(before, after):
    start, edge, _ = before
    _, stop, _ = after
    low = max(edge - 2, (start + edge) // 2 + 1)
    high = min(edge + 2, (edge + stop) // 2, stop - 1)
    return low, high


def relocate(base, signal):
    """Move eligible boundaries at most two sentences while preserving the role path."""
    original = sorted(runs(base))
    updated = base.copy()
    for before, after in zip(original, original[1:]):
        _, edge, left_role = before
        _, _, right_role = after
        low, high = shift_window(before, after)
        margin = signal[:, left_role] - signal[:, right_role]
        choices = [(0., 0, edge)]
        for pos in range(low, edge):
            choices.append((-float(margin[pos:edge].sum()), -(edge - pos), pos))
        for pos in range(edge + 1, high + 1):
            choices.append((float(margin[edge:pos].sum()), -(pos - edge), pos))
        _, _, best = max(choices)
        if best < edge:
            updated[best:edge] = right_role
        elif best > edge:
            updated[edge:best] = left_role
    assert [s[2] for s in sorted(runs(updated))] == [s[2] for s in original]
    return updated


def oracle_relocate(base, truth):
    """Find the greatest gold span overlap possible within the same shift windows."""
    original = sorted(runs(base))
    gold = runs(truth)
    windows = [range(lo, hi + 1) for lo, hi in
               (shift_window(left, right) for left, right in zip(original, original[1:]))]
    starts = {0: (0, 0, None)}
    history = []
    for (_, edge, label), options in zip(original, windows + [[len(base)]]):
        ends = {}
        for stop in options:
            ends[stop] = max((hits + ((start, stop, label) in gold), cost - abs(stop - edge), start)
                             for start, (hits, cost, _) in starts.items() if start < stop)
        history.append(ends)
        starts = ends
    updated = base.copy()
    end = len(base)
    for (_, _, label), states in zip(reversed(original), reversed(history)):
        _, _, start = states[end]
        updated[start:end] = label
        end = start
    assert end == 0 and [s[2] for s in sorted(runs(updated))] == [s[2] for s in original]
    assert len(runs(updated) & gold) >= len(runs(base) & gold)
    return updated


def signals(case, weights):
    if weights is None:
        return native(case)
    observations = case["native"].sum(1).astype(float)
    for channel, tensor in enumerate(weights):
        observations += extra(case, tensor, channel)
    return merge_rows(case, observations / 3)


def intervals(method, base, indexes):
    def values(data):
        hit, gold, pred = data[indexes].sum(axis=1).transpose(2, 0, 1)
        return np.stack([ratio(2 * hit, gold + pred), ratio(hit, gold), ratio(hit, pred)], axis=2)

    return np.quantile(values(method) - values(base), [0.025, 0.975], axis=0).transpose(1, 2, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", choices=("four", "sixteen"))
    stage = parser.parse_args().cohort
    report_file = OUT / f"{stage}.json"
    prediction_file = OUT / f"{stage}-predictions.json"
    source = read_json(report_file)
    manifest = read_json(OUT / "freeze-sha256.json")
    for relative, digest in manifest.items():
        assert checksum(HERE / relative) == digest, relative
    assert source["freeze_sha256"] == manifest["results/freeze.json"]
    cases = readcases("val", source["ids"])
    saved = read_json(prediction_file)
    reference = {case: np.array(saved["incumbent"][case]) for case in cases}
    all_predictions = {"incumbent": reference,
                       "oracle_relocate": {k: oracle_relocate(base, cases[k]["y"]) for k, base in reference.items()}}
    diagnostics = {}

    for label in ("native", "edit", "shuffled"):
        weights = None
        if label != "native":
            tensors = load_file(str(OUT / f"{label}-delta.safetensors"))
            weights = [tensors[name] for name in KEYS]
        proposals, possible, recovered = {}, {}, {}
        role_predictions, inserted, oracle, relocated = {}, {}, {}, {}
        for key, case in cases.items():
            truth = short(runs(case["y"]))
            base = reference[key]
            signal = signals(case, weights)
            role_predictions[key] = signal.argmax(1)
            np.testing.assert_array_equal(role_predictions[key], saved[label][key])
            offered = interior(base, role_predictions[key])
            recoverable = offered & truth
            denominator = interior(base, case["y"])
            assert recoverable <= denominator
            proposals[key] = counts(truth, offered)
            possible[key] = len(denominator)
            recovered[key] = len(recoverable)
            inserted[key] = insert(base, offered)
            oracle[key] = insert(base, recoverable)
            relocated[key] = relocate(base, signal)
        all_predictions.update({label: role_predictions, f"{label}_insert": inserted,
                                f"{label}_relocate": relocated, f"{label}_oracle_insert": oracle})
        denom = sum(possible.values())
        diagnostics[label] = dict(proposals=pooled(list(proposals.values())),
                                  eligible_missed_gold=denom, recovered_gold=sum(recovered.values()),
                                  fraction_recoverable=float(ratio(sum(recovered.values()), denom)),
                                  eligible_by_judgment=possible, recovered_by_judgment=recovered)

    n = len(cases)
    ix = np.random.default_rng(ROLE_SCORE_SEED).integers(0, n, (2000, n))
    reference_spans = np.array([span_stats(cases[k]["y"], reference[k]) for k in cases])
    groups = ("all_exact", "short_exact", "unlabeled_boundary")
    evaluated = {}
    for method, prediction in all_predictions.items():
        measures = evaluate(cases, prediction)[0]
        raw = np.array([span_stats(cases[k]["y"], prediction[k]) for k in cases])
        gaps = intervals(raw, reference_spans, ix)
        measures.update({group: pooled(raw[:, i]) for i, group in enumerate(groups)})
        measures["gain_over_incumbent_ci95"] = {
            group: dict(zip(("f1", "recall", "precision"), gaps[i].tolist()))
            for i, group in enumerate(groups)
        }
        evaluated[method] = measures

    write_json(OUT / f"{stage}-span-headroom.json", dict(
        protocol_sha256=checksum(HERE / "span-repair-protocol.json"),
        source_report_sha256=checksum(report_file), source_predictions_sha256=checksum(prediction_file),
        judgments=source["ids"], bootstrap_seed=ROLE_SCORE_SEED, measures=evaluated, short_run_diagnostics=diagnostics,
    ))
    write_json(OUT / f"{stage}-span-repair-predictions.json", {
        method: {key: values.tolist() for key, values in prediction.items()}
        for method, prediction in all_predictions.items()
        if "oracle" not in method and method.endswith(("_insert", "_relocate"))
    })
    for method, values in evaluated.items():
        print(method, "macro F1", round(values["macro_f1"], 5),
              "exact F1 / recall", round(values["all_exact"]["f1"], 5), round(values["all_exact"]["recall"], 5),
              "short F1 / recall", round(values["short_exact"]["f1"], 5), round(values["short_exact"]["recall"], 5))


if __name__ == "__main__":
    main()
