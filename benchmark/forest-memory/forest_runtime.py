"""Apply a frozen random-partition correction table to contextual role scores."""

import numpy as np


ROLE_COUNT = 7
NONE = 0
REASONING = 5
EYE = np.eye(ROLE_COUNT, dtype=np.float32)


def row_probabilities(scores):
    values = np.exp(np.clip(scores - scores.max(1, keepdims=True), -36., 0.))
    return values / values.sum(1, keepdims=True)


def is_pair(labels):
    return (labels == NONE) | (labels == REASONING)


def winners_and_rivals(scores):
    winner = scores.argmax(1).astype(np.int32)
    candidates = scores.copy()
    candidates[np.arange(len(winner)), winner] = -np.inf
    rival = candidates.argmax(1).astype(np.int32)
    return winner, rival


def context_means(values, radius):
    n = len(values)
    rows = np.arange(n)
    pre = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(values, axis=0, dtype=np.float64)))
    left = np.maximum(rows - radius, 0)
    right = np.minimum(rows + 1 + radius, n)
    return ((pre[rows] - pre[left]) / np.maximum(rows - left, 1)[:, None],
            (pre[right] - pre[rows + 1]) / np.maximum(right - rows - 1, 1)[:, None])


def design(scores, expert, before, offsets, temperature):
    """Build the exact, label-free routing features within each judgment."""
    scores = np.asarray(scores, dtype=np.float32)
    expert = np.asarray(expert, dtype=np.float32)
    winner, rival = winners_and_rivals(scores)
    row = np.arange(len(scores))
    z = (scores - scores.mean(1, keepdims=True)) / temperature
    p = row_probabilities(z)
    rival_gaps = np.column_stack((z[row, winner] - z[row, rival], z[:, REASONING] - z[:, NONE],
                                  p[row, winner], p[row, rival], p[:, REASONING] - p[:, NONE]))
    views, extras, role_margins = [], [], []
    for c in (0, 22):
        for a, factor in ((c, .65), (c + 7, .80)):
            values = expert[:, a:a + ROLE_COUNT] * factor
            q = row_probabilities(values)
            others, runner = winners_and_rivals(values)
            views.append(q)
            extras.extend((EYE[others], EYE[runner]))
            role_margins.append(np.column_stack((q[row, winner] - q[row, rival], q[:, REASONING] - q[:, NONE],
                                                 others == winner, others == rival)))
        extras.append(np.clip(expert[:, c + 14:c + 22], -5., 5.))
    average = np.mean(np.stack(views), axis=0)
    features = np.column_stack((z, p, EYE[winner], EYE[rival], EYE[before], rival_gaps, *views,
                                *extras, *role_margins, *[q - p for q in views]))
    groups = []
    for a, b in zip(offsets[:-1], offsets[1:]):
        count = b - a
        i = np.arange(count, dtype=np.float32)
        pos = (i + .5) / count
        local_p = p[a:b]
        others = average[a:b]
        mix = np.column_stack((local_p, others))
        doc = mix.mean(0)
        curve = np.column_stack((pos, pos ** 2, np.log1p(i), np.log1p(count - i - 1),
                                 np.full(count, np.log1p(count)), pos * (others[:, REASONING] - others[:, NONE])))
        parts = [features[a:b], curve, np.broadcast_to(doc, mix.shape), mix - doc,
                 np.broadcast_to(EYE[winner[a:b]].mean(0), (count, ROLE_COUNT))]
        for r in (1, 5, 19):
            left, right = context_means(mix, r)
            parts.extend((left - mix, right - mix, left - right))
        groups.append(np.column_stack(parts).astype(np.float32))
    return np.concatenate(groups), winner, rival


def projected(x, columns, weights):
    values = np.zeros((len(x), len(columns)), dtype=np.float32)
    for j in range(columns.shape[1]):
        values += x[:, columns[:, j]] * weights[:, j]
    return values


def tree_indices(x, winner, rival, params):
    z = np.clip((x - params["center"]) * params["scale"], -5., 5.)
    indices = []
    depth = params["threshold"].shape[1]
    factors = 1 << np.arange(depth, dtype=np.int32)
    bucket = winner * ROLE_COUNT + rival if int(params["uses_rival"][0]) else winner
    for columns, direction, cutoff in zip(params["columns"], params["direction"], params["threshold"]):
        values = projected(z, columns, direction)
        branch = (values > cutoff).astype(np.int32) @ factors
        indices.append((bucket << depth) + branch)
    return np.stack(indices)


def readout(indices, values):
    result = np.zeros((indices.shape[1], ROLE_COUNT), dtype=np.float32)
    for rows, table in zip(indices, values):
        result += table[rows]
    return result / len(values)


def decide(scores, delta, winner, temperature, gain, minimum_gap, pair_only):
    """Return final labels and the relative score supporting each proposed switch."""
    z = scores / temperature + gain * delta
    if pair_only:
        proposal = np.where(z[:, REASONING] > z[:, NONE], REASONING, NONE)
    else:
        proposal = z.argmax(1)
    row = np.arange(len(scores))
    gap = z[row, proposal] - z[row, winner]
    active = (gap > minimum_gap) & (proposal != winner)
    if pair_only:
        active &= is_pair(winner)
    return np.where(active, proposal, winner).astype(np.int8), gap


def predict(scores, experts, before, offsets, params, temperature, gain, minimum_gap, pair_only=False):
    features, winner, rival = design(scores, experts, before, offsets, temperature)
    indices = tree_indices(features, winner, rival, params)
    return decide(scores, readout(indices, params["leaf.weight"]), winner, temperature, gain, minimum_gap, pair_only)
