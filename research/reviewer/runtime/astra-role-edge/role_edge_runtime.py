"""Gradient-free lexical observations for 49 rhetorical-role transition scores."""
import re

import numpy as np

WORDS = re.compile(r"[a-z]+|[0-9]+", re.I | re.ASCII)
FEATURE_MASS = {"r1": .35, "r2": .55, "r3": .85, "l1": .18, "l2": .27,
                "lr": .35, "pun": .18, "zone": .25}
OFF_DIAGONAL = ~np.eye(7, dtype=bool)


def terms(text):
    return ["#" if s.isdecimal() else s.lower() for s in WORDS.findall(text)]


def signatures(text, units):
    items = []
    for start, end in units:
        forward = text[start:min(end, start + 150)].lstrip()
        backward = text[max(start, end - 100):end].rstrip()
        prefix = terms(forward)[:3] or ["empty"]
        suffix = terms(backward)[-2:] or ["empty"]
        mark = backward[-1:] or "_"
        face = "number" if forward[:1].isdigit() else "punct" if not forward[:1].isalpha() else "word"
        items.append((prefix, suffix, mark, face, start))
    for left, right in zip(items, items[1:]):
        beginning, ending = right[0], left[1]
        yield ("r1:" + beginning[0], "r2:" + ' '.join(beginning[:2]),
               "r3:" + ' '.join(beginning), "l1:" + ending[-1],
               "l2:" + ' '.join(ending), "lr:" + ending[-1] + ' ' + beginning[0],
               "pun:" + left[2] + ' ' + right[3],
               "zone:" + str(min(15, right[4] * 16 // len(text))))


def transition_scores(text, units, vocab, tensors):
    index = [[vocab.get(k, 0) for k in row] for row in signatures(text, units)]
    if not index:
        return np.zeros((0, 7, 7), np.float32)
    return tensors["edge.log_ratio"][index].sum(axis=1)


def decompose(scores, mode):
    if mode == "joint":
        return scores
    diagonal = scores.diagonal(axis1=1, axis2=2)
    additive = (diagonal[:, :, None] + diagonal[:, None, :]) * .5
    if mode == "unary":
        return additive
    interaction = scores - additive
    if mode == "interaction":
        return interaction
    if mode == "pooled":
        return interaction[:, OFF_DIAGONAL].mean(axis=1)[:, None, None] * OFF_DIAGONAL
    raise ValueError(mode)


def decode(logits, transition, edge, strength):
    """Maximize the written sequence score over all seven-role paths."""
    if not len(logits):
        raise ValueError("Expected at least one role emission")
    rows = np.arange(7)
    trans = transition[3:, 3:]
    current = logits[0] + transition[1, 3:]
    origins = np.empty((len(logits) - 1, 7), dtype=np.int8)
    for i, scores in enumerate(strength * edge):
        options = current[:, None] + trans + scores
        best = options.argmax(0)
        origins[i] = best
        current = options[best, rows] + logits[i+1]
    labels = np.empty(len(logits), dtype=np.int8)
    labels[-1] = np.argmax(current + transition[3:, 2])
    for i in range(len(origins)-1, -1, -1):
        labels[i] = origins[i, labels[i+1]]
    return labels
