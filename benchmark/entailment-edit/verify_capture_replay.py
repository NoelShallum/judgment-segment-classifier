"""Verify locked fusion against a direct native-head recomputation of every held-out pair."""

from io import BytesIO
from zipfile import ZipFile

import numpy as np
from safetensors.numpy import load_file

from prepare import ARCHIVE, HERE, baseline
from protocol import digest
from solve import CAPTURES, FROZEN, WEIGHTS, Scored, candidates, capture_metadata, verified


def centered_logits(hidden, head, bias):
    logits = hidden @ head.T + bias
    margin = logits[..., 0] - logits[..., 2]
    return margin - margin.mean(1, keepdims=True)


def verify():
    lock = baseline.read(FROZEN)
    assert digest(CAPTURES) == lock["capture_sha256"] and digest(WEIGHTS) == lock["weights_sha256"]
    weight = load_file(str(WEIGHTS))
    config = lock["evaluated_edit"]
    scale, temp = lock["native_scale"], lock["document_scale"]
    result = {}
    with ZipFile(CAPTURES) as z:
        metadata, source = capture_metadata(z)
        np.testing.assert_array_equal(source, weight["source"])
        expected_bias = np.asarray(metadata["effective_classifier_bias"] or [0, 0, 0], dtype=np.float32)
        if "classifier.bias" in weight:
            np.testing.assert_array_equal(expected_bias, weight["classifier.bias"])
        for group in ("select", "confirm", "fresh"):
            raw = verified(z, f"capture-{group}.npz", metadata["captures"][group]["output_sha256"])
            with np.load(BytesIO(raw), allow_pickle=False) as stored:
                hidden = stored["hidden"].astype(np.float32)
                idx = stored["rows"]
            bank = baseline.load_bank(ARCHIVE, group)
            gate = candidates(Scored(bank, idx, None, None, 0.), temp)[config["scope"]]
            original = centered_logits(hidden, source, expected_bias)
            current = centered_logits(hidden, weight["classifier.weight"], expected_bias)
            effective = current if config["composition"] == "edited" else current - original
            correction = config["gain"] * temp * np.clip(effective / scale, -6., 6.)
            decisions = bank.base.copy()
            decisions[idx] = np.where(gate, bank.scores[idx] + correction, -np.inf).argmax(1)
            with np.load(HERE / f"{group}-predictions.npz", allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["case_ids"], bank.keys)
                np.testing.assert_array_equal(archive["edited-lock"], decisions)
            result[group] = {"decisions_verified": len(decisions), "routed": len(idx), "nli_pairs": 7 * len(idx)}
    baseline.write(HERE / "complete-native-replay.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    verify()
