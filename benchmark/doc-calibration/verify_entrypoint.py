"""Verify the seven-value patch through the existing CPU judgment entrypoint."""

import argparse
from pathlib import Path
import sys

import numpy as np
from safetensors.numpy import load_file

import calibrate as c

RELEASE = c.LEGAL / "release" / "portable"


def qa(donors):
    native = c.read(c.HERE / "native-audit.json")
    patch = c.HERE / "doc-semantic-bias.safetensors"
    assert c.digest(patch) == native["patched_decoder_sha256"]
    new = load_file(patch)
    sys.path.insert(0, str(RELEASE))
    sys.path.insert(0, str(RELEASE / "research"))
    import classify
    import context_assets
    import solve

    context_assets.verify_release()
    original_load = classify.read_weights
    original_decide = solve.decide
    receipt = {}

    def read_weights(directory, name):
        source = original_load(directory, name)
        if name != "doc-semantic.safetensors":
            return source
        actual = new["coef"] - source["coef"]
        assert np.count_nonzero(actual) == 7
        np.testing.assert_array_equal(actual[-1], np.asarray(native["delta"], dtype=np.float32))
        receipt["patched_weights_passed_to_entrypoint"] = True
        return new

    def decide(response, base, anchor, margin, scope):
        label = original_decide(response, base, anchor, margin, scope)
        delta = np.asarray(native["delta"], dtype=response.dtype)
        old = original_decide(response - delta, base, anchor, margin, scope)
        receipt.update(patched_labels=label.tolist(), upstream_labels=old.tolist(),
                       changes=int(np.count_nonzero(old != label)), number_of_rows=len(old))
        return label

    classify.read_weights = read_weights
    solve.decide = decide
    source = RELEASE / "example.json"
    try:
        result = classify.classify(classify.read_sentences(source), context_assets.donors_at(donors))
    finally:
        classify.read_weights = original_load
        solve.decide = original_decide
    wanted = [classify.LABELS[k] for k in receipt["patched_labels"]]
    assert result["labels"] == wanted
    c.write(c.HERE / "entrypoint-verification.json", {
        **receipt, "device": "CPU", "source": "Published release fixture and full standard whole-judgment classifier.",
        "fixture_sha256": c.digest(source), "classifier_sha256": c.digest(RELEASE / "classify.py"),
        "patch_sha256": c.digest(patch), "labels": wanted, "backbone_windows": result["backbone_windows"],
    })
    print(receipt, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--donor-dir", type=Path, required=True)
    qa(cli.parse_args().donor_dir.resolve())
