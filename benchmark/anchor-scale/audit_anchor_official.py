"""Recompute the locked anchor test result and verify its historical row alignment."""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from build_anchor_release import exact_archive
from fit_anchor import HERE, checked_npz, hash_file, metrics, rank, write
from verify_stages import digest, npz


def verify(path):
    with ZipFile(path) as archive:
        exact_archive(archive)
        result = json.loads(archive.read("official-results.json"))
        lock = json.loads(archive.read("official-lock.json"))
        freeze = json.loads(archive.read("freeze.json"))
        assert result["lock_sha256"] == digest(archive, "official-lock.json")
        assert lock["stages"][str(result["chosen_cases"])]["freeze_sha256"] == digest(archive, "freeze.json")
        assert freeze["weights_sha256"] == result["weights_sha256"] == digest(archive, "expanded-anchors.safetensors")
        assert freeze["basis_sha256"] == result["basis_sha256"] == digest(archive, "basis.safetensors")
        assert result["predictions_sha256"] == digest(archive, "official-predictions.npz")
        stages = lock["stages"]
        for size, candidate in stages.items():
            assert rank(candidate["selector"]) == tuple(candidate["quality"])
            dev = HERE / f"anchor-{size}-evidence.zip"
            if dev.is_file():
                with ZipFile(dev) as evidence:
                    assert digest(evidence, "head-fit/freeze.json") == candidate["freeze_sha256"]
                    assert digest(evidence, "head-fit/results.json") == candidate["results_sha256"]
        pick = max(stages, key=lambda n: stages[n]["quality"])
        assert int(pick) == lock["chosen_cases"] == freeze["cases"] == result["chosen_cases"]
        assert stages[pick]["selected_arm"] == lock["chosen_arm"] == freeze["selection"] == result["chosen_arm"]
        assert not lock["official_selection"] and not freeze["official_test_read"]
        test = npz(archive, "official-predictions.npz")
    earlier = HERE.parent / "reviewer/official-predictions.npz"
    assert hash_file(earlier) == "b0b70e63657904c4f90a73f77b8e48a1f123fba013faf195d35a9836d7fad4bc"
    baseline = checked_npz(earlier)
    for identical in ("case_ids", "offsets", "gold", "incumbent"):
        np.testing.assert_array_equal(baseline[identical], test[identical])
    assert len(set(test["case_ids"])) == result["judgments"] == 712
    assert len(test["gold"]) == result["rows"] == 149_868
    for field in ("incumbent", "doc_semantic", "selected"):
        assert metrics(test["gold"], test[field]) == result["metrics"][field.replace("_", "-")]
    return {"archive_sha256": hash_file(path), "compared_to_historical_predictions_sha256": hash_file(earlier),
            "case_alignment": True, "matrices_recomputed": True, "selected": result["chosen_arm"], "cases_fitted": result["chosen_cases"],
            "judgments": result["judgments"], "rows": result["rows"], "metrics": result["metrics"], "paired_95": result["pairs_95"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "official-independent-audit.json")
    args = parser.parse_args()
    checked = verify(args.archive)
    write(args.output, checked)
    for label, numbers in checked["metrics"].items():
        print(label, numbers["macro_f1"], numbers["accuracy"], numbers["pair_errors"], flush=True)
