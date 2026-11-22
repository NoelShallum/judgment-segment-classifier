"""Recover the archived official pre-review role stream for a pre-review selector lock."""

import argparse
from pathlib import Path
import sys

import numpy as np

from fit_anchor import HERE, checked_npz, hash_file, read, write


def extract(lock_path, dest):
    lock = read(lock_path)
    assert lock["chosen_arm"] == "pre-review" and not lock["official_selection"]
    assert lock["chosen_cases"] in (256, 512)
    workspace = HERE.parents[1]
    sys.path.insert(0, str(workspace))

    import sentence_referral_actual_replay as routing
    import sentence_referral_utility_write as utility
    from sentence_document_graph_write import cm, row_summary, torch

    torch.set_num_threads(2)
    archive = workspace / "results/sentence-wide-bert-benchmark.json"
    historical = read(archive)
    bank, pre, ranks, route_hash = routing.risk("test")
    assert route_hash == historical["method"]["router_freeze_sha256"]
    assert row_summary(cm(bank[2], pre)) == historical["splits"]["test"]["baseline"]
    expected = HERE.parent / "reviewer/official-predictions.npz"
    assert hash_file(expected) == "b0b70e63657904c4f90a73f77b8e48a1f123fba013faf195d35a9836d7fad4bc"
    result = checked_npz(expected)
    for column, original in zip(("case_ids", "offsets", "gold"), bank[:3]):
        np.testing.assert_array_equal(result[column], original)
    positions = utility.route(bank[1], ranks["confidence"], .20)
    np.testing.assert_array_equal(positions, result["review"])
    changed = np.flatnonzero(result["incumbent"] != pre)
    assert np.isin(changed, positions).all()
    np.savez_compressed(dest, pre=pre, **{k: result[k] for k in ("case_ids", "offsets", "gold", "incumbent")})
    write(dest.with_suffix(".json"), {"file_sha256": hash_file(dest), "official_lock_sha256": hash_file(lock_path),
                                     "router_sha256": route_hash, "historical_benchmark_sha256": hash_file(archive),
                                     "judgments": len(bank[0]), "rows": len(pre), "original_changes": len(changed)})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lock", type=Path, required=True)
    p.add_argument("--output", type=Path, default=HERE / "official-pre-review.npz")
    args = p.parse_args()
    extract(args.lock, args.output)
