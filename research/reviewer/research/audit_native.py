"""Verify serialized reviewer writes through the native readout and audit its gate."""

import importlib.util
import sys
from types import SimpleNamespace

import numpy as np
from safetensors.numpy import load_file
import torch

from reviewer_head import (DIRECTION, HERE, OLD, PLAN, POOLS, REASONING, decisions, is_pair, read_bank,
                           read_json, sha, summary, write_json)


def reviewer_class():
    sys.path.insert(0, str(OLD))
    name = "deployed_reviewer"
    spec = importlib.util.spec_from_file_location(name, OLD / "fit.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.Readout, module.prob


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    native, prob = reviewer_class()
    head = native(OLD, "cpu")
    receipt = read_json(POOLS / "pool-receipt.json")
    plan = read_json(PLAN)
    freeze = read_json(HERE / "freeze.json")
    assert sha(POOLS / "pool-receipt.json") == freeze["pool_receipt_sha256"]
    report = {"native_readout_file_sha256": sha(OLD / "fit.py"), "frozen_selection_sha256": sha(HERE / "freeze.json"),
              "groups": {}}
    for group in ("select", "confirm", "fresh"):
        bank = read_bank(group, plan, receipt)
        pos = bank.positions
        input_ = SimpleNamespace(parent=bank.parent, prior=bank.prior, review=pos, original=bank.original)
        current, target = bank.reference[pos], bank.gold[pos]
        aux, pool = torch.from_numpy(bank.aux), torch.from_numpy(bank.pool)
        origin = native(OLD, "cpu")
        with torch.inference_mode():
            raw_reference = origin.raw(aux, pool)
            p = prob(raw_reference)
            reference = origin.review(input_, p)
        np.testing.assert_array_equal(reference, bank.reference)
        pair_error = is_pair(target) & is_pair(current) & (current != target)
        reference_parent = bank.parent[pos]
        sign = np.where(target == REASONING, 1., -1.)
        raw = raw_reference.numpy()
        dominant = raw + (10. * np.maximum(np.maximum(raw, 0.).sum(1), .25) * sign)[:, None] * DIRECTION
        gold_at_dominant = decisions(dominant, reference_parent, bank.original, bank.prior, origin.rule) == target
        unlimited_mirror = bank.reference.copy()
        unlimited_mirror[pos[pair_error & gold_at_dominant]] = target[pair_error & gold_at_dominant]
        pair_truth_raw_winner = p.argmax(1).numpy() == target
        high_confidence = p[np.arange(len(p)), target].numpy() - p[np.arange(len(p)), reference_parent].numpy() >= origin.rule[1]
        role_fix = {
            "review_rows": len(pos), "within_route_pair_errors": int(pair_error.sum()),
            "pair_errors_with_review_argmax_already_gold": int(np.count_nonzero(pair_error & pair_truth_raw_winner)),
            "subset_above_margin_and_blocked_by_veto": int(np.count_nonzero(pair_error & pair_truth_raw_winner & high_confidence & (bank.original == target))),
            "fixed_by_dominant_gold_pair_score_under_native_rule": int(np.count_nonzero(pair_error & gold_at_dominant)),
            "remaining_under_native_rule": int(np.count_nonzero(pair_error & ~gold_at_dominant)),
            "oracle_with_existing_gate": summary(bank, unlimited_mirror),
        }
        pred_path = HERE / (group + "-predictions.npz")
        methods = {}
        with np.load(pred_path, allow_pickle=False) as expected:
            for name, choice in freeze["selections"].items():
                if choice is None:
                    continue
                versions = [(name, choice["weights_sha256"])]
                if "shuffled_weights_sha256" in choice:
                    versions.append((name + "-shuffled", choice["shuffled_weights_sha256"]))
                for version, digest in versions:
                    state = HERE / (version + ".safetensors")
                    assert sha(state) == digest
                    weights = load_file(state)
                    head.params = {key: torch.from_numpy(value) for key, value in weights.items()}
                    head.rule = choice["rule"]
                    with torch.inference_mode():
                        pred = head.review(input_, prob(head.raw(aux, pool)))
                    if choice["guard"]:
                        diff = ~is_pair(pred) | ~is_pair(reference)
                        pred[diff] = reference[diff]
                    np.testing.assert_array_equal(pred, expected[version])
                    methods[version] = {"prediction_differences": 0, "sentences": len(pred), "weights_sha256": digest}
        report["groups"][group] = {"native_baseline_agreement": len(reference), "gate_audit": role_fix, "methods": methods}
        print(group, "native parity across", len(methods), "methods", "pair route oracle",
              role_fix["fixed_by_dominant_gold_pair_score_under_native_rule"], "/", role_fix["within_route_pair_errors"], flush=True)
    write_json(HERE / "native-audit.json", report)


if __name__ == "__main__":
    main()

