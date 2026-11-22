"""Replay the one locked review-head candidate on the matched LegalSeg test split."""

import argparse
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
from safetensors.numpy import load_file

from reviewer_head import HERE, OLD, ROOT, paired, read_json, sha, summary, write_json


ARM = "all-role"
FREEZE_PATH = HERE / "freeze.json"
BENCHMARK = ROOT / "results/sentence-wide-bert-benchmark.json"
POLICY = ROOT / "results/mixedbread-referral-overlay-frozen.json"
DONOR_PLAN = ROOT / "results/sentence-bert-referral-frozen.json"
LOCK_PATH = HERE / "official-freeze.json"


def commit():
    frozen = read_json(FREEZE_PATH)
    candidate = frozen["selections"][ARM]
    assert candidate["selected_over_identity"] and not candidate["guard"]
    assert candidate["rule"] == read_json(OLD / "review.json")["rule"]
    assert sha(HERE / (ARM + ".safetensors")) == candidate["weights_sha256"]
    validation = read_json(HERE / "results.json")
    audit = read_json(HERE / "native-audit.json")
    assert validation["freeze_sha256"] == audit["frozen_selection_sha256"] == sha(FREEZE_PATH)
    for cohort in ("confirm", "fresh"):
        v = validation["groups"][cohort]["methods"]
        assert v[ARM]["macro_f1"] > v["incumbent"]["macro_f1"]
        assert v[ARM]["none_reasoning_errors"] < v["incumbent"]["none_reasoning_errors"]
        assert audit["groups"][cohort]["methods"][ARM]["prediction_differences"] == 0
    return {"candidate": ARM, "candidate_sha256": candidate["weights_sha256"], "fit_freeze_sha256": sha(FREEZE_PATH),
            "validation_sha256": sha(HERE / "results.json"), "native_audit_sha256": sha(HERE / "native-audit.json"),
            "historical_benchmark_sha256": sha(BENCHMARK), "hierarchy_plus_reviewer_policy_sha256": sha(POLICY),
            "old_readout_recipe_sha256": sha(DONOR_PLAN), "old_readout_weights_sha256": sha(OLD / "head.safetensors"),
            "source": "Existing FP32 encoded window cache; CPU-only reconstruction and head application.",
            "hypothesis": "Read-only evaluation of the selected centered residual on the fixed historical 20% confidence route with original veto.",
            "case_fingerprints": "Prior benchmark official split order and 149,868 nonblank rows. Must recreate the exact 7×7 original confusion before candidate scores.",
            "test_scope": "Only this validation-frozen candidate is scored. Official test was previously used across the project. Candidate selection used official train and validation only.",
            "gradient_steps": 0}


def replay(lock):
    sys.path.insert(0, str(ROOT))
    import mixedbread_referral_overlay as overlay
    import sentence_bert_referral_write as bert
    import sentence_referral_actual_replay as routing
    import sentence_referral_utility_write as utility
    import sentence_wide_bert_benchmark as wide
    from sentence_document_graph_write import OUT, cm, row_summary, torch

    torch.set_num_threads(1)

    def frozen_features_only(*unused, **ignored):
        raise RuntimeError("An archived embedding group is missing; CPU replay requires the full matching cache")

    # The legacy cache reloader imports optional encoder hooks before reading any
    # shards. Install strict cache-only hooks so this replay needs no GPU runtime.
    cache_hooks = ModuleType("openvino_worker")
    cache_hooks.get_engine = frozen_features_only
    cache_hooks.embed_ids = frozen_features_only
    sys.modules["openvino_worker"] = cache_hooks
    benchmark = read_json(BENCHMARK)
    former = benchmark["splits"]["test"]
    benchmark_recipe = benchmark["method"]
    cfg = read_json(DONOR_PLAN)
    policy = read_json(POLICY)
    assert benchmark_recipe["bert_freeze_sha256"] == lock["old_readout_recipe_sha256"]
    assert benchmark_recipe["local_overlay_sha256"] == lock["hierarchy_plus_reviewer_policy_sha256"]
    assert benchmark_recipe["rate"] == .20
    actual_head_file = OUT / (bert.TAG + "-2.safetensors")
    assert sha(actual_head_file) == lock["old_readout_weights_sha256"] == cfg["checkpoints"][overlay.BERT]
    setting = policy["arms"]["BERT only"]["choice"]
    assert setting == read_json(OLD / "review.json")["rule"]
    original_head = load_file(actual_head_file)
    print("Rebuilding the frozen official route", flush=True)
    bank, base, ranks, trace = routing.risk("test")
    assert trace == benchmark_recipe["router_freeze_sha256"]
    assert len(bank[0]) == 712 and len(bank[2]) == 149_868
    assert row_summary(cm(bank[2], base)) == former["baseline"]
    confidence = utility.route(bank[1], ranks["confidence"], .20)
    all_positions = np.union1d(confidence, utility.route(bank[1], ranks["utility"], .20))
    print("Rebuilding legacy reviewer features", len(confidence), "reviewed /", len(all_positions), "in cache union", flush=True)
    rows = wide.feature_bank(bank, "test", all_positions)
    cached, evidence = wide.encode(rows, "test", former["embedding_provenance"]["device"], trace)
    assert evidence == former["embedding_provenance"]
    design = bert.design(rows, cached, cfg["frozen_modes"][overlay.BERT][0], original_head)
    prior = np.concatenate([r[5] for r in rows])
    original = np.concatenate([r[2][r[3]] for r in rows])
    order = np.searchsorted(all_positions, confidence)
    np.testing.assert_array_equal(all_positions[order], confidence)

    def apply(w):
        vote = bert.response(design[order], w)
        return overlay.edit(base, confidence, prior[order], vote, *setting[:2], original[order] if setting[2] else None)

    baseline = apply(original_head)
    former_labels = former["comparison"]["routes"]["confidence"]["labels"]
    assert row_summary(cm(bank[2], baseline)) == former_labels
    m = former_labels["confusion"]
    assert (m[0][5], m[5][0]) == (11_323, 14_089)
    print("Official baseline parity", former_labels["macro_f1"], former_labels["accuracy"], flush=True)
    patch_path = HERE / (ARM + ".safetensors")
    assert sha(patch_path) == lock["candidate_sha256"]
    updated = apply(load_file(patch_path))
    heldout = SimpleNamespace(gold=bank[2], offsets=bank[1], reference=baseline)
    original_record = summary(heldout, baseline)
    candidate = summary(heldout, updated)
    candidate["paired_95"] = paired(heldout, updated, baseline)
    np.savez_compressed(HERE / "official-predictions.npz", case_ids=bank[0], offsets=bank[1], gold=bank[2],
                        incumbent=baseline, all_role=updated, review=confidence)
    return {"lock_sha256": sha(LOCK_PATH), "cache": evidence, "compute": "CPU, no new encoder inference",
            "cases": len(bank[0]), "sentences": len(base), "reviewed": len(confidence), "incumbent_matches_historical_confusion": True,
            "incumbent": original_record, "all_role": candidate, "predictions_sha256": sha(HERE / "official-predictions.npz")}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("freeze", "evaluate"))
    action = p.parse_args().action
    source = commit()
    if action == "freeze":
        write_json(LOCK_PATH, source)
    else:
        assert read_json(LOCK_PATH) == source
        result = replay(source)
        write_json(HERE / "official-results.json", result)
        q = result["all_role"]
        print(result["cases"], result["sentences"], q["macro_f1"], q["accuracy"], q["none_reasoning_errors"], q["paired_95"], flush=True)
