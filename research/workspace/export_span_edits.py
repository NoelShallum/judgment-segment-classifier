"""Package the verified whole-judgment edit profiles and their evidence."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from hashlib import file_digest
import json
import os

ROOT=Path(__file__).resolve().parent
OUTPUT=ROOT.parents[1]/"outputs/judgment-role/AttoFlow-judgment-span-edits.zip"
PRETRAINED=ROOT/"references/hier"
PATCH=ROOT/"checkpoints/hierarchical"

def digest(path):
    with path.open("rb") as f:return file_digest(f,"sha256").hexdigest()

def main():
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    trials=ROOT/"results"
    files={"hier_model.py":ROOT/"hier_model.py","span_model.py":ROOT/"span_model.py",
           "exact_span.safetensors":PATCH/"role-specific-whole.safetensors",
           "token_balanced.safetensors":PATCH/"role-specific-balanced.safetensors",
           "evidence/role-specific-transitions.json":trials/"role-specific-transitions.json",
           "evidence/transition-token-tradeoff.json":trials/"transition-token-tradeoff.json",
           "evidence/boundary-span-coverage.json":trials/"boundary-span-coverage.json",
           "evidence/transition-per-role-audit.json":trials/"transition-per-role-audit.json",
           "evidence/archived-sentence-span-references.json":trials/"archived-sentence-span-references.json",
           "evidence/named-hierarchy-span-oracle.json":trials/"named-hierarchy-span-oracle.json",
           "evidence/archive-span-paired-comparison.json":trials/"archive-span-paired-comparison.json",
           "reproduce/transition_span_edit.py":ROOT/"transition_span_edit.py",
           "reproduce/transition_tradeoff.py":ROOT/"transition_tradeoff.py"}

    if (trials/"discourse-span-weights.json").exists():
        files.update({"discourse_exact.safetensors":PATCH/"whole-discourse-exact.safetensors",
                      "discourse_balanced.safetensors":PATCH/"whole-discourse-balanced.safetensors",
                      "predict_discourse_spans.py":ROOT/"predict_discourse_spans.py","role_cues.py":ROOT/"role_cues.py",
                      "evidence/discourse-span-weights.json":trials/"discourse-span-weights.json",
                      "reproduce/discourse_span_edit.py":ROOT/"discourse_span_edit.py"})
    if (trials/"span-boundary-oracle.json").exists():
        files["evidence/span-boundary-oracle.json"]=trials/"span-boundary-oracle.json"
    for name in ("independent-archive-replay.json","independent-archive-provenance.json","rhetoricllama-parser-sensitivity.json"):
        if (trials/name).is_file():files["evidence/"+name]=trials/name
    if (ROOT/"astra-audit/archive_document_replay.py").is_file():
        files["reproduce/archive_document_replay.py"]=ROOT/"astra-audit/archive_document_replay.py"
    if (ROOT/"astra-audit/archive_provenance_replay.py").is_file():
        files["reproduce/archive_provenance_replay.py"]=ROOT/"astra-audit/archive_provenance_replay.py"
    if (ROOT/"astra-benchmarks/replay_rhetoricllama.py").is_file():
        files["reproduce/replay_rhetoricllama.py"]=ROOT/"astra-benchmarks/replay_rhetoricllama.py"
    for code in ("empirical_duration_write.py","focused_cloud_comparison.py","full_span_error_decomposition.py",
                 "raw_emitter_write.py","lstm_input_subspace_write.py","span_role_relabel_write.py",
                 "deberta_span_write.py","deberta_isolated_screen.py","julia_choice_span_write.py","score_frozen_edge_sentence_rows.py",
                 "measure_role_edge_latency.py","role_edge_interpolation.py","span_posterior_decode.py",
                 "role_edge_posterior_study.py","role_edge_feature_mix.py","role_edge_interpolation_qa.py",
                 "optional_boundary_runtime.py","optional_boundary_study.py","optional_merge_study.py",
                 "boundary_relocation_study.py","optional_cut_reachability.py","calibrated_optional_cut.py",
                 "calibrated_cut_ranking.py","packed_role_decode.cpp","role_pair_weight_search.py",
                 "known_sentence_pair_edit.py","role_pair_position_edit.py","verify_sentence_pair_checkpoint.py",
                 "sentence_role_retention.py","sentence_emission_calibration.py","sentence_edit_composition.py",
                 "audit_known_sentence_archives.py","sentence_calibration_gate.py","audit_sentence_calibration_changes.py",
                 "sentence_label_change_edit.py","verify_sentence_label_change.py",
                 "sentence_hidden_edit.py","sentence_lexical_fusion_edit.py","sentence_hidden_repair.py",
                 "segment_weight_write.py","segment_span_runtime.py","packed_segment_path.cpp","verify_segment_spans.py",
                 "segment_preference_edit.py","segment_coverage_audit.py","short_span_proposals_audit.py","short_span_weight_edit.py",
                 "verify_short_span_density.py","novel_span_repair.py","novel_span_repair_audit.py","sentence_prototype_edit.py",
                 "sentence_prototype_gate.py","precompute_repair_test_states.py","sentence_minilm_fusion.py",
                 "audit_minilm_fusion_batch.py","audit_sentence_fusion_routing.py","verify_sentence_uncertainty.py",
                 "sentence_uncertainty_runtime.py","sentence_joint_decoding.py","sentence_joint_runtime.py",
                 "verify_sentence_joint.py","verify_sentence_joint_portable.py","whole_joint_transfer.py",
                 "precompute_raw_joint_states.py","verify_whole_joint_transfer.py","whole_joint_runtime.py",
                 "verify_raw_joint_direct.py","sentence_conditioned_edges.py","verify_sentence_conditioned_edges.py",
                 "probe_embeddinggemma_q4.py","embeddinggemma_routed_write.py","verify_embeddinggemma_routed.py",
                 "confirm_embeddinggemma_routed.py","probe_embeddinggemma2_q8.py","gemma2_neighbor_write.py",
                 "audit_author_complementarity.py","audit_joint_repair_routing.py","sentence_ngram_residual_write.py","verify_sentence_ngram_residual.py","sentence_ngram_pair_gate.py","verify_sentence_ngram_pair_gate.py",
                 "sentence_document_graph_write.py","audit_sentence_document_graph.py","compare_sentence_document_graph.py","verify_sentence_local_readout_portable.py",
                 "confirm_sentence_graph_holdout.py","verify_raw_local_readout.py","verify_sentence_document_readout_portable.py",
                 "sentence_graph_runtime.py","audit_sentence_context_budget.py","isolated_role_readout_write.py","verify_isolated_role_readout_portable.py"):
        files["reproduce/"+code]=ROOT/code
    for name in ("whole-length-strata","role-entry-selected","role-entry-diagnostic","qwen-sixhundred-engineering","qwen-sixhundred-engineering-qa",
                 "segment-coverage-audit","segment-exact-cpu-latency","short-span-proposals-audit","short-span-density-qa",
                 "novel-span-repair-audit","sentence-minilm-batch-audit","sentence-minilm-routing-audit",
                 "sentence-uncertainty-cpu-latency","sentence-joint-cpu-latency","sentence-joint-qa",
                 "sentence-joint-portable-qa","whole-joint-transfer-qa","raw-joint-direct-qa","sentence-conditioned-edges-qa","embeddinggemma-q4-cpu-probe",
                 "embeddinggemma2-q8-cpu-probe","sentence-author-complementarity","legalseg-paper-figure-recheck",
                 "sentence-document-graph-affinity"):
        source=trials/(name+".json")
        if source.is_file():files["evidence/"+source.name]=source
    for name in ("length-adaptive-span-write","contextual-boundary-write","contextual-boundary-independent-validation",
                 "encoder-span-minilm","encoder-span-granite-accelerated","encoder-span-gte-accelerated",
                 "encoder-norm-edit-granite","encoder-norm-edit-gte","role-specialized-encoder-fusion","hard-preservation-emitter","whole-span-cpu-latency",
                 "corpus-residual-memory","nli-minilm-span-edit","nli-minilm-full-confirmation","quotation-span-edit","encoder-source-keys-granite",
                 "empirical-duration-write","empirical-duration-frozen","focused-span-gemini-replay","context-size-validation",
                 "current-discourse-span-error-decomposition","deberta-span-write","deberta-isolated-screen","lstm-input-subspace-write",
                 "lstm-input-subspace-frozen","raw-conditioned-emitter-screen","raw-conditioned-emitter-frozen",
                 "raw-conditioned-emitter-full","role-relabel-write","role-relabel-frozen","role-edge-given-sentences",
                 "julia-choice-span-write","julia-choice-frozen","role-edge-cpu-latency",
                 "role-edge-interpolation","role-edge-interpolation-frozen","role-edge-posterior",
                 "role-edge-posterior-frozen","role-edge-feature-mix","role-edge-feature-mix-frozen",
                 "role-edge-interpolation-qa","optional-boundary-lattice","optional-boundary-frozen",
                 "optional-boundary-train","optional-merge-lattice","optional-merge-frozen",
                 "boundary-relocation","boundary-relocation-frozen","optional-boundary-reachability",
                 "calibrated-boundary-lattice","calibrated-boundary-frozen","calibrated-boundary-train",
                 "calibrated-boundary-ranking","role-pair-search","role-pair-search-train","role-pair-search-frozen",
                 "known-sentence-pair-edit","known-sentence-pair-edit-frozen","known-sentence-pair-edit-qa",
                 "role-pair-position-edit","role-pair-position-frozen","sentence-role-retention","sentence-role-retention-frozen",
                 "sentence-emission-calibration","sentence-emission-calibration-frozen","sentence-edit-composition",
                 "sentence-edit-composition-frozen","known-sentence-numerical-archives",
                 "sentence-calibration-gate","sentence-calibration-gate-frozen","sentence-calibration-change-audit",
                 "sentence-label-change","sentence-label-change-frozen","sentence-label-change-qa",
                 "sentence-hidden-edit","sentence-hidden-edit-frozen","sentence-lexical-fusion-edit",
                 "sentence-lexical-fusion-frozen"):
        record=trials/(name+".json")
        if record.is_file():files["evidence/"+record.name]=record
    repair=trials/"sentence-hidden-repair.json"
    repair_frozen=trials/"sentence-hidden-repair-frozen.json"
    if repair.is_file() and repair_frozen.is_file():
        study=json.loads(repair.read_text())
        if study["freeze_sha256"]==digest(repair_frozen) and (study["choice"] is None or "test" in study["splits"]):
            for record in (repair,repair_frozen):files["evidence/"+record.name]=record
    for stem in ("segment-weight","segment-posterior-ends"):
        written_segments=trials/(stem+"-write.json")
        frozen_segments=trials/(stem+"-frozen.json")
        if written_segments.is_file() and frozen_segments.is_file():
            study=json.loads(written_segments.read_text())
            if study["freeze_sha256"]==digest(frozen_segments) and "test" in study["splits"]:
                for record in (written_segments,frozen_segments):files["evidence/"+record.name]=record
    for stem in ("segment-preference","short-span-density","sentence-prototype","sentence-prototype-gate","sentence-minilm-fusion",
                 "sentence-minilm-token-table","sentence-joint","whole-joint-transfer"):
        study_file=trials/(stem+"-edit.json")
        freeze_file=trials/(stem+"-frozen.json")
        if study_file.is_file() and freeze_file.is_file():
            study=json.loads(study_file.read_text())
            if study["freeze_sha256"]==digest(freeze_file) and "test" in study["splits"]:
                for record in (study_file,freeze_file):files["evidence/"+record.name]=record
    repair=trials/"novel-span-repair.json"
    repair_freeze=trials/"novel-span-repair-frozen.json"
    if repair.is_file() and repair_freeze.is_file():
        record=json.loads(repair.read_text())
        if record["freeze_sha256"]==digest(repair_freeze) and "test" in record["splits"]:
            for path in (repair,repair_freeze):files["evidence/"+path.name]=path
    archived_rows=trials/"known-sentence-numerical-archives.npz"
    if archived_rows.is_file():files["evidence/"+archived_rows.name]=archived_rows
    edge=ROOT/"astra-role-edge"
    qualifies=False
    intermediate=False
    sentence_profile=None
    calibrated_profile=None
    segment_profile=None
    uncertainty_profile=None
    joint_profile=None
    raw_joint_profile=None
    local_profile=None
    document_profile=None
    document_calibrated_profile=None
    raw_local_profile=None
    isolated_profile=None
    conditional_edit=trials/"sentence-conditioned-edges.json"
    conditional_weights=PATCH/"known-sentence-conditioned-edges.safetensors"
    if conditional_edit.is_file() and conditional_weights.is_file():
        proof=json.loads(conditional_edit.read_text())
        freeze=trials/"sentence-conditioned-edges-frozen.json"
        if "test" in proof["splits"] and proof["freeze_sha256"]==digest(freeze):
            assert json.loads(freeze.read_text())["weights_sha256"]==digest(conditional_weights)
            for path in (conditional_edit,freeze):
                files["evidence/"+path.name]=path
            files["reproduce/checkpoints/known-sentence-conditioned-edges.safetensors"]=conditional_weights
    gemma=trials/"embeddinggemma-routed-write.json"
    gemma_freeze=trials/"embeddinggemma-routed-frozen.json"
    if gemma.is_file() and gemma_freeze.is_file():
        study=json.loads(gemma.read_text())
        if study["freeze_sha256"]==digest(gemma_freeze) and "confirm" in study["splits"]:
            for path in (gemma,gemma_freeze):
                files["evidence/"+path.name]=path
            for name in ("embeddinggemma-routed-qa.json","embeddinggemma-routed-independent.json"):
                record=trials/name
                if record.is_file() and json.loads(record.read_text())["freeze_sha256"]==study["freeze_sha256"]:
                    files["evidence/"+name]=record
            frozen=json.loads(gemma_freeze.read_text())
            for arm,checksum in frozen["weights_sha256"].items():
                source=PATCH/f"embeddinggemma-routed-{arm}.safetensors"
                assert digest(source)==checksum
                files["reproduce/checkpoints/"+source.name]=source
    context_study=trials/"gemma2-neighbor.json"
    context_freeze=trials/"gemma2-neighbor-frozen.json"
    if context_study.is_file() and context_freeze.is_file():
        data=json.loads(context_study.read_text())
        if "independent" in data["splits"] and data["freeze_sha256"]==digest(context_freeze):
            for source in (context_study,context_freeze):
                files["evidence/"+source.name]=source
            for name,expected in json.loads(context_freeze.read_text())["weights_sha256"].items():
                source=PATCH/f"gemma2-neighbor-{name}.safetensors"
                assert digest(source)==expected
                files["reproduce/checkpoints/"+source.name]=source
    routing_study=trials/"sentence-joint-repair-routing.json"
    routing_freeze=trials/"sentence-joint-repair-routing-frozen.json"
    if routing_study.is_file() and routing_freeze.is_file():
        data=json.loads(routing_study.read_text())
        if "test" in data["splits"] and data["freeze_sha256"]==digest(routing_freeze):
            for source in (routing_study,routing_freeze):
                files["evidence/"+source.name]=source
            source=PATCH/"sentence-joint-repair-routing.safetensors"
            assert digest(source)==json.loads(routing_freeze.read_text())["weights_sha256"]
            files["reproduce/checkpoints/"+source.name]=source
    ngram=trials/"sentence-ngram-residual.json"
    ngram_freeze=trials/"sentence-ngram-residual-frozen.json"
    if ngram.is_file() and ngram_freeze.is_file():
        data=json.loads(ngram.read_text())
        fixed=json.loads(ngram_freeze.read_text())
        expected="independent" if fixed["choice"][0]=="unchanged" else "test"
        if expected in data["splits"] and data["freeze_sha256"]==digest(ngram_freeze):
            for record in (ngram,ngram_freeze):
                files["evidence/"+record.name]=record
            qa=trials/"sentence-ngram-residual-qa.json"
            if qa.is_file():
                replay=json.loads(qa.read_text())
                if replay["freeze_sha256"]==data["freeze_sha256"] and replay["candidate"]==data["splits"]["test"]["candidate"]:
                    files["evidence/"+qa.name]=qa
            for label,suffix in (("memory_sha256","-memory"),("head_sha256","")):
                if checksum:=fixed[label]:
                    artifact=PATCH/f"sentence-ngram-residual{suffix}.safetensors"
                    assert digest(artifact)==checksum
                    files["reproduce/checkpoints/"+artifact.name]=artifact
    gate=trials/"sentence-ngram-pair-gate.json"
    gate_freeze=trials/"sentence-ngram-pair-gate-frozen.json"
    if gate.is_file() and gate_freeze.is_file():
        data=json.loads(gate.read_text())
        fixed=json.loads(gate_freeze.read_text())
        expected="independent" if fixed["choice"][0]=="unchanged" else "test"
        if expected in data["splits"] and data["freeze_sha256"]==digest(gate_freeze):
            for record in (gate,gate_freeze):
                files["evidence/"+record.name]=record
            qa=trials/"sentence-ngram-pair-gate-qa.json"
            if qa.is_file() and "test" in data["splits"]:
                replay=json.loads(qa.read_text())
                if replay["freeze_sha256"]==data["freeze_sha256"] and replay["candidate"]==data["splits"]["test"]["candidate"]:
                    files["evidence/"+qa.name]=qa
            if checksum:=fixed["gate_sha256"]:
                artifact=PATCH/"sentence-ngram-pair-gate.safetensors"
                assert digest(artifact)==checksum
                files["reproduce/checkpoints/"+artifact.name]=artifact
    graph=trials/"sentence-document-graph.json"
    graph_freeze=trials/"sentence-document-graph-frozen.json"
    if graph.is_file() and graph_freeze.is_file():
        data=json.loads(graph.read_text())
        fixed=json.loads(graph_freeze.read_text())
        expected="independent" if fixed["choice"][0]=="unchanged" else "test"
        if expected in data["splits"] and data["freeze_sha256"]==digest(graph_freeze):
            for artifact in (graph,graph_freeze):
                files["evidence/"+artifact.name]=artifact
            if checksum:=fixed["weights_sha256"]:
                artifact=PATCH/"sentence-document-graph.safetensors"
                assert checksum==digest(artifact)
                files["reproduce/checkpoints/"+artifact.name]=artifact
            comparison=trials/"sentence-document-graph-comparison.json"
            if comparison.is_file():
                r=json.loads(comparison.read_text())
                if "test" in r["splits"] and r["freeze_sha256"]==data["freeze_sha256"]:
                    files["evidence/"+comparison.name]=comparison
                    for name,checksum in r["comparison_weights_sha256"].items():
                        artifact=PATCH/f"sentence-document-graph-{name}.safetensors"
                        assert checksum==digest(artifact)
                        files["reproduce/checkpoints/"+artifact.name]=artifact
    reserved=trials/"sentence-graph-holdout.json"
    reserved_freeze=trials/"sentence-graph-holdout-frozen.json"
    if reserved.is_file() and reserved_freeze.is_file():
        v=json.loads(reserved.read_text())
        if v["freeze_sha256"]==digest(reserved_freeze):
            for record in (reserved,reserved_freeze):
                files["evidence/"+record.name]=record
    raw_local=trials/"raw-local-transfer.json"
    if raw_local.is_file():
        v=json.loads(raw_local.read_text())
        if v["local_weights_sha256"]==digest(PATCH/"sentence-local-readout-portable.safetensors"):
            files["evidence/"+raw_local.name]=raw_local
    context_budget=trials/"sentence-context-budget.json"
    if context_budget.is_file() and reserved_freeze.is_file():
        proof=json.loads(context_budget.read_text())
        if proof["holdout_sha256"]==digest(reserved_freeze):
            files["evidence/"+context_budget.name]=context_budget
    input_size=trials/"sentence-input-size.json"
    doc_source=PATCH/"sentence-document-readout-portable.safetensors"
    if input_size.is_file() and doc_source.is_file():
        proof=json.loads(input_size.read_text())
        if proof["weights_sha256"]==digest(doc_source):
            files["evidence/"+input_size.name]=input_size
            files["reproduce/audit_sentence_input_size.py"]=ROOT/"audit_sentence_input_size.py"
    lone=trials/"isolated-role-readout.json"
    lone_freeze=trials/"isolated-role-readout-frozen.json"
    if lone.is_file() and lone_freeze.is_file():
        proof=json.loads(lone.read_text())
        metadata=json.loads(lone_freeze.read_text())
        if "test" in proof["splits"] and proof["freeze_sha256"]==digest(lone_freeze):
            for item in (lone,lone_freeze):
                files["evidence/"+item.name]=item
            if checksum:=metadata["weights_sha256"]:
                artifact=PATCH/"isolated-role-readout.safetensors"
                assert digest(artifact)==checksum
                files["reproduce/checkpoints/"+artifact.name]=artifact
    fusion=trials/"sentence-readout-fusion.json"
    fusion_freeze=trials/"sentence-readout-fusion-frozen.json"
    if fusion.is_file() and fusion_freeze.is_file():
        proof=json.loads(fusion.read_text())
        metadata=json.loads(fusion_freeze.read_text())
        if "test" in proof["splits"] and proof["freeze_sha256"]==digest(fusion_freeze):
            files["reproduce/sentence_readout_fusion_write.py"]=ROOT/"sentence_readout_fusion_write.py"
            for item in (fusion,fusion_freeze):
                files["evidence/"+item.name]=item
            for name,checksum in metadata["fitted_weights_sha256"].items():
                artifact=PATCH/name
                assert digest(artifact)==checksum
                files["reproduce/checkpoints/"+name]=artifact
    bert=trials/"sentence-bert-referral.json"
    bert_freeze=trials/"sentence-bert-referral-frozen.json"
    if bert.is_file() and bert_freeze.is_file():
        proof=json.loads(bert.read_text())
        plan=json.loads(bert_freeze.read_text())
        if proof["freeze_sha256"]==digest(bert_freeze):
            files["reproduce/sentence_bert_referral_write.py"]=ROOT/"sentence_bert_referral_write.py"
            for item in (bert,bert_freeze):
                files["evidence/"+item.name]=item
            for i,(name,expected) in enumerate(plan["checkpoints"].items()):
                artifact=PATCH/f"sentence-bert-referral-{i}.safetensors"
                assert digest(artifact)==expected
                files["reproduce/checkpoints/"+artifact.name]=artifact
            for name,expected in plan.get("direct_classifier_tensors",{}).items():
                artifact=PATCH/name
                assert digest(artifact)==expected
                files["reproduce/checkpoints/"+artifact.name]=artifact
            files["evidence/sentence-bert-referral-encoder-source.json"]=ROOT/"runtime/openvino-bert-fp32/source.json"
            files["reproduce/openvino_worker.py"]=ROOT/"openvino_worker.py"
            parity=trials/"sentence-bert-referral-backend-parity.json"
            if parity.is_file() and json.loads(parity.read_text())["frozen_sha256"]==digest(bert_freeze):
                files["evidence/"+parity.name]=parity
            cli=trials/"sentence-bert-referral-cli-qa.json"
            if cli.is_file() and json.loads(cli.read_text())["freeze_sha256"]==digest(bert_freeze):
                files["evidence/"+cli.name]=cli
                files["reproduce/verify_sentence_bert_referral_cli.py"]=ROOT/"verify_sentence_bert_referral_cli.py"
    dense=trials/"sentence-dense-readout.json"
    dense_plan=trials/"sentence-dense-readout-frozen.json"
    if dense.is_file() and dense_plan.is_file():
        proof=json.loads(dense.read_text())
        meta=json.loads(dense_plan.read_text())
        if proof["freeze_sha256"]==digest(dense_plan):
            prospect=trials/"sentence-dense-readout-prospective.json"
            assert digest(prospect)==meta["prospective_sha256"]
            for record in (dense,dense_plan,prospect):
                files["evidence/"+record.name]=record
            files["reproduce/sentence_dense_readout_write.py"]=ROOT/"sentence_dense_readout_write.py"
            for i,arm in enumerate(meta["arms"].values()):
                artifact=PATCH/f"sentence-dense-readout-{i}.safetensors"
                assert digest(artifact)==arm["sha256"]
                files["reproduce/checkpoints/"+artifact.name]=artifact
            unguarded=trials/"sentence-dense-unrestricted.json"
            unguarded_plan=trials/"sentence-dense-unrestricted-frozen.json"
            if unguarded.is_file() and unguarded_plan.is_file():
                q=json.loads(unguarded_plan.read_text())
                assert q["dense_freeze_sha256"]==digest(dense_plan)
                assert json.loads(unguarded.read_text())["freeze_sha256"]==digest(unguarded_plan)
                artifact=PATCH/"sentence-dense-unrestricted.safetensors"
                assert digest(artifact)==q["weights_sha256"]
                files["reproduce/checkpoints/"+artifact.name]=artifact
                files["reproduce/sentence_dense_unrestricted.py"]=ROOT/"sentence_dense_unrestricted.py"
                for record in (unguarded,unguarded_plan):
                    files["evidence/"+record.name]=record
            guard=trials/"sentence-dense-issue-guard.json"
            guard_plan=trials/"sentence-dense-issue-guard-frozen.json"
            if guard.is_file() and guard_plan.is_file():
                q=json.loads(guard_plan.read_text())
                reserved_guard=trials/"sentence-dense-readout-role-guard.json"
                assert q["heldout_sha256"]==digest(reserved_guard)
                assert q["frozen_dense_sha256"]==digest(dense_plan)
                assert json.loads(guard.read_text())["freeze_sha256"]==digest(guard_plan)
                for record in (guard,guard_plan,reserved_guard):
                    files["evidence/"+record.name]=record
                files["reproduce/sentence_dense_issue_guard.py"]=ROOT/"sentence_dense_issue_guard.py"
            live=trials/"sentence-dense-readout-cli-qa.json"
            if live.is_file() and json.loads(live.read_text())["freeze_sha256"]==digest(dense_plan):
                files["evidence/"+live.name]=live
                files["reproduce/verify_sentence_dense_cli.py"]=ROOT/"verify_sentence_dense_cli.py"
            merge=trials/"sentence-dense-referral.json"
            merge_freeze=trials/"sentence-dense-referral-frozen.json"
            if merge.is_file() and merge_freeze.is_file() and json.loads(merge.read_text())["freeze_sha256"]==digest(merge_freeze):
                files["reproduce/sentence_dense_referral_combine.py"]=ROOT/"sentence_dense_referral_combine.py"
                for record in (merge,merge_freeze):
                    files["evidence/"+record.name]=record
    recurrent=trials/"sentence-recurrent-inverse.json"
    recurrent_plan=trials/"sentence-recurrent-inverse-frozen.json"
    if recurrent.is_file() and recurrent_plan.is_file():
        proof=json.loads(recurrent.read_text())
        plan=json.loads(recurrent_plan.read_text())
        if proof["freeze_sha256"]==digest(recurrent_plan):
            files["reproduce/sentence_recurrent_inverse.py"]=ROOT/"sentence_recurrent_inverse.py"
            for source in (recurrent,recurrent_plan):
                files["evidence/"+source.name]=source
            for i,v in enumerate(plan["arms"].values()):
                artifact=PATCH/f"sentence-recurrent-inverse-{i}.safetensors"
                assert digest(artifact)==v["sha256"]
                files["reproduce/checkpoints/"+artifact.name]=artifact
            qa=trials/"sentence-recurrent-inverse-weight-qa.json"
            if qa.is_file() and json.loads(qa.read_text())["freeze_sha256"]==digest(recurrent_plan):
                files["evidence/"+qa.name]=qa
                files["reproduce/verify_sentence_recurrent_weights.py"]=ROOT/"verify_sentence_recurrent_weights.py"
            refit=trials/"sentence-recurrent-refit.json"
            refit_plan=trials/"sentence-recurrent-refit-frozen.json"
            if refit.is_file() and refit_plan.is_file():
                receipt=json.loads(refit_plan.read_text())
                assert receipt["source_gate_plan_sha256"]==digest(recurrent_plan)
                assert json.loads(refit.read_text())["freeze_sha256"]==digest(refit_plan)
                files["reproduce/sentence_recurrent_refit.py"]=ROOT/"sentence_recurrent_refit.py"
                for record in (refit,refit_plan):
                    files["evidence/"+record.name]=record
                for i,v in enumerate(receipt["weights_sha256"].values()):
                    a=PATCH/f"sentence-recurrent-refit-{i}.safetensors"
                    assert digest(a)==v
                    files["reproduce/checkpoints/"+a.name]=a
            context=trials/"sentence-recurrent-context.json"
            if context.is_file() and json.loads(context.read_text())["weight_freeze_sha256"]==digest(recurrent_plan):
                files["evidence/"+context.name]=context
                files["reproduce/sentence_recurrent_context.py"]=ROOT/"sentence_recurrent_context.py"
    small=trials/"legal-bert-small-referral.json"
    small_plan=trials/"legal-bert-small-referral-frozen.json"
    if small.is_file() and small_plan.is_file():
        assert json.loads(small.read_text())["freeze_sha256"]==digest(small_plan)
        sizes=json.loads(small_plan.read_text())
        provenance=ROOT/"references/legal-bert-small/provenance.json"
        assert json.loads(provenance.read_text())["encoder_sha256"]==sizes["backbone_sha256"]
        for record in (small,small_plan):
            files["evidence/"+record.name]=record
        files["evidence/legal-bert-small-source.json"]=provenance
        for name in ("setup_legal_bert_small.py","legal_bert_small_referral.py"):
            files["reproduce/"+name]=ROOT/name
        qa=trials/"legal-bert-small-referral-qa.json"
        if qa.is_file() and json.loads(qa.read_text())["freeze_sha256"]==digest(small_plan):
            files["evidence/"+qa.name]=qa
            files["reproduce/verify_legal_bert_small_referral.py"]=ROOT/"verify_legal_bert_small_referral.py"
        for i,arm in enumerate(sizes["arms"].values()):
            artifact=PATCH/f"legal-bert-small-referral-{i}.safetensors"
            assert digest(artifact)==arm["weights_sha256"]
            files["reproduce/checkpoints/"+artifact.name]=artifact
    input_audit=trials/"sentence-identical-input-audit.json"
    if input_audit.is_file():
        proof=json.loads(input_audit.read_text())
        assert proof["source_sha256"]==digest(ROOT/"references/official/test.csv")
        files["evidence/"+input_audit.name]=input_audit
        files["reproduce/audit_identical_sentence_context.py"]=ROOT/"audit_identical_sentence_context.py"
    phrase=trials/"sentence-phrase-readout.json"
    phrase_plan=trials/"sentence-phrase-readout-frozen.json"
    if phrase.is_file() and phrase_plan.is_file():
        q=json.loads(phrase.read_text())
        recipe=json.loads(phrase_plan.read_text())
        assert q["freeze_sha256"]==digest(phrase_plan)
        lex=trials/"sentence-phrase-readout-lexical-frozen.json"
        prospect=trials/"sentence-dense-readout-phrase-prospect.json"
        assert recipe["lexical_freeze_sha256"]==digest(lex) and recipe["prospect_sha256"]==digest(prospect)
        text_lex=json.loads(lex.read_text())
        for record in (phrase,phrase_plan,lex,prospect):
            files["evidence/"+record.name]=record
        assets={"sentence-phrase-readout-lexical.safetensors":text_lex["weights_sha256"],
                "sentence-phrase-readout-vocabulary.json":text_lex["vocabulary_sha256"]}
        for i,v in enumerate(recipe["arms"].values()):
            assets[f"sentence-phrase-readout-{i}.safetensors"]=v["weights_sha256"]
        for name,expected in assets.items():
            path=PATCH/name
            assert digest(path)==expected
            files["reproduce/checkpoints/"+path.name]=path
        files["reproduce/sentence_phrase_readout.py"]=ROOT/"sentence_phrase_readout.py"
        independent=trials/"sentence-phrase-standalone.json"
        independent_plan=trials/"sentence-phrase-standalone-frozen.json"
        if independent.is_file() and independent_plan.is_file():
            assert json.loads(independent.read_text())["freeze_sha256"]==digest(independent_plan)
            assert json.loads(independent_plan.read_text())["lexical_freeze_sha256"]==digest(lex)
            for record in (independent,independent_plan):
                files["evidence/"+record.name]=record
            files["reproduce/sentence_phrase_standalone.py"]=ROOT/"sentence_phrase_standalone.py"
        phrase_bert=trials/"sentence-phrase-referral.json"
        phrase_bert_freeze=trials/"sentence-phrase-referral-frozen.json"
        if phrase_bert.is_file() and phrase_bert_freeze.is_file():
            assert json.loads(phrase_bert.read_text())["freeze_sha256"]==digest(phrase_bert_freeze)
            assert json.loads(phrase_bert_freeze.read_text())["phrase_freeze_sha256"]==digest(phrase_plan)
            for record in (phrase_bert,phrase_bert_freeze):
                files["evidence/"+record.name]=record
            files["reproduce/sentence_phrase_referral.py"]=ROOT/"sentence_phrase_referral.py"
        lexical_qa=trials/"sentence-phrase-readout-lexical-qa.json"
        if lexical_qa.is_file():
            checked=json.loads(lexical_qa.read_text())
            assert checked["lexical_freeze_sha256"]==digest(lex)
            assert checked["file_sha256"]==text_lex["weights_sha256"]
            files["evidence/"+lexical_qa.name]=lexical_qa
            files["reproduce/verify_sentence_phrase_lexical.py"]=ROOT/"verify_sentence_phrase_lexical.py"
        short=trials/"sentence-phrase-short-validation.json"
        short_freeze=trials/"sentence-phrase-short-validation-frozen.json"
        if short.is_file() and short_freeze.is_file():
            assert json.loads(short.read_text())["freeze_sha256"]==digest(short_freeze)
            assert json.loads(short_freeze.read_text())["model_freeze_sha256"]==digest(phrase_plan)
            for record in (short,short_freeze):
                files["evidence/"+record.name]=record
            files["reproduce/sentence_phrase_short_validation.py"]=ROOT/"sentence_phrase_short_validation.py"
        for stem,source in (("sentence-phrase-context","sentence_phrase_context.py"),
                            ("sentence-phrase-readout-cli-qa","verify_sentence_phrase_cli.py")):
            record=trials/(stem+".json")
            if record.is_file() and json.loads(record.read_text())["freeze_sha256"]==digest(phrase_plan):
                files["evidence/"+record.name]=record
                files["reproduce/"+source]=ROOT/source
    sequence=trials/"sentence-phrase-sequence.json"
    seq_plan=trials/"sentence-phrase-sequence-frozen.json"
    if sequence.is_file() and seq_plan.is_file():
        seq_hash=digest(seq_plan)
        r=json.loads(sequence.read_text())
        spec=json.loads(seq_plan.read_text())
        assert seq_hash==r["freeze_sha256"]
        q=PATCH/"sentence-phrase-sequence.safetensors"
        assert digest(q)==spec["checkpoint_sha256"]
        files["reproduce/checkpoints/"+q.name]=q
        for record in (sequence,seq_plan):
            files["evidence/"+record.name]=record
        files["reproduce/sentence_phrase_sequence.py"]=ROOT/"sentence_phrase_sequence.py"
        for stem,source in (("sentence-phrase-sequence-raw","sentence_phrase_sequence_raw.py"),
                             ("sentence-phrase-sequence-cli-qa","verify_sentence_phrase_sequence_cli.py"),
                             ("sentence-phrase-span-archive-audit","audit_sentence_phrase_span_archive.py")):
            record=trials/(stem+".json")
            if record.is_file() and json.loads(record.read_text())["sequence_freeze_sha256"]==seq_hash:
                files["evidence/"+record.name]=record
                files["reproduce/"+source]=ROOT/source
        scrub=trials/"sentence-phrase-raw-input-qa.json"
        raw_transfer=trials/"sentence-phrase-sequence-raw.json"
        if scrub.is_file() and raw_transfer.is_file():
            assert json.loads(scrub.read_text())["source_raw_benchmark_sha256"]==digest(raw_transfer)
            files["evidence/"+scrub.name]=scrub
            files["reproduce/audit_raw_phrase_inputs.py"]=ROOT/"audit_raw_phrase_inputs.py"
        for stem,source in (("sentence-phrase-sequence-referral","sentence_phrase_sequence_referral.py"),
                             ("sentence-phrase-duration","sentence_phrase_duration.py")):
            record=trials/(stem+".json")
            freeze=trials/(stem+"-frozen.json")
            if record.is_file() and freeze.is_file():
                x=json.loads(freeze.read_text())
                assert x["source_sequence_sha256"]==seq_hash
                assert json.loads(record.read_text())["freeze_sha256"]==digest(freeze)
                for item in (record,freeze):
                    files["evidence/"+item.name]=item
                files["reproduce/"+source]=ROOT/source
                if stem=="sentence-phrase-duration":
                    q=PATCH/(stem+".safetensors")
                    assert digest(q)==x["weights_sha256"]
                    files["reproduce/checkpoints/"+q.name]=q
                    files["reproduce/packed_duration_decode.cpp"]=ROOT/"packed_duration_decode.cpp"
    pair_freeze=trials/"sentence-pairwise-repair-frozen.json"
    pair_model=PATCH/"sentence-pairwise-repair.safetensors"
    for stem,source in (("sentence-pairwise-repair","sentence_pairwise_repair.py"),
                         ("sentence-pairwise-consensus","sentence_pairwise_consensus.py")):
        freeze=trials/(stem+"-frozen.json")
        record=trials/(stem+".json")
        if freeze.is_file() and record.is_file():
            evidence=json.loads(record.read_text())
            if "test" not in evidence["splits"]:
                continue
            spec=json.loads(freeze.read_text())
            assert evidence["freeze_sha256"]==digest(freeze)
            if freeze==pair_freeze:
                assert digest(pair_model)==spec["checkpoint_sha256"]
                files["reproduce/checkpoints/"+pair_model.name]=pair_model
            else:
                assert digest(pair_freeze)==spec["repair_freeze_sha256"]
            for artifact in (freeze,record):
                files["evidence/"+artifact.name]=artifact
            files["reproduce/"+source]=ROOT/source
    pair_consensus=trials/"sentence-pairwise-consensus-frozen.json"
    if "reproduce/sentence_pairwise_repair.py" in files and "reproduce/sentence_pairwise_consensus.py" in files:
        files["reproduce/predict_pairwise_judgment.py"]=ROOT/"predict_pairwise_judgment.py"
        for stem,source in (("sentence-pairwise-cli-qa","verify_pairwise_judgment.py"),
                            ("sentence-pairwise-gemini-replay","compare_pairwise_gemini.py"),
                            ("sentence-pairwise-archive-audit","audit_pairwise_archive.py")):
            artifact=trials/(stem+".json")
            if artifact.is_file():
                spec=json.loads(artifact.read_text())
                assert spec["repair_freeze_sha256"]==digest(pair_freeze)
                if "consensus_freeze_sha256" in spec:
                    assert spec["consensus_freeze_sha256"]==digest(pair_consensus)
                files["evidence/"+artifact.name]=artifact
                files["reproduce/"+source]=ROOT/source
    jp_plan=trials/"sentence-joint-pairwise-frozen.json"
    jp_score=trials/"sentence-joint-pairwise.json"
    if jp_plan.is_file() and jp_score.is_file() and "test" in json.loads(jp_score.read_text())["splits"]:
        joint_freeze=digest(jp_plan)
        assert json.loads(jp_score.read_text())["freeze_sha256"]==joint_freeze
        jp_spec=json.loads(jp_plan.read_text())
        assert jp_spec["source_repair_freeze_sha256"]==digest(pair_freeze)
        assert jp_spec["source_sequence_freeze_sha256"]==digest(seq_plan)
        for artifact in (jp_plan,jp_score):
            files["evidence/"+artifact.name]=artifact
        files["reproduce/sentence_joint_pairwise.py"]=ROOT/"sentence_joint_pairwise.py"
        for name,source in (("raw-test","compare_joint_pairwise_raw.py"),("raw-cloud","compare_joint_pairwise_raw.py"),
                            ("cli-qa","verify_joint_pairwise_cli.py"),("span-archive","audit_joint_pairwise_span_archive.py")):
            artifact=trials/("sentence-joint-pairwise-"+name+".json")
            if artifact.is_file():
                assert json.loads(artifact.read_text())["joint_freeze_sha256"]==joint_freeze
                files["evidence/"+artifact.name]=artifact
                files["reproduce/"+source]=ROOT/source
    for stem,source,needed in (("pairwise-referral-coverage","audit_pairwise_referral_coverage.py","test"),
                               ("mixedbread-referral","mixedbread_referral_write.py","confirm"),
                               ("mixedbread-referral-overlay","mixedbread_referral_overlay.py","confirm"),
                               ("sentence-kernel-weight-write","sentence_kernel_weight_write.py","test")):
        freeze=trials/(stem+"-frozen.json")
        record=trials/(stem+".json")
        if not (freeze.is_file() and record.is_file()):
            continue
        evidence=json.loads(record.read_text())
        if needed not in evidence["splits"]:
            continue
        match=json.loads(freeze.read_text())
        assert evidence["freeze_sha256"]==digest(freeze)
        if "pairwise_freeze_sha256" in match:
            assert match["pairwise_freeze_sha256"]==digest(pair_freeze)
        if "repair_freeze_sha256" in match:
            assert match["repair_freeze_sha256"]==digest(pair_freeze)
        if stem=="mixedbread-referral-overlay":
            assert match["sources"]["mixedbread-referral"]==digest(trials/"mixedbread-referral-frozen.json")
            assert match["sources"]["sentence-pairwise-repair"]==digest(pair_freeze)
        for artifact in (freeze,record):
            files["evidence/"+artifact.name]=artifact
        files["reproduce/"+source]=ROOT/source
        trained={}
        if stem=="mixedbread-referral":
            files["reproduce/legal_bert_small_referral.py"]=ROOT/"legal_bert_small_referral.py"
            files["evidence/mixedbread-model-provenance.json"]=ROOT/"references/mxbai-embed-xsmall/provenance.json"
            trained={stem+f"-{i}":v["sha256"] for i,v in enumerate(match["heads"].values())}
        elif stem=="mixedbread-referral-overlay":
            trained[stem]=match["arms"]["combined readout"].get("checkpoint_sha256")
        elif stem=="sentence-kernel-weight-write":
            trained[stem]=match["checkpoint_sha256"]
        for key,sha in trained.items():
            if sha:
                artifact=PATCH/(key+".safetensors")
                assert digest(artifact)==sha
                files["reproduce/checkpoints/"+artifact.name]=artifact
    mixed_context=trials/"mixedbread-referral-context.json"
    if mixed_context.is_file() and "evidence/mixedbread-referral.json" in files:
        assert json.loads(mixed_context.read_text())["mixedbread_freeze_sha256"]==digest(trials/"mixedbread-referral-frozen.json")
        files["evidence/"+mixed_context.name]=mixed_context
        files["reproduce/audit_mixedbread_context.py"]=ROOT/"audit_mixedbread_context.py"
    for stem,source in (("sentence-phase-readout","sentence_phase_readout_write.py"),
                        ("sentence-full-corpus-lexical","sentence_full_corpus_lexical.py"),
                        ("sentence-trigram-transition","sentence_trigram_transition_write.py")):
        record=trials/(stem+".json")
        freeze=trials/(stem+"-frozen.json")
        if not (record.is_file() and freeze.is_file() and "test" in json.loads(record.read_text())["splits"]):
            continue
        assert json.loads(record.read_text())["freeze_sha256"]==digest(freeze)
        for artifact in (record,freeze):
            files["evidence/"+artifact.name]=artifact
        files["reproduce/"+source]=ROOT/source
        model=json.loads(freeze.read_text())
        if stem=="sentence-trigram-transition":
            assert digest(seq_plan)==model["source_seq_sha256"]
            tensor=PATCH/(stem+".safetensors")
            assert digest(tensor)==model["patch_sha256"]
            files["reproduce/checkpoints/"+tensor.name]=tensor
            cpp=ROOT/"packed_role_trigram.cpp"
            assert digest(cpp)==model["cpp_sha256"]
            files["reproduce/"+cpp.name]=cpp
        else:
            key="repair_freeze_sha256" if stem=="sentence-phase-readout" else "pairwise_freeze_sha256"
            assert model[key]==digest(pair_freeze)
            for index,spec in enumerate(model["arms"].values()):
                if spec.get("sha256"):
                    tensor=PATCH/(stem+f"-{index}.safetensors")
                    assert digest(tensor)==spec["sha256"]
                    files["reproduce/checkpoints/"+tensor.name]=tensor
        if stem=="sentence-full-corpus-lexical":
            mem=trials/(stem+"-memory-frozen.json")
            assert digest(mem)==model["memory_freeze_sha256"]
            tensor=PATCH/(stem+"-memory.safetensors")
            assert digest(tensor)==json.loads(mem.read_text())["model_sha256"]
            files["evidence/"+mem.name]=mem
            files["reproduce/checkpoints/"+tensor.name]=tensor
        hybrid=trials/(stem+"-bert.json")
        hybrid_freeze=trials/(stem+"-bert-frozen.json")
        if hybrid.is_file() and hybrid_freeze.is_file() and "test" in json.loads(hybrid.read_text())["splits"]:
            assert json.loads(hybrid.read_text())["freeze_sha256"]==digest(hybrid_freeze)
            assert json.loads(hybrid_freeze.read_text())["source_edit_sha256"]==digest(freeze)
            for artifact in (hybrid,hybrid_freeze):
                files["evidence/"+artifact.name]=artifact
            files["reproduce/compare_broad_edits_to_bert.py"]=ROOT/"compare_broad_edits_to_bert.py"
    for stem in ("sentence-minilm-whitened","sentence-minilm-whitened-active"):
        minilm_record=trials/(stem+".json")
        minilm_freeze=trials/(stem+"-frozen.json")
        if not (minilm_record.is_file() and minilm_freeze.is_file() and "test" in json.loads(minilm_record.read_text())["splits"]):
            continue
        proof=json.loads(minilm_record.read_text())
        cfg=json.loads(minilm_freeze.read_text())
        assert proof["freeze_sha256"]==digest(minilm_freeze)
        assert proof["bert_overlay_sha256"]==digest(trials/"mixedbread-referral-overlay-frozen.json")
        ancestor=trials/"sentence-minilm-fusion-frozen.json"
        assert cfg["miniLM_source_freeze_sha256"]==digest(ancestor)
        files["evidence/"+ancestor.name]=ancestor
        assert cfg["pair_freeze_sha256"]==digest(pair_freeze)
        for artifact in (minilm_record,minilm_freeze):
            files["evidence/"+artifact.name]=artifact
        files["reproduce/sentence_minilm_whitened_write.py"]=ROOT/"sentence_minilm_whitened_write.py"
        for i,arm in enumerate(cfg["arms"].values()):
            if arm["sha256"]:
                path=PATCH/f"{stem}-{i}.safetensors"
                assert digest(path)==arm["sha256"]
                files["reproduce/checkpoints/"+path.name]=path
    utility_freeze=trials/"sentence-referral-utility-frozen.json"
    utility_record=trials/"sentence-referral-utility.json"
    if utility_freeze.is_file() and utility_record.is_file() and "test" in json.loads(utility_record.read_text())["splits"]:
        spec=json.loads(utility_freeze.read_text())
        assert json.loads(utility_record.read_text())["freeze_sha256"]==digest(utility_freeze)
        assert spec["pairwise_freeze_sha256"]==digest(pair_freeze)
        for artifact in (utility_record,utility_freeze):
            files["evidence/"+artifact.name]=artifact
        files["reproduce/sentence_referral_utility_write.py"]=ROOT/"sentence_referral_utility_write.py"
        if spec["patch_sha256"]:
            tensor=PATCH/"sentence-referral-utility.safetensors"
            assert digest(tensor)==spec["patch_sha256"]
            files["reproduce/checkpoints/"+tensor.name]=tensor
    for suffix in ("archive","minilm"):
        loc=trials/f"sentence-referral-actual-replay-{suffix}.json"
        if loc.is_file():
            proof=json.loads(loc.read_text())
            if suffix=="minilm" and "test" not in proof["splits"]:
                continue
            assert proof["router_freeze_sha256"]==digest(utility_freeze)
            files["evidence/"+loc.name]=loc
            files["reproduce/sentence_referral_actual_replay.py"]=ROOT/"sentence_referral_actual_replay.py"
    for stem,code in (("sentence-template-memory","sentence_template_memory_write.py"),
                      ("sentence-bert-reviewer-gain","sentence_bert_reviewer_gain.py")):
        freeze=trials/(stem+"-frozen.json")
        result=trials/(stem+".json")
        if not (freeze.is_file() and result.is_file() and "test" in json.loads(result.read_text())["splits"]):
            continue
        config=json.loads(freeze.read_text())
        assert json.loads(result.read_text())["freeze_sha256"]==digest(freeze)
        for item in (freeze,result):
            files["evidence/"+item.name]=item
        files["reproduce/"+code]=ROOT/code
        if stem=="sentence-template-memory":
            prov=trials/(stem+"-source.json")
            assert config["pairwise_freeze_sha256"]==digest(pair_freeze)
            assert config["source_sha256"]==digest(prov)
            files["evidence/"+prov.name]=prov
            weight=json.loads(prov.read_text())["weights_sha256"]
        else:
            for name,sha in config["source_sha256"].items():
                assert digest(trials/(name+"-frozen.json"))==sha
            weight=config["weights_sha256"]
        if weight:
            tensor=PATCH/(stem+".safetensors")
            assert digest(tensor)==weight
            files["reproduce/checkpoints/"+tensor.name]=tensor
    template_frozen=trials/"sentence-template-memory-frozen.json"
    for name in ("sentence-template-memory-bert","sentence-template-memory-cli-qa"):
        path=trials/(name+".json")
        if path.is_file() and "evidence/sentence-template-memory.json" in files:
            data=json.loads(path.read_text())
            assert data["freeze_sha256"]==digest(template_frozen)
            if "splits" in data and "test" not in data["splits"]:
                continue
            files["evidence/"+path.name]=path
            if name.endswith("cli-qa"):
                files["reproduce/verify_template_memory_cli.py"]=ROOT/"verify_template_memory_cli.py"
    wide=trials/"sentence-wide-bert-benchmark.json"
    prevalence=trials/"sentence-role-prevalence.json"
    if prevalence.is_file():
        proof=json.loads(prevalence.read_text())
        freeze=trials/"sentence-role-prevalence-frozen.json"
        policy=json.loads(freeze.read_text())
        tensor=PATCH/"sentence-role-prevalence.safetensors"
        assert proof["freeze_sha256"]==digest(freeze)
        assert policy["model_sha256"]==digest(tensor)
        assert policy["parent_sha256"]==digest(pair_freeze)
        for item in (prevalence,freeze):
            files["evidence/"+item.name]=item
        files["reproduce/checkpoints/"+tensor.name]=tensor
        files["reproduce/sentence_role_prevalence_write.py"]=ROOT/"sentence_role_prevalence_write.py"
    headroom=trials/"sentence-role-pair-headroom.json"
    if headroom.is_file():
        proof=json.loads(headroom.read_text())
        assert proof["source_sha256"]==digest(trials/proof["source"])
        assert "evidence/"+proof["source"] in files
        files["evidence/"+headroom.name]=headroom
        files["reproduce/audit_role_pair_headroom.py"]=ROOT/"audit_role_pair_headroom.py"
    fuzzy=trials/"sentence-fuzzy-pair-memory.json"
    if fuzzy.is_file():
        freeze=trials/"sentence-fuzzy-pair-memory-frozen.json"
        policy=json.loads(freeze.read_text())
        proof=json.loads(fuzzy.read_text())
        assert proof["freeze_sha256"]==digest(freeze)
        assert policy["parent_sha256"]==digest(pair_freeze)
        assert policy["memory"]["dataset_sha256"]==digest(trials/"sentence-template-memory-source.json")
        cache=ROOT/"cache/sentence-fuzzy-pair-memory"
        assert policy["memory"]["row_file_sha256"]==digest(cache/"rows.npz")
        assert policy["memory"]["coefficient_file_sha256"]==digest(cache/"coefficients.npz")
        for item in (fuzzy,freeze):
            files["evidence/"+item.name]=item
        files["reproduce/sentence_fuzzy_pair_memory.py"]=ROOT/"sentence_fuzzy_pair_memory.py"
    if wide.is_file():
        run=json.loads(wide.read_text())
        assert run["method"]["router_freeze_sha256"]==digest(utility_freeze)
        assert run["method"]["bert_freeze_sha256"]==digest(trials/"sentence-bert-referral-frozen.json")
        assert run["method"]["local_overlay_sha256"]==digest(trials/"mixedbread-referral-overlay-frozen.json")
        files["evidence/"+wide.name]=wide
        files["reproduce/sentence_wide_bert_benchmark.py"]=ROOT/"sentence_wide_bert_benchmark.py"
        live_cli=trials/"sentence-wide-bert-cli-qa.json"
        if live_cli.is_file():
            checked=json.loads(live_cli.read_text())
            assert checked["method"]==run["method"]
            assert len(checked["cases"])>=3 and all(q["changed"]>0 for q in checked["cases"])
            files["evidence/"+live_cli.name]=live_cli
            for code in ("predict_wide_bert_review.py","verify_wide_bert_cli.py","sentence_phrase_sequence.py"):
                files["reproduce/"+code]=ROOT/code
        budget=trials/"sentence-wide-bert-budget.json"
        if budget.is_file():
            small=json.loads(budget.read_text())
            assert small["source_method"]==run["method"]
            for split,trial in small["splits"].items():
                for route,curve in trial["routes"].items():
                    assert curve["0.2"]["labels"]==run["splits"][split]["comparison"]["routes"][route]["labels"]
            files["evidence/"+budget.name]=budget
            files["reproduce/audit_wide_bert_budget.py"]=ROOT/"audit_wide_bert_budget.py"
        multiclass=trials/"sentence-wide-bert-multiclass.json"
        if multiclass.is_file():
            freeze=trials/"sentence-wide-bert-multiclass-frozen.json"
            policy=json.loads(freeze.read_text())
            proof=json.loads(multiclass.read_text())
            assert proof["freeze_sha256"]==digest(freeze)
            assert policy["wide_method"]==run["method"]
            for name,sha in policy["dependencies"].items():
                assert digest(trials/(name+"-frozen.json"))==sha
            for split,trial in proof["splits"].items():
                upstream=run["splits"][split]
                assert trial["embedding_sha256"]==upstream["embedding_provenance"]["embedding_sha256"]
                assert trial["parent"]==upstream["comparison"]["routes"]["confidence"]["labels"]
            for item in (multiclass,freeze):
                files["evidence/"+item.name]=item
            if policy["weights_sha256"]:
                tensor=PATCH/"sentence-wide-bert-multiclass.safetensors"
                assert digest(tensor)==policy["weights_sha256"]
                files["reproduce/checkpoints/"+tensor.name]=tensor
            files["reproduce/sentence_wide_bert_multiclass_write.py"]=ROOT/"sentence_wide_bert_multiclass_write.py"
            files["reproduce/sentence_bert_reviewer_gain.py"]=ROOT/"sentence_bert_reviewer_gain.py"
        composition=trials/"sentence-wide-bert-template.json"
        if composition.is_file():
            proof=json.loads(composition.read_text())
            assert proof["method"]["wide_method"]==run["method"]
            assert proof["method"]["template_freeze_sha256"]==digest(template_frozen)
            assert proof["method"]["template_source_sha256"]==digest(trials/"sentence-template-memory-source.json")
            for split,trial in proof["splits"].items():
                upstream=run["splits"][split]
                assert trial["embedding_sha256"]==upstream["embedding_provenance"]["embedding_sha256"]
                assert trial["rates"]["0.2"]["base"]==upstream["comparison"]["routes"]["confidence"]["labels"]
            files["evidence/"+composition.name]=composition
            files["reproduce/audit_wide_bert_template_composition.py"]=ROOT/"audit_wide_bert_template_composition.py"
    quant=trials/"sentence-bert-int8-review.json"
    if quant.is_file():
        proof=json.loads(quant.read_text())
        prov=ROOT/"runtime/sentence-bert-int8-review/provenance.json"
        bert_freeze=trials/"sentence-bert-referral-frozen.json"
        assert proof["bert_freeze_sha256"]==digest(bert_freeze)
        assert proof["overlay_freeze_sha256"]==digest(trials/"mixedbread-referral-overlay-frozen.json")
        assert proof["graph"]==json.loads(prov.read_text())
        assert proof["graph"]["source_weights_sha256"]==json.loads(bert_freeze.read_text())["backbone_weights_sha256"]
        files["evidence/"+quant.name]=quant
        files["evidence/sentence-bert-int8-review-graph-provenance.json"]=prov
        files["reproduce/bench_bert_int8_review.py"]=ROOT/"bench_bert_int8_review.py"
        precision=trials/"sentence-bert-int8-precision.json"
        if precision.is_file():
            diagnostic=json.loads(precision.read_text())
            assert diagnostic["source"]==proof["graph"]
            assert diagnostic["results"]["unsigned-reduced"]["graph_sha256"]==proof["graph"]["quantized_sha256"]
            files["evidence/"+precision.name]=precision
            files["reproduce/diagnose_bert_int8_precision.py"]=ROOT/"diagnose_bert_int8_precision.py"
    if (edge/"results.json").is_file():
        edge_scores=json.loads((edge/"results.json").read_text())["splits"].get("test",{})
        for name in ("results.json","frozen.json","train.json","text-disjoint-validation.json","independent-replay.json",
                     "per-case-published-comparison.npz","qa.json","overlap.json"):
            artifact=edge/name
            if artifact.is_file():files["evidence/role-edge/"+name]=artifact
        def improves(profile):
            return (profile["cases"] == 712 and profile["paired_gap_95pct"]["exact_span_f1"][0] > 0)
        qualifies=(edge/"qa.json").is_file() and set(edge_scores)=={"exact","balanced"} and all(map(improves,edge_scores.values()))
        if qualifies:
            for name in ("predict_role_edge.py","role_edge_runtime.py","deployment.json","vocab.json",
                         "role-edge-exact.safetensors","role-edge-balanced.safetensors","method.txt"):
                files["role-edge/"+name]=edge/name
            for name in ("run_role_edge.py","role_edge_followup.py","role_edge_qa.py","role_edge_export.py",
                         "role_edge_disjoint.py","role_edge_overlap.py","role_edge_runtime.py","predict_role_edge.py"):
                files["reproduce/astra-role-edge/"+name]=edge/name
            helper=ROOT/"astra-run-penalty"
            for name in ("run_entry.py","run_entry_runtime.py"):
                if (helper/name).is_file():files["reproduce/astra-run-penalty/"+name]=helper/name
            qa=trials/"role-edge-interpolation-qa.json"
            patch=PATCH/"whole-role-edge-interpolated-exact.safetensors"
            deploy=json.loads((edge/"deployment.json").read_text())
            if qa.is_file() and patch.is_file():
                checked=json.loads(qa.read_text())
                intermediate=checked["cases_replayed"]==712 and checked["sha256"]==digest(patch)
            if intermediate:
                deploy["profiles"]["interpolated"]={**deploy["profiles"]["exact"],
                    "file":"role-edge-interpolated.safetensors","sha256":digest(patch),"gain":.35}
                files["role-edge/role-edge-interpolated.safetensors"]=patch
            sentence=trials/"known-sentence-pair-edit-qa.json"
            sentence_weights=PATCH/"known-sentence-role-pair.safetensors"
            if sentence.is_file() and sentence_weights.is_file():
                checked=json.loads(sentence.read_text())
                if (checked["case_count"]==712 and checked["sha256"]==digest(sentence_weights)
                    and checked["gap_ci95_macro_f1_accuracy"][0][0]>0):
                    sentence_profile=checked["scores"]
                    deploy["profiles"]["sentence-context"]={**deploy["profiles"]["balanced"],
                        "file":"role-edge-sentence-context.safetensors","sha256":digest(sentence_weights),
                        "input_format":"sentences"}
                    files["role-edge/role-edge-sentence-context.safetensors"]=sentence_weights
            patched=PATCH/"known-sentence-label-change.safetensors"
            patch_receipt=trials/"sentence-label-change-qa.json"
            if sentence_profile and patched.is_file() and patch_receipt.is_file():
                checked=json.loads(patch_receipt.read_text())
                if (checked["cases"]==712 and checked["weights_sha256"]==digest(patched)
                    and checked["paired_to_preceding"]["macro_f1"][0]>0):
                    calibrated_profile=checked["scores"]
                    spec=deploy["profiles"]["sentence-context"]
                    deploy["profiles"]["sentence-calibrated"]={**spec,
                        "file":"role-edge-sentence-calibrated.safetensors","sha256":digest(patched),
                        "tensors":{**spec["tensors"],"emission.weight":[7,7],"emission.bias":[7],
                                   "label_change.utility":[7,7,5],"label_change.margin_bins":[4],
                                   "label_change.threshold":[]}}
                    files["role-edge/role-edge-sentence-calibrated.safetensors"]=patched
            routed=trials/"sentence-uncertainty-qa.json"
            routed_weights=PATCH/"known-sentence-uncertainty.safetensors"
            routed_freeze=trials/"sentence-minilm-token-table-frozen.json"
            if calibrated_profile and all(p.is_file() for p in (routed,routed_weights,routed_freeze)):
                checked=json.loads(routed.read_text())
                files["evidence/"+routed.name]=routed
                scores=checked["scores"]
                if (checked["cases"]==712 and checked["weights_sha256"]==digest(routed_weights)
                    and checked["fit_freeze_sha256"]==digest(routed_freeze)
                    and checked["paired_to_context"]["macro_f1"][0]>0
                    and checked["paired_to_calibrated"]["accuracy"][0]>0
                    and scores["accuracy"]>=checked["context"]["accuracy"]):
                    uncertainty_profile=scores
                    spec=deploy["profiles"]["sentence-calibrated"]
                    deploy["profiles"]["sentence-uncertainty"]={**spec,
                        "file":"role-edge-sentence-uncertainty.safetensors","sha256":digest(routed_weights),
                        "tensors":{**spec["tensors"],"uncertainty.center":[44],"uncertainty.scale":[44],
                                   "uncertainty.weight":[44,7],"uncertainty.bias":[7],"uncertainty.settings":[3]}}
                    files["role-edge/role-edge-sentence-uncertainty.safetensors"]=routed_weights
                    for name in ("sentence_uncertainty_runtime.py","span_posterior_decode.py"):
                        files[name]=ROOT/name
            joint_qa=trials/"sentence-joint-qa.json"
            joint_portable=trials/"sentence-joint-portable-qa.json"
            joint_result=trials/"sentence-joint-edit.json"
            joint_weights=PATCH/"known-sentence-joint-portable.safetensors"
            if uncertainty_profile and all(p.is_file() for p in (joint_qa,joint_portable,joint_result,joint_weights)):
                checked=json.loads(joint_qa.read_text())
                live=json.loads(joint_portable.read_text())
                study=json.loads(joint_result.read_text())
                held=study["splits"].get("test")
                if (held and checked["cases"]==held["cases"]==712 and live["weights_sha256"]==digest(joint_weights)
                    and checked["weights_sha256"]==live["packed_weights_sha256"]
                    and checked["freeze_sha256"]==live["freeze_sha256"]==study["freeze_sha256"]
                    and checked["scores"]==held["candidate"] and checked["paired_to_preceding"]["macro_f1"][0]>0):
                    joint_profile=checked["scores"]
                    spec=deploy["profiles"]["sentence-uncertainty"]
                    shape={"joint.denominator":[7],"joint.ref_f1":[7],"joint.transition":[7,7],
                           "joint.settings":[4],"joint.mode":[]}
                    mode=study["choice"][0]
                    if mode!="posterior":
                        width,heads=(258,7) if mode=="context" else (58,1)
                        shape.update({"joint.center":[width],"joint.scale":[width],"joint.weight":[heads,width,7],
                                      "joint.bias":[heads,7]})
                    deploy["profiles"]["sentence-joint"]={**spec,"file":"role-edge-sentence-joint.safetensors",
                       "sha256":digest(joint_weights),"tensors":{**spec["tensors"],**shape}}
                    files["role-edge/role-edge-sentence-joint.safetensors"]=joint_weights
                    files["sentence_joint_runtime.py"]=ROOT/"sentence_joint_runtime.py"
            local_qa=trials/"sentence-local-readout-portable-qa.json"
            local_comparison=trials/"sentence-document-graph-comparison.json"
            local_weights=PATCH/"sentence-local-readout-portable.safetensors"
            if joint_profile and all(p.is_file() for p in (local_qa,local_comparison,local_weights)):
                checked=json.loads(local_qa.read_text())
                study=json.loads(local_comparison.read_text())
                held=study["splits"].get("test",{}).get("arms",{}).get("local")
                width=checked["width"]
                if (held and checked["scores"]==held["scores"] and checked["official_cases"]==712
                    and checked["weights_sha256"]==digest(local_weights)
                    and checked["freeze_sha256"]==study["freeze_sha256"]
                    and checked["source_sha256"]==digest(PATCH/"sentence-document-graph.safetensors")
                    and len(checked["live_cases"])>=3 and held["paired_to_baseline"]["macro_f1"][0]>0
                    and held["paired_to_baseline"]["accuracy"][0]>0):
                    local_profile=checked["scores"]
                    spec=deploy["profiles"]["sentence-joint"]
                    shape={"local_readout.center":[width],"local_readout.scale":[width],
                           "local_readout.matrix":[width,7],"local_readout.bias":[7],"local_readout.settings":[3]}
                    deploy["profiles"]["sentence-local"]={**spec,"file":"role-edge-sentence-local.safetensors",
                        "sha256":digest(local_weights),"tensors":{**spec["tensors"],**shape}}
                    files["role-edge/role-edge-sentence-local.safetensors"]=local_weights
                    files["sentence_local_readout_runtime.py"]=ROOT/"sentence_local_readout_runtime.py"
                    files["evidence/"+local_qa.name]=local_qa
            raw_receipt=trials/"raw-joint-direct-qa.json"
            raw_study=trials/"whole-joint-transfer-edit.json"
            if joint_profile and raw_receipt.is_file() and raw_study.is_file():
                raw=json.loads(raw_receipt.read_text())
                held=json.loads(raw_study.read_text())["splits"].get("test")
                if (held and raw["cases"]==held["cases"]==712 and raw["weights_sha256"]==digest(joint_weights)
                    and raw["scores"]==held["standalone_sentence_joint"] and raw["verified_live_cases"]
                    and raw["paired_to_token_balanced"]["token_accuracy"][0]>0
                    and raw["paired_to_role_edge_balanced"]["token_accuracy"][0]>0):
                    raw_joint_profile=raw["scores"]
                    deploy["profiles"]["raw-joint"]={**deploy["profiles"]["sentence-joint"],"input_format":"raw"}
            raw_local_evidence=trials/"raw-local-transfer.json"
            if local_profile and raw_joint_profile and raw_local_evidence.is_file():
                raw=json.loads(raw_local_evidence.read_text())
                if (raw["cases"]==712 and raw["parent"]==raw_joint_profile and len(raw["live_cases"])>=3
                    and raw["local_weights_sha256"]==digest(local_weights)
                    and raw["joint_weights_sha256"]==digest(joint_weights)
                    and raw["paired"]["exact_span_f1"][0]>0 and raw["paired"]["token_accuracy"][0]>0):
                    raw_local_profile=raw["candidate"]
                    deploy["profiles"]["raw-local"]={**deploy["profiles"]["sentence-local"],"input_format":"raw"}
            doc_qa=trials/"sentence-document-readout-portable-qa.json"
            doc_weights=PATCH/"sentence-document-readout-portable.safetensors"
            if local_profile and all(p.is_file() for p in (doc_qa,doc_weights,reserved,reserved_freeze)):
                checked=json.loads(doc_qa.read_text())
                study=json.loads(local_comparison.read_text())
                held=study["splits"].get("test",{}).get("arms",{}).get("two-hop")
                fresh=json.loads(reserved.read_text())
                fresh_plan=json.loads(reserved_freeze.read_text())
                target=fresh["arms"]["two-hop"]
                if (held and checked["scores"]==held["scores"] and checked["official_cases"]==712
                    and checked["weights_sha256"]==digest(doc_weights) and len(checked["live_cases"])>=3
                    and checked["source_sha256"]==study["comparison_weights_sha256"]["two-hop"]
                    and checked["freeze_sha256"]==study["freeze_sha256"]==fresh_plan["graph_freeze_sha256"]
                    and fresh["freeze_sha256"]==digest(reserved_freeze) and fresh["cases"]==384
                    and held["paired_to_baseline"]["macro_f1"][0]>0
                    and target["paired_to_baseline"]["macro_f1"][0]>0 and target["paired_to_local"]["accuracy"][0]>0):
                    document_profile=checked["scores"]
                    width=checked["width"]
                    spec=deploy["profiles"]["sentence-local"]
                    shape={"local_readout.center":[width],"local_readout.scale":[width],
                           "local_readout.matrix":[width,7],"local_readout.two_hop":[]}
                    deploy["profiles"]["sentence-document"]={**spec,"file":"role-edge-sentence-document.safetensors",
                        "sha256":digest(doc_weights),"tensors":{**spec["tensors"],**shape}}
                    files["role-edge/role-edge-sentence-document.safetensors"]=doc_weights
                    files["sentence_graph_runtime.py"]=ROOT/"sentence_graph_runtime.py"
                    files["evidence/"+doc_qa.name]=doc_qa
            doc_calibrated_qa=trials/"sentence-document-calibrated-portable-qa.json"
            doc_calibrated_weights=PATCH/"sentence-document-calibrated-portable.safetensors"
            if document_profile and all(p.is_file() for p in (doc_calibrated_qa,doc_calibrated_weights,fusion,fusion_freeze)):
                checked=json.loads(doc_calibrated_qa.read_text())
                study=json.loads(fusion.read_text())
                freeze=json.loads(fusion_freeze.read_text())
                test=study["splits"]["test"]["arms"]["document only"]
                hold=study["splits"]["holdout"]["arms"]["document only"]
                if (checked["cases"]==712 and len(checked["live_cases"])>=3
                    and checked["scores"]==test["scores"] and checked["parent_sha256"]==digest(doc_weights)
                    and checked["weights_sha256"]==digest(doc_calibrated_weights)
                    and checked["freeze_sha256"]==study["freeze_sha256"]==digest(fusion_freeze)
                    and checked["fit_weights_sha256"]==freeze["fitted_weights_sha256"]["sentence-readout-fusion-document-only.safetensors"]
                    and test["paired_to_base"]["accuracy"][0]>0 and hold["paired_to_base"]["accuracy"][0]>0):
                    document_calibrated_profile=checked["scores"]
                    spec=deploy["profiles"]["sentence-document"]
                    name="role-edge-sentence-document-calibrated.safetensors"
                    deploy["profiles"]["sentence-document-calibrated"]={**spec,"file":name,
                        "sha256":digest(doc_calibrated_weights),"tensors":{**spec["tensors"],**checked["tensors"]}}
                    files["role-edge/"+name]=doc_calibrated_weights
                    files["evidence/"+doc_calibrated_qa.name]=doc_calibrated_qa
                    files["reproduce/verify_document_calibration_portable.py"]=ROOT/"verify_document_calibration_portable.py"
            measured=trials/"segment-weight-qa.json"
            segment_path=PATCH/"whole-segment-weights.safetensors"
            if measured.is_file() and segment_path.is_file():
                checked=json.loads(measured.read_text())
                if (checked["cases"]==712 and checked["weights_sha256"]==digest(segment_path)
                    and checked["gap_ci95"]["exact_span_f1"][0]>0):
                    segment_profile=checked["scores"]
                    deploy["profiles"]["segment-exact"]={
                        "file":"role-edge-segment-exact.safetensors","sha256":digest(segment_path),
                        "input_format":"raw","parent_profiles":["exact","balanced"],
                    }
                    files["role-edge/role-edge-segment-exact.safetensors"]=segment_path
                    for name in ("segment_span_runtime.py","span_posterior_decode.py"):
                        files[name]=ROOT/name
                    files["evidence/"+measured.name]=measured
            isolated_qa=trials/"isolated-role-readout-portable-qa.json"
            isolated_weights=PATCH/"sentence-isolated-portable.safetensors"
            if all(item.is_file() for item in (lone,lone_freeze,isolated_qa,isolated_weights)):
                study=json.loads(lone.read_text())
                checked=json.loads(isolated_qa.read_text())
                proof=study["splits"].get("test")
                reserve=study["splits"].get("holdout")
                if (proof and reserve and checked["scores"]==proof["candidate"] and checked["cases"]==712
                    and checked["freeze_sha256"]==study["freeze_sha256"]==digest(lone_freeze)
                    and checked["source_weights_sha256"]==digest(PATCH/"isolated-role-readout.safetensors")
                    and checked["weights_sha256"]==digest(isolated_weights) and len(checked["verified_live_cases"])>=3
                    and proof["paired"]["macro_f1"][0]>0 and proof["paired"]["accuracy"][0]>0
                    and reserve["paired"]["macro_f1"][0]>0):
                    isolated_profile=checked["scores"]
                    width=checked["width"]
                    deploy["profiles"]["sentence-isolated"]={"file":"role-edge-sentence-isolated.safetensors",
                        "sha256":digest(isolated_weights),"input_format":"sentences",
                        "tensors":{"isolated.center":[width],"isolated.scale":[width],"isolated.matrix":[width,7],
                                   "isolated.bias":[7],"isolated.prior_shift":[7],
                                   "isolated.lexical":[checked["lexical_width"],7]}}
                    files["role-edge/role-edge-sentence-isolated.safetensors"]=isolated_weights
                    files["isolated_role_readout_runtime.py"]=ROOT/"isolated_role_readout_runtime.py"
                    files["evidence/"+isolated_qa.name]=isolated_qa
            if intermediate or sentence_profile or calibrated_profile or segment_profile or uncertainty_profile or joint_profile or isolated_profile:
                updated=trials/"deployment-expanded.json"
                updated.write_text(json.dumps(deploy,indent=2)+"\n")
                files["role-edge/deployment.json"]=updated
    meta={"adapter_files":{k:digest(v) for k,v in files.items()},
          "base_model_files":{k:digest(PRETRAINED/k) for k in ("hier.safetensors","word2idx.json")},
          "base_distribution":"hierarchical/ in AttoFlow-judgment-role-edit.zip",
          "upstream":"https://huggingface.co/L-NLProc/LegalSeg_Hier_BiLSTM-CRF/tree/837c2a85ce9a6df21c9e3ef6d1b908b23b66d985/Hier_BiLSTM-CRF",
          "taxonomy_size":7,"model_parameters":16816110}
    readme="""AttoFlow whole-judgment span edits

The profiles require the hierarchical/ directory from the companion
AttoFlow judgment-role archive and Python with torch, numpy and safetensors.

python hier_model.py --model-dir /path/to/hierarchical-model --spans --profile token_balanced.safetensors --text-file judgment.txt
python hier_model.py --model-dir /path/to/hierarchical-model --spans --profile exact_span.safetensors --text-file judgment.txt

Both return seven-role character spans. Offsets are half-open Python Unicode indices.
In the 712-case raw test, token_balanced reached 0.260 exact span F1 and 61.6% token accuracy.
The exact_span profile reached 0.279 and 51.8%. The incumbent was 0.257 and 60.4%.
These measurements use English LegalSeg text reconstructed by joining the source rows.
Original publisher and licensing details accompany the separately downloaded base.
The reproduce folder holds experiment scripts. To rerun original model comparisons, merge it with the companion kit's reproduce directory and restore the documented official CSVs, weights and feature caches. The archive projection scripts can use the publisher CSVs directly.
The pinned base files and patch hashes are listed in manifest.json.
"""
    if "discourse_exact.safetensors" in files:
        readme+="""
The newer discourse profiles use the same base model with a written cue and position residual.

python predict_discourse_spans.py --model-dir /path/to/hierarchical --profile discourse_balanced.safetensors --text-file judgment.txt
python predict_discourse_spans.py --model-dir /path/to/hierarchical --profile discourse_exact.safetensors --text-file judgment.txt

The discourse_balanced profile gives 0.278 exact span F1 and 59.8% token accuracy on all 712 cases.
The discourse_exact profile gives 0.286 exact span F1 and 50.3% token accuracy on the same input.
Both are research checkpoints. Refer to the evidence reports for paired uncertainty and context.
"""
    if qualifies:
        role_pair="""The role-edge directory contains the newest whole-judgment profiles. They use the
same base and the original seven labels. Weights and vocabulary are checksum-pinned.

python role-edge/predict_role_edge.py judgment.txt --model-dir /path/to/hierarchical --profile exact
python role-edge/predict_role_edge.py judgment.txt --model-dir /path/to/hierarchical --profile balanced

The role-pair table is derived from counts in LegalSeg's public training split.
On the 712-case test the role-pair exact profile scores 0.301 exact-span F1
and 51.9% token accuracy. The balanced profile scores 0.288 and 60.6%.
The profiling and matched archive results are under evidence/role-edge.

"""
        if segment_profile:
            role_pair+=f"""The experimental segment-exact profile jointly selects labeled runs from raw text.
It scores {segment_profile['exact_labeled_spans']['f1']:.3f} exact span F1 and
{100*segment_profile['token_accuracy']:.1f}% token accuracy on the same 712 judgments.

python role-edge/predict_role_edge.py judgment.txt --model-dir /path/to/hierarchical --profile segment-exact

"""
        if intermediate:
            role_pair+="""An additional interpolation scores 0.300 exact-span F1 and 53.4% token accuracy.

python role-edge/predict_role_edge.py judgment.txt --model-dir /path/to/hierarchical --profile interpolated

"""
        if sentence_profile:
            role_pair+=f"""The sentence-context profile reads a JSON array of nonblank sentences from one judgment
and returns one label for each sentence in order. It scores {sentence_profile['macro_f1']:.3f} macro-F1 and
{100*sentence_profile['accuracy']:.1f}% sentence accuracy on the {sentence_profile['rows']:,} nonblank LegalSeg test rows.

python role-edge/predict_role_edge.py judgment-sentences.json --sentences --model-dir /path/to/hierarchical --profile sentence-context

"""
        if calibrated_profile:
            role_pair+=f"""The sentence-calibrated profile accepts the same JSON array and scores
{calibrated_profile['macro_f1']:.3f} macro-F1 and {100*calibrated_profile['accuracy']:.1f}% sentence accuracy.
Its reasoning F1 is {calibrated_profile['per_role_f1']['reasoning']:.3f}.

python role-edge/predict_role_edge.py judgment-sentences.json --sentences --model-dir /path/to/hierarchical --profile sentence-calibrated

"""
        if uncertainty_profile:
            role_pair+=f"""The sentence-uncertainty profile uses frozen hierarchy scores and a small
readout to correct low-confidence sentence predictions. It scored {uncertainty_profile['macro_f1']:.4f}
macro-F1 and {100*uncertainty_profile['accuracy']:.2f}% accuracy on the same official sentences.
The inference uses the existing hierarchy and the same JSON sentence input.

python role-edge/predict_role_edge.py judgment-sentences.json --sentences --model-dir /path/to/hierarchical --profile sentence-uncertainty

"""
        if joint_profile:
            role_pair+=f"""The sentence-joint profile applies posterior and learned transition
scores together to supplied sentences. It scored {joint_profile['macro_f1']:.5f} macro-F1
and {100*joint_profile['accuracy']:.2f}% sentence accuracy on the official test.
It uses the existing local hierarchy and returns one of the seven roles per input.

python role-edge/predict_role_edge.py judgment-sentences.json --sentences --model-dir /path/to/hierarchical --profile sentence-joint

"""
        if local_profile:
            role_pair+=f"""The sentence-local profile adds a small readout that uses the role scores
of the preceding and following sentences. It scored {local_profile['macro_f1']:.5f}
macro-F1 and {100*local_profile['accuracy']:.2f}% accuracy on the official sentence test.

python role-edge/predict_role_edge.py judgment-sentences.json --sentences --model-dir /path/to/hierarchical --profile sentence-local

"""
        if isolated_profile:
            role_pair+=f"""The sentence-isolated profile classifies each input sentence independently.
It scored {isolated_profile['macro_f1']:.5f} macro-F1 and
{100*isolated_profile['accuracy']:.2f}% accuracy on the official sentence-only test.

python role-edge/predict_role_edge.py sentences.json --sentences --model-dir /path/to/hierarchical --profile sentence-isolated

"""
        if document_profile:
            role_pair+=f"""The sentence-document profile adds two-hop document context to supplied
sentences. It scored {document_profile['macro_f1']:.5f} macro-F1 and
{100*document_profile['accuracy']:.2f}% accuracy on the official sentence test.

python role-edge/predict_role_edge.py judgment-sentences.json --sentences --model-dir /path/to/hierarchical --profile sentence-document

"""
        if document_calibrated_profile:
            role_pair+=f"""The sentence-document-calibrated profile adds a closed-form
calibration layer to the document scores. It scored {document_calibrated_profile['macro_f1']:.5f}
macro-F1 and {100*document_calibrated_profile['accuracy']:.2f}% accuracy on the official sentence test.
Its accuracy gain has a positive paired interval. Its macro-F1 interval spans zero.

python role-edge/predict_role_edge.py judgment-sentences.json --sentences --model-dir /path/to/hierarchical --profile sentence-document-calibrated

"""
        if raw_local_profile:
            role_pair+=f"""The raw-local profile uses the local readout on raw text. It scored
{100*raw_local_profile['token_accuracy']:.2f}% token accuracy and
{raw_local_profile['exact_labeled_spans']['f1']:.4f} exact-span F1 on official judgments.

python role-edge/predict_role_edge.py judgment.txt --model-dir /path/to/hierarchical --profile raw-local

"""
        if raw_joint_profile:
            role_pair+=f"""The raw-joint profile shares the sentence-joint weights and accepts raw
judgment text. It returns labeled Unicode spans. For full official judgments it
scored {100*raw_joint_profile['token_accuracy']:.2f}% token accuracy and
{raw_joint_profile['exact_labeled_spans']['f1']:.4f} exact-span F1.
Use segment-exact when matching gold boundaries is the priority.

python role-edge/predict_role_edge.py judgment.txt --model-dir /path/to/hierarchical --profile raw-joint

"""
        title,remainder=readme.split("\n\n",1)
        readme=title+"\n\n"+role_pair+remainder
    if "evidence/sentence-bert-referral.json" in files:
        readme+="""

The InLegalBERT referral checkpoints in reproduce/checkpoints are experimental
closed-form readouts. Their benchmark scripts use the pinned LegalSeg InLegalBERT
checkpoint in references/checkpoint and the OpenVINO encoder described by
evidence/sentence-bert-referral-encoder-source.json. The companion archive supplies
the encoder, cache and CSV loading utilities. The full judgment feeds the existing
document encoder. Referred rows also feed a 96-WordPiece BERT input.
These commands use the recorded FP32 Intel GPU backend. With a CPU runtime
installed, choose --device CPU to run the same experimental procedure.

python sentence_bert_referral_write.py encode fit select confirm --device GPU
python sentence_bert_referral_write.py choose
python sentence_bert_referral_write.py evaluate select confirm
python sentence_bert_referral_write.py encode holdout test --device GPU --mode 1
python sentence_bert_referral_write.py evaluate holdout test --mode 1

When the working directory and pinned weights are restored, the experimental
CPU interface accepts a JSON array of ordered nonblank sentences.

python sentence_bert_referral_write.py predict --input judgment-sentences.json --device CPU
"""
    if "evidence/sentence-wide-bert-budget.json" in files:
        readme+="""

The wider reviewer benchmark scores a fixed 20% confidence and utility review
budget. The audit replays smaller nested budgets from its completed FP32
feature caches. Restore the recorded benchmark environment before running.

python sentence_wide_bert_benchmark.py --device GPU
python audit_wide_bert_budget.py confirm
"""
    if "evidence/sentence-wide-bert-cli-qa.json" in files:
        readme+="""

The CPU command runs the 20% confidence review on a new judgment. It accepts
supplied sentences as a JSON array or raw text with --raw. The same pinned
OpenVINO assets used by the original referral command are required.

python predict_wide_bert_review.py --input judgment-sentences.json --device CPU
python predict_wide_bert_review.py --input judgment.txt --raw --device CPU
"""
    if "evidence/sentence-wide-bert-multiclass.json" in files:
        readme+="""

The seven-role reviewer correction uses cross-fitted training activations and
the FP32 features from the wider review benchmark. Evaluation uses the saved
matrix and the confidence route.

python sentence_wide_bert_multiclass_write.py fit select confirm
python sentence_wide_bert_multiclass_write.py evaluate test
"""
    if "evidence/sentence-wide-bert-template.json" in files:
        readme+="""

The memory composition replays the original template policy with the wider
FP32 reviewer, preserving accepted reviewer changes.

python audit_wide_bert_template_composition.py select confirm
"""
    if "evidence/sentence-role-prevalence.json" in files:
        readme+="""

The judgment-level prevalence write estimates corrections to seven role
frequencies from official training judgments. The paired comparison uses
the restored LegalSeg CSVs and the pinned sequence classifier.

python sentence_role_prevalence_write.py select confirm holdout test
"""
    if "evidence/sentence-role-pair-headroom.json" in files:
        readme+="""

The pairwise headroom audit uses the official confusion matrix of the
local InLegalBERT plus template-memory classifier.

python audit_role_pair_headroom.py
"""
    if "evidence/sentence-fuzzy-pair-memory.json" in files:
        readme+="""

The near-match experiment reconstructs a sparse character n-gram memory
from the restored official LegalSeg training CSV. It uses the same judgment
exclusions recorded in the exact-template source receipt.

python sentence_fuzzy_pair_memory.py select confirm
"""
    if "evidence/sentence-bert-int8-review.json" in files:
        readme+="""

The CPU reviewer diagnostic rebuilds FP32 and INT8 ONNX graphs from the pinned
checkpoint and compares them on the existing referral sets. It requires the
reference FP32 feature caches, PyTorch, ONNX and ONNX Runtime.

python bench_bert_int8_review.py confirm select --threads 4
"""
    if "evidence/sentence-bert-int8-precision.json" in files:
        readme+="""

The precision diagnostic compares signed and unsigned weight settings on the
same deterministic subset of 32 confirmation windows.

python diagnose_bert_int8_precision.py
"""
    if "evidence/sentence-dense-readout.json" in files:
        readme+="""

The dense readouts in reproduce/checkpoints use the frozen hierarchical CPU
encoder, its released document calibration, and the pinned lexical table. The
experimental predictor uses NumPy, SciPy, scikit-learn, safetensors and PyTorch.
After restoring the workspace inputs and weights described above, pass a JSON
array of ordered sentences from one judgment.

python sentence_dense_readout_write.py predict --arm nonlinear --input judgment-sentences.json

The --preserve-issues option keeps the previous document model’s issue
membership decisions. The report describes the official test and the trade-off
on a separate validation cohort.
"""
    if "evidence/sentence-recurrent-inverse.json" in files:
        readme+="""

The recurrent inverse experiment uses native candidate-state LSTM tensors in
reproduce/checkpoints. Its readout-only control, shared decoder settings and
matched readout refit are included with their benchmark evidence.

python sentence_recurrent_inverse.py prepare fit select confirm holdout test
python sentence_recurrent_inverse.py learn
python sentence_recurrent_inverse.py evaluate select confirm holdout test
python sentence_recurrent_context.py
python sentence_recurrent_refit.py confirm holdout test
"""
    if "evidence/sentence-phrase-readout.json" in files:
        readme+="""

The phrase experiment includes the fixed word and character vocabularies, the
lexical projection and the fitted dense readouts. The CPU command accepts an
ordered JSON array of nonblank sentences. It uses the same reference
hierarchy and lexical input files as the other experimental readouts.

python sentence_phrase_readout.py predict --arm "contextual phrases" --input judgment-sentences.json
"""
    if "evidence/sentence-phrase-sequence.json" in files:
        readme+="""

The sequence experiment provides two frozen objectives. Supplied-sentence mode
returns a label for each input and contiguous spans with half-open sentence
indices. Raw mode returns labeled spans with half-open character offsets over
the original UTF-8 input.

python sentence_phrase_sequence.py predict --profile spans --input judgment-sentences.json
python sentence_phrase_sequence.py predict --profile spans --input judgment.txt --raw
"""
    if "reproduce/predict_pairwise_judgment.py" in files:
        readme+="""

The pairwise experiment adds closed-form classification readouts to the frozen
sequence model. With the research working directory and the pinned assets
restored, the supplied-sentence CPU command returns labels and sentence spans.

python predict_pairwise_judgment.py --method direct --input judgment-sentences.json
python predict_pairwise_judgment.py --method consensus --input judgment-sentences.json
"""
    if "evidence/sentence-joint-pairwise-cli-qa.json" in files:
        readme+="""

The joint pairwise experiment applies the pair evidence inside the sequence
decoder. Both supplied sentences and raw text are supported.

python sentence_joint_pairwise.py predict --profile spans --input judgment-sentences.json
python sentence_joint_pairwise.py predict --profile spans --input judgment.txt --raw
"""
    if "evidence/mixedbread-referral.json" in files:
        readme+="""

The Mixedbread referral benchmark uses mxbai-embed-xsmall-v1 at revision
e6ac24e5d6efb8782b59de1647b3ececb4ece94e. Restore the files named in
evidence/mixedbread-model-provenance.json to references/mxbai-embed-xsmall.
Its 96-token inputs use the original BERT referral positions and up to two
earlier sentences. The pretrained Mixedbread weights are downloaded separately.

python mixedbread_referral_write.py
python mixedbread_referral_overlay.py fit
python mixedbread_referral_overlay.py evaluate
"""
    if "evidence/sentence-full-corpus-lexical.json" in files:
        readme+="""

The broader weight edits include the position-conditioned heads, the large
lexical memory, and the higher-order sequence decoder. The bundle contains
the written weights and hash-pinned benchmark receipts. Rebuilding the memory
uses the official LegalSeg training CSV and the pinned sequence assets.

python sentence_phase_readout_write.py evaluate
python sentence_full_corpus_lexical.py evaluate
python sentence_trigram_transition_write.py evaluate
python compare_broad_edits_to_bert.py corpus
"""
    if "evidence/sentence-minilm-whitened-active.json" in files:
        readme+="""

The MiniLM whitening study uses the archived embeddings from the pinned
quantized MiniLM graph and the original 36-judgment batch schedule. Restore
the original feature caches to replay the exact measurements. The archive
includes the benchmark receipts and the diagnostic heads.

python sentence_minilm_whitened_write.py evaluate
python sentence_minilm_whitened_write.py evaluate --diagnostic
"""
    if "evidence/sentence-referral-actual-replay-archive.json" in files:
        readme+="""

The archived-reviewer comparison uses the two hash-pinned author prediction
files named in the research report. It is a benchmark replay using saved
predictions. The local template-memory command accepts a JSON sentence list.

python sentence_referral_actual_replay.py archive
python sentence_template_memory_write.py predict --input judgment-sentences.json
python sentence_template_memory_write.py predict --bert-device CPU --input judgment-sentences.json
"""
    stage=OUTPUT.with_suffix(".part")
    with ZipFile(stage,"w",ZIP_DEFLATED) as z:
        for k,v in files.items():
            if not v.is_file():raise FileNotFoundError(v)
            z.write(v,k)
        z.writestr("manifest.json",json.dumps(meta,indent=2)+"\n")
        z.writestr("README.txt",readme)
    with ZipFile(stage) as z:
        assert z.testzip() is None
    os.replace(stage,OUTPUT)
if __name__=="__main__":main()
