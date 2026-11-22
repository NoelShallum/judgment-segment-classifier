AttoFlow whole-judgment span edits

The role-edge directory contains the newest whole-judgment profiles. They use the
same base and the original seven labels. Weights and vocabulary are checksum-pinned.

python role-edge/predict_role_edge.py judgment.txt --model-dir /path/to/hierarchical --profile exact
python role-edge/predict_role_edge.py judgment.txt --model-dir /path/to/hierarchical --profile balanced

The role-pair table is derived from counts in LegalSeg's public training split.
On the 712-case test the role-pair exact profile scores 0.301 exact-span F1
and 51.9% token accuracy. The balanced profile scores 0.288 and 60.6%.
The profiling and matched archive results are under evidence/role-edge.

The experimental segment-exact profile jointly selects labeled runs from raw text.
It scores 0.304 exact span F1 and
51.5% token accuracy on the same 712 judgments.

python role-edge/predict_role_edge.py judgment.txt --model-dir /path/to/hierarchical --profile segment-exact

An additional interpolation scores 0.300 exact-span F1 and 53.4% token accuracy.

python role-edge/predict_role_edge.py judgment.txt --model-dir /path/to/hierarchical --profile interpolated

The sentence-context profile reads a JSON array of nonblank sentences from one judgment
and returns one label for each sentence in order. It scores 0.627 macro-F1 and
65.0% sentence accuracy on the 149,868 nonblank LegalSeg test rows.

python role-edge/predict_role_edge.py judgment-sentences.json --sentences --model-dir /path/to/hierarchical --profile sentence-context

The sentence-calibrated profile accepts the same JSON array and scores
0.629 macro-F1 and 64.9% sentence accuracy.
Its reasoning F1 is 0.556.

python role-edge/predict_role_edge.py judgment-sentences.json --sentences --model-dir /path/to/hierarchical --profile sentence-calibrated

The sentence-uncertainty profile uses frozen hierarchy scores and a small
readout to correct low-confidence sentence predictions. It scored 0.6292
macro-F1 and 65.05% accuracy on the same official sentences.
The inference uses the existing hierarchy and the same JSON sentence input.

python role-edge/predict_role_edge.py judgment-sentences.json --sentences --model-dir /path/to/hierarchical --profile sentence-uncertainty

The sentence-joint profile applies posterior and learned transition
scores together to supplied sentences. It scored 0.63217 macro-F1
and 64.94% sentence accuracy on the official test.
It uses the existing local hierarchy and returns one of the seven roles per input.

python role-edge/predict_role_edge.py judgment-sentences.json --sentences --model-dir /path/to/hierarchical --profile sentence-joint

The sentence-local profile adds a small readout that uses the role scores
of the preceding and following sentences. It scored 0.63385
macro-F1 and 65.13% accuracy on the official sentence test.

python role-edge/predict_role_edge.py judgment-sentences.json --sentences --model-dir /path/to/hierarchical --profile sentence-local

The sentence-isolated profile classifies each input sentence independently.
It scored 0.40580 macro-F1 and
46.31% accuracy on the official sentence-only test.

python role-edge/predict_role_edge.py sentences.json --sentences --model-dir /path/to/hierarchical --profile sentence-isolated

The sentence-document profile adds two-hop document context to supplied
sentences. It scored 0.63429 macro-F1 and
65.16% accuracy on the official sentence test.

python role-edge/predict_role_edge.py judgment-sentences.json --sentences --model-dir /path/to/hierarchical --profile sentence-document

The sentence-document-calibrated profile adds a closed-form
calibration layer to the document scores. It scored 0.63469
macro-F1 and 65.29% accuracy on the official sentence test.
Its accuracy gain has a positive paired interval. Its macro-F1 interval spans zero.

python role-edge/predict_role_edge.py judgment-sentences.json --sentences --model-dir /path/to/hierarchical --profile sentence-document-calibrated

The raw-local profile uses the local readout on raw text. It scored
63.96% token accuracy and
0.2462 exact-span F1 on official judgments.

python role-edge/predict_role_edge.py judgment.txt --model-dir /path/to/hierarchical --profile raw-local

The raw-joint profile shares the sentence-joint weights and accepts raw
judgment text. It returns labeled Unicode spans. For full official judgments it
scored 63.75% token accuracy and
0.2409 exact-span F1.
Use segment-exact when matching gold boundaries is the priority.

python role-edge/predict_role_edge.py judgment.txt --model-dir /path/to/hierarchical --profile raw-joint

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

The newer discourse profiles use the same base model with a written cue and position residual.

python predict_discourse_spans.py --model-dir /path/to/hierarchical --profile discourse_balanced.safetensors --text-file judgment.txt
python predict_discourse_spans.py --model-dir /path/to/hierarchical --profile discourse_exact.safetensors --text-file judgment.txt

The discourse_balanced profile gives 0.278 exact span F1 and 59.8% token accuracy on all 712 cases.
The discourse_exact profile gives 0.286 exact span F1 and 50.3% token accuracy on the same input.
Both are research checkpoints. Refer to the evidence reports for paired uncertainty and context.


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


The wider reviewer benchmark scores a fixed 20% confidence and utility review
budget. The audit replays smaller nested budgets from its completed FP32
feature caches. Restore the recorded benchmark environment before running.

python sentence_wide_bert_benchmark.py --device GPU
python audit_wide_bert_budget.py confirm


The CPU command runs the 20% confidence review on a new judgment. It accepts
supplied sentences as a JSON array or raw text with --raw. The same pinned
OpenVINO assets used by the original referral command are required.

python predict_wide_bert_review.py --input judgment-sentences.json --device CPU
python predict_wide_bert_review.py --input judgment.txt --raw --device CPU


The seven-role reviewer correction uses cross-fitted training activations and
the FP32 features from the wider review benchmark. Evaluation uses the saved
matrix and the confidence route.

python sentence_wide_bert_multiclass_write.py fit select confirm
python sentence_wide_bert_multiclass_write.py evaluate test


The memory composition replays the original template policy with the wider
FP32 reviewer, preserving accepted reviewer changes.

python audit_wide_bert_template_composition.py select confirm


The judgment-level prevalence write estimates corrections to seven role
frequencies from official training judgments. The paired comparison uses
the restored LegalSeg CSVs and the pinned sequence classifier.

python sentence_role_prevalence_write.py select confirm holdout test


The pairwise headroom audit uses the official confusion matrix of the
local InLegalBERT plus template-memory classifier.

python audit_role_pair_headroom.py


The near-match experiment reconstructs a sparse character n-gram memory
from the restored official LegalSeg training CSV. It uses the same judgment
exclusions recorded in the exact-template source receipt.

python sentence_fuzzy_pair_memory.py select confirm


The CPU reviewer diagnostic rebuilds FP32 and INT8 ONNX graphs from the pinned
checkpoint and compares them on the existing referral sets. It requires the
reference FP32 feature caches, PyTorch, ONNX and ONNX Runtime.

python bench_bert_int8_review.py confirm select --threads 4


The precision diagnostic compares signed and unsigned weight settings on the
same deterministic subset of 32 confirmation windows.

python diagnose_bert_int8_precision.py


The dense readouts in reproduce/checkpoints use the frozen hierarchical CPU
encoder, its released document calibration, and the pinned lexical table. The
experimental predictor uses NumPy, SciPy, scikit-learn, safetensors and PyTorch.
After restoring the workspace inputs and weights described above, pass a JSON
array of ordered sentences from one judgment.

python sentence_dense_readout_write.py predict --arm nonlinear --input judgment-sentences.json

The --preserve-issues option keeps the previous document model’s issue
membership decisions. The report describes the official test and the trade-off
on a separate validation cohort.


The recurrent inverse experiment uses native candidate-state LSTM tensors in
reproduce/checkpoints. Its readout-only control, shared decoder settings and
matched readout refit are included with their benchmark evidence.

python sentence_recurrent_inverse.py prepare fit select confirm holdout test
python sentence_recurrent_inverse.py learn
python sentence_recurrent_inverse.py evaluate select confirm holdout test
python sentence_recurrent_context.py
python sentence_recurrent_refit.py confirm holdout test


The phrase experiment includes the fixed word and character vocabularies, the
lexical projection and the fitted dense readouts. The CPU command accepts an
ordered JSON array of nonblank sentences. It uses the same reference
hierarchy and lexical input files as the other experimental readouts.

python sentence_phrase_readout.py predict --arm "contextual phrases" --input judgment-sentences.json


The sequence experiment provides two frozen objectives. Supplied-sentence mode
returns a label for each input and contiguous spans with half-open sentence
indices. Raw mode returns labeled spans with half-open character offsets over
the original UTF-8 input.

python sentence_phrase_sequence.py predict --profile spans --input judgment-sentences.json
python sentence_phrase_sequence.py predict --profile spans --input judgment.txt --raw


The pairwise experiment adds closed-form classification readouts to the frozen
sequence model. With the research working directory and the pinned assets
restored, the supplied-sentence CPU command returns labels and sentence spans.

python predict_pairwise_judgment.py --method direct --input judgment-sentences.json
python predict_pairwise_judgment.py --method consensus --input judgment-sentences.json


The joint pairwise experiment applies the pair evidence inside the sequence
decoder. Both supplied sentences and raw text are supported.

python sentence_joint_pairwise.py predict --profile spans --input judgment-sentences.json
python sentence_joint_pairwise.py predict --profile spans --input judgment.txt --raw


The Mixedbread referral benchmark uses mxbai-embed-xsmall-v1 at revision
e6ac24e5d6efb8782b59de1647b3ececb4ece94e. Restore the files named in
evidence/mixedbread-model-provenance.json to references/mxbai-embed-xsmall.
Its 96-token inputs use the original BERT referral positions and up to two
earlier sentences. The pretrained Mixedbread weights are downloaded separately.

python mixedbread_referral_write.py
python mixedbread_referral_overlay.py fit
python mixedbread_referral_overlay.py evaluate


The broader weight edits include the position-conditioned heads, the large
lexical memory, and the higher-order sequence decoder. The bundle contains
the written weights and hash-pinned benchmark receipts. Rebuilding the memory
uses the official LegalSeg training CSV and the pinned sequence assets.

python sentence_phase_readout_write.py evaluate
python sentence_full_corpus_lexical.py evaluate
python sentence_trigram_transition_write.py evaluate
python compare_broad_edits_to_bert.py corpus


The MiniLM whitening study uses the archived embeddings from the pinned
quantized MiniLM graph and the original 36-judgment batch schedule. Restore
the original feature caches to replay the exact measurements. The archive
includes the benchmark receipts and the diagnostic heads.

python sentence_minilm_whitened_write.py evaluate
python sentence_minilm_whitened_write.py evaluate --diagnostic


The archived-reviewer comparison uses the two hash-pinned author prediction
files named in the research report. It is a benchmark replay using saved
predictions. The local template-memory command accepts a JSON sentence list.

python sentence_referral_actual_replay.py archive
python sentence_template_memory_write.py predict --input judgment-sentences.json
python sentence_template_memory_write.py predict --bert-device CPU --input judgment-sentences.json
