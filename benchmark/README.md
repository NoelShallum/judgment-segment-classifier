# AttoFlow LegalSeg weight-editing experiments

## Verify the official scores

Use Python 3.11 or later, NumPy, the public test CSV, and the row predictions bundled here.

```bash
python -m pip install numpy
python audit.py --csv /path/to/test.csv --prediction-file official-predictions.npz --output my-audit.json
```

Download test.csv from https://huggingface.co/datasets/L-NLProc/LegalSeg_CSV/tree/main if you do not already have it. The recorded SHA-256 for the CSV is 5e1a2f9a03fd7581d4585ef97f55787d058afbed063b5e03b6e3611cbe45f9ee. The scorer checks grouping and exact gold order against the CSV. It computes seven-role macro-F1, ordinary accuracy, confusion matrices, direction-specific swaps and paired judgment intervals.

The main denominator is 149,868 nonblank sentences in 712 judgments. A second result includes all 149,881 CSV rows. The 13 blank inputs receive the train-modal blank label None and are omitted when assembling sentence context. Case IDs and offsets identify original judgment order in the prediction archive. Original row order is preserved within each judgment.

## Interpret the comparisons

The numerical reference is 77.00 macro-F1 for Hierarchical BiLSTM-CRF in Table 3 of https://aclanthology.org/2025.findings-naacl.63.pdf. It is a paper-reported result. Publicly released predictions did not independently verify that row, so the report gives no paired significance test against the published number.

The original, reviewer, author_heads, legal_context and exact_and_aligned arrays identify the previous document model, the centered rank-six reviewer edit, two original public BERT heads with a fitted readout, the two-encoder semantic readout, and lexical memory layered over it. The semantic_neighbors array scores 67.88 macro-F1. It adds a memory of 1,032,067 labeled training sentences from 4,600 judgments. The anchored_scale array scores 67.44 using an expanded readout with the previous reviewer's labels. The decoder_pair array scores 67.21 after editing the two relevant decoder columns. A zero-strength setting can produce exactly the baseline prediction array.

The audit resamples 712 complete judgments with replacement 4,000 times. Deltas in the JSON use unit-scale F1 or accuracy, and signed counts for mistakes. The oracle uses known gold and reports hypothetical headroom. The validation subsets and official test have prior project exposure.

## Research evidence

The subdirectories contain methods, candidate locks, selected weights and evaluation receipts. Several research scripts refer to the original intermediate feature banks and directory layout. The scorer above runs independently of those banks. The separately packaged AttoFlow-LegalSeg-context-model.zip reproduces the 67.22 system from text. Its installer pins three public donor checkpoints and verifies their checksums. Its classify_fast.py command accepts an ordered JSON array of sentences. The 67.44 CPU prototype passed a one-judgment replay. The 67.88 model's official predictions were verified on the T4. Its end-to-end laptop runtime remains unmeasured. See scale/standalone.log for an independent raw-input approach that failed comparison with its development reference and was excluded.

The primary context model used 192 training judgments for its new head and readout. The centered reviewer fit used 384 judgments. Expanded-corpus runs used disjoint additional training judgments. Public test labels supplied metric calculations after each contender was locked. All new accelerator work ran on a Tesla T4. CPU linear algebra, report generation, inference verification and replay of earlier frozen features also contributed. The Gemini evidence is a previous isolated-sentence evaluation.

Source references include the LegalSeg authors' code at https://github.com/ShubhamKumarNigam/LegalSeg and pinned weights at https://huggingface.co/L-NLProc/LegalSeg_InLegalBERT/tree/44d7925e5ab422ecf04acd5dcbb0b62a6f8d8f53/InLegalBERT.

SHA256SUMS.json covers every other archive member. Several nested archives also carry member checksums. No sign-in information is needed to audit the local scores.
