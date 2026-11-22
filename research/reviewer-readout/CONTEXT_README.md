# AttoFlow LegalSeg context model

Classify a complete English Indian judgment supplied as an ordered JSON array of sentences. The CPU CLI returns one LegalSeg role per sentence and groups adjacent sentences with the same label into spans.

## Setup

The release includes the original AttoFlow document pipeline and all three sets of fitted context tensors. It requires Linux, Python 3.11, a C++17 `g++` compiler and the three pinned public InLegalBERT donors. Each donor occupies about 418 MiB. The installer fetches them without credentials and verifies their exact SHA-256 digests.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
python -m pip install -r requirements.txt
python context_assets.py verify
python context_assets.py install
python classify.py --input example.json
```

The installer also accepts `--incumbent-file`, `--past-file` and `--both-file` for already downloaded checkpoint files. It stores the verified copies in `models`. Both installation and classification accept `--donor-dir` to select a different local path. Classification can run offline after setup. Its first invocation compiles the bundled document sequence decoder.

## Input and output

Supply the nonblank sentences of a single judgment in order. `labels` follows that input order. In `spans`, `start_sentence` is zero-based and `end_sentence` is exclusive. `incumbent_labels` returns the original document pipeline's labels before its two-encoder successor. `backbone_windows` gives the window count and maximum encoded length for each contextual encoder.

The model evaluates every target in two views. One includes the two preceding sentences. The other includes the immediate preceding and following sentences. Sequences that exceed 512 tokens use overlapping windows and a coverage-weighted combination of their pooled vectors. Each donor contributes its original role logits, edited role logits and projected semantic features to a frozen document readout. The readout combines them across the judgment and uses the original AttoFlow labels as an input.

## Evaluation and reproduction

The validation-frozen model scored **67.22 macro-F1** and **66.57% accuracy** on 149,868 nonblank sentences from 712 official test judgments. The comparable original system scored 64.39 macro-F1 and 66.00% accuracy. The paired 95% interval for the macro-F1 change was +2.35 to +3.33 points. The contextual readout fitted on original author logits scored 67.21 macro-F1. Total None–Reasoning swaps changed from 25,412 to 25,208. Their paired 95% interval includes zero. None → Reasoning increased from 11,323 to 13,346, and Reasoning → None fell from 14,089 to 11,862. These scores use full T4 FP16 encoder passes and the locked document readout. The project had previously examined the test split.

The frozen model scored 72.57 and 72.39 macro-F1 on two separate, previously examined LegalSeg validation cohorts of 72 and 96 judgments. Their full T4 results, including all seven class metrics, are in `evidence/evaluation.json`. The release includes the exact tensors and public-source pins. The underlying experiment used the paper's seven labels and supplied sentence boundaries. [LegalSeg paper](https://aclanthology.org/2025.findings-naacl.63.pdf#page=7)

The CPU CLI recomputes both encoders from input sentences. The packaged verification command compares its labels with the archived T4 outputs on two whole validation judgments. Those two checks include both an ordinary case and a case with inputs that overflow the first BERT window.

```bash
python verify_validation.py --validation /path/to/val.csv
```

`research/` includes the exact fitting, encoder and decision references from the T4 experiment. Re-running those research programs requires the original experiment files. The supplied CLI uses the packaged models and the three public donors and runs its own CPU inference.

### CPU arithmetic

On 1 complete QA judgment, `classify.py` matched 41/41 final T4 labels with CPU FP16 arithmetic and took 525 seconds in total on the test laptop. The optional `classify_fast.py` uses FP32 arithmetic with the same half-rounded weights and FP16 window pooling. Across both cases, it matched 119/119 final labels and took 492 seconds. The complete comparison and any differing rows are in `evidence/cpu-validation.json`.

```bash
python classify_fast.py --input example.json
python verify_validation.py --validation /path/to/val.csv --backend fp32
```

## Sources

- Public checkpoints and model attribution. [Pinned InLegalBERT release](https://huggingface.co/L-NLProc/LegalSeg_InLegalBERT/tree/44d7925e5ab422ecf04acd5dcbb0b62a6f8d8f53/InLegalBERT)
- Public validation judgments. [LegalSeg CSVs](https://huggingface.co/datasets/L-NLProc/LegalSeg_CSV/tree/main)
