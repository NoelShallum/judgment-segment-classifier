# AttoFlow LegalSeg anchor model

Classify a complete English Indian judgment supplied as a JSON array of ordered sentences. The CPU CLI emits one of the seven LegalSeg roles per input and groups consecutive sentences of the same role into spans.

The 48-judgment selector chose the `exact-reviewed` readout fitted on 256 public LegalSeg training judgments.

## Setup

The archive includes the original document model, its reviewer and the frozen contextual weights and expanded readout. It requires Linux, Python 3.11, a C++17 `g++` compiler and three pinned public InLegalBERT donors of approximately 418 MiB each. The installer verifies each checkpoint SHA-256 and can download the donors anonymously. After installation, classification runs offline.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
python -m pip install -r requirements.txt
python context_assets.py verify
python context_assets.py install
python classify_anchor.py --input example.json
```

The `--incumbent-file`, `--past-file` and `--both-file` installer options accept already downloaded checkpoints. Use `--donor-dir` during installation and inference to reuse a model directory. The first inference compiles the bundled sequence decoder.

## Model behavior

The model encodes each sentence in two BERT contexts. One view includes the two preceding sentences. The other includes one preceding and one following sentence. Views longer than 512 tokens are split into overlapping windows with weighted pooling. The model combines score features, compact semantic features and the selected original-model role stream across the judgment. The expanded seven-role readout was fitted with an analytic ridge solve on the public training labels with frozen encoder weights.

`labels` and `spans` contain the selected model predictions. `context_labels` shows the previous contextual model. `incumbent_labels` and `pre_review_labels` show the original document model's final and pre-review labels. Span offsets are zero-based and end-exclusive. The default CPU implementation uses FP32 arithmetic with the same FP16-rounded contextual donor weights and pooling as the T4 feature extraction. `--encoding fp16` uses CPU half-precision arithmetic and will generally take longer.

## Evidence and reproduction

`anchor-evidence/results.json` gives the 48-judgment selector and separate 72- and 96-judgment confirmations. The archive includes the selected weight file with per-tensor SHA-256 receipts and the frozen model selection. When present, `anchor-evidence/official-results.json` records the matched official result and paired intervals against both historical models. The project had previously examined the official split.

The CPU verification command rebuilds features from the public validation text and checks the selected predictions, context predictions and both original-model streams on two complete judgments. `anchor-evidence/cpu-qa.json` records the checked cases, timing and any disagreements for the released artifact.

```bash
python verify_anchor_cli.py --validation /path/to/val.csv
```

The fitting and evaluator sources are in `research/`. Re-running those experiments requires the aligned research banks. The user-facing inference command uses the packaged weights and the three pinned checkpoints.

## Sources

- Author models and checkpoint attribution. [Pinned InLegalBERT release](https://huggingface.co/L-NLProc/LegalSeg_InLegalBERT/tree/44d7925e5ab422ecf04acd5dcbb0b62a6f8d8f53/InLegalBERT)
- Benchmark and public source judgments. [LegalSeg paper](https://aclanthology.org/2025.findings-naacl.63/) and [public CSVs](https://huggingface.co/datasets/L-NLProc/LegalSeg_CSV/tree/main)
