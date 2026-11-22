# AttoFlow Judgment Segment Classifier

This project classifies sentences in English-language Indian judgments into seven LegalSeg rhetorical roles. The whole-judgment benchmark supplies ordered sentence boundaries. The inference CLI returns one role per sentence and merges adjacent sentences with identical labels into spans.

## Official LegalSeg test

The primary results cover 149,868 nonblank sentences in 712 judgments.

| System | Macro-F1 | Accuracy | None ↔ Reasoning errors |
|---|---:|---:|---:|
| Original AttoFlow document model | 64.39 | 66.00% | 25,412 |
| Two contextual encoders and document readout | 67.22 | 66.57% | 25,208 |
| Direct None–Reasoning decoder edit | 67.21 | 66.72% | 24,968 |
| Readout using earlier reviewer predictions | 67.44 | 66.84% | 25,062 |
| Semantic nearest-neighbor memory | **67.88** | 66.50% | 25,230 |
| LegalSeg hierarchical BiLSTM-CRF, published | 77.00 | 62.00% | — |

The semantic-memory system ranks first by macro-F1 among AttoFlow models. The reviewer readout ranks first by accuracy, and the direct decoder edit records the fewest None–Reasoning errors among the three new variants shown. The public author outputs did not independently reproduce the paper's 77.00 result. See the [research report](reports/AttoFlow-LegalSeg-weight-editing-official-report.pdf) and [LegalSeg Table 3](https://aclanthology.org/2025.findings-naacl.63.pdf#page=7).

## Run inference or audit the results

- **67.22 CPU package.** In Drive, open **AttoFlow / Judgment Segment Classifier — Models** and download `AttoFlow-LegalSeg-context-model.zip`. Extract it and follow the [setup and inference instructions](research/context/README.md). The installer downloads pinned public donor weights. The CPU path reproduced 119 of 119 T4 labels across two example judgments.
- **67.44 CPU prototype.** Download `AttoFlow-LegalSeg-reviewer-readout-67-44-preview.zip` from the same folder. Its CPU parity check covered one complete judgment. See the [prototype instructions](research/reviewer-readout/README.md).
- **67.88 research variant.** Use the [frozen configuration, T4 experiment code, scores and predictions](benchmark/scale) alongside the archived compact training features in Drive. The archive contains feature shards for rebuilding the retrieval index. The original research replay also requires the locked contextual activations and runtime layout. End-to-end laptop inference has yet to be verified.
- **Audit the scores.** With Python 3.11+, NumPy, and [the public LegalSeg test CSV](https://huggingface.co/datasets/L-NLProc/LegalSeg_CSV/tree/main), run the command below. The audit includes per-role F1, confusion counts, paired intervals, and a companion score including the 13 blank rows.

```sh
python benchmark/audit.py --csv /path/to/test.csv \
  --prediction-file benchmark/official-predictions.npz --output local-audit.json
```

## Research archive

- [Reports and model catalog](reports) cover the original opportunity research, sentence and span experiments, and the final checkpoint comparison.
- [Benchmark evidence](benchmark) includes official row predictions, validation results, frozen configurations and software for the recent edits.
- [Earlier experiment code and results](research) covers sentence-only, paragraph and span work as well as the model release sources. The source trees are references from the packaged releases. Use the complete model ZIPs in Drive for inference and package checksum verification.
- Third-party data and model files retain their upstream terms and attribution. See the linked source releases and the notices in the model packages.
