# AttoFlow seven-role judgment classifier

A local research prototype for English Indian judgments. Give `predict` one sentence or paragraph to get one dominant role. Give `segment` a longer text to get contiguous labeled spans. All seven roles are retained: `none`, `facts`, `issue`, `petitioner_argument`, `respondent_argument`, `reasoning`, and `decision`.

## Run the smaller model offline

The `hierarchical` directory includes the complete 16.82M-parameter recurrent model and vocabulary. From the extracted archive run:

```sh
python hierarchical/hier_model.py 'We therefore allow the appeal.'
python hierarchical/hier_model.py --spans --text-file whole-judgment.txt
python hierarchical/hier_model.py --fragment --text-file excerpt-from-middle.txt
```

The complete-judgment and clipped-passage programs were selected on separate development samples. Their exact-span F1 values are task-specific. The BERT profiles below reconstruct a second model family.

## Run the BERT profiles on a CPU

With Python 3.11, run the following inside the extracted archive. Reconstruction downloads the public 438 MB donor and verifies SHA-256. An existing donor can be supplied with `--source /path/to/donor.safetensors`.

```sh
python -m pip install -r requirements.txt
python rebuild.py --profile compact --output-dir model
python model/predict.py "Notice was served on 3 May. We therefore allow the appeal."
python rebuild.py --profile document --output-dir document-model
python document-model/predict.py --spans < judgment.txt
```

In a ready-made model directory the predictor can be imported directly.

```python
from predict import JudgmentRoleClassifier

reader = JudgmentRoleClassifier()
one = reader.predict("We therefore dismiss the appeal.")
ranges = reader.segment("The notice was served. Counsel for the appellant challenged it. We dismiss the appeal.")
```

Offsets are Python Unicode character positions, with an inclusive start and an exclusive end. Use `text[start:end]` to recover the exact original characters. Spans cover the original string without gaps. Whitespace-only input raises an error. Raw excerpts that exceed one BERT window are fully covered with overlapping windows. The sequence decoder proposes candidate boundaries from punctuation and paragraph breaks, scores the resulting units, and merges adjacent equal roles. The document profile uses the strongest sequence program selected on local validation. The compact profile also includes its independently selected sequence weights.

## Weights and evaluation

Every BERT profile has 109,487,623 parameters and a 437,974,028-byte FP32 checkpoint. Each BERT classifier head has 5,376 output weights and seven biases. The internal profile also installs a replacement for the final feed-forward projection and its bias. Rebuild it with the internal profile and a separate output directory. The sequence layer adds 49 transition values and three neighbor coefficients. The `centroid` profile reconstructs the original edit, and `graft` reconstructs the exploratory sparse donor-row experiment for single-excerpt classification. When the letter-based instruction readout is included, its folder documents the larger local decoder separately.

See `evidence/AttoFlow-judgment-role-results.pdf` for measured performance, sample construction, uncertainty and limitations. The active legal results use seven-role LegalSeg and separate authored diagnostics. The report also describes historical controls and published references. Some inputs are intact annotations, some are adjacent rows joined under an explicit rule, and fictional examples have authored intended labels. Gemini Decision 1 receives a typed choice question. None of the active weights use provider predictions or gradient updates. The donor checkpoints were previously trained by their publishers.

The public checkpoint named for the standalone model is byte-identical to a file named for the previous sentence and previous gold label. The released prediction CSV and the loaded checkpoint produce different results. The report keeps those systems distinct.

The ready-made classifiers are research candidates. Their accuracy is an empirical result in the report. The JSON response is a role prediction, with uncertainty particularly high when speaker attribution or procedural context is absent.

[LegalSeg authors’ paper](https://aclanthology.org/2025.findings-naacl.63/) · [Pinned donor weights](https://huggingface.co/L-NLProc/LegalSeg_InLegalBERT/tree/44d7925e5ab422ecf04acd5dcbb0b62a6f8d8f53/InLegalBERT) · [Official dataset](https://huggingface.co/datasets/L-NLProc/LegalSeg_CSV) · [Gemini Decisions](https://aistudio.google.com/docs/decisions-api)
