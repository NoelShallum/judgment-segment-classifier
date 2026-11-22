# Fixed-partition role residuals

The experiment freezes random input partitions and writes one seven-value table per leaf from the signed residuals observed in public LegalSeg training cases. The routing features include original and edited donor logits, the current role and rival, document position, and left and right context. Split points use quantiles of training features. Weighted residual means shrink toward zero. A leaf needs support from at least three distinct fitting judgments to receive nonzero values.

The fit uses 43,401 sentences from 192 judgments. The same judgments contributed to fitting the base document readout. The experiment chooses settings on 48 separate judgments and replays them on two separate confirmation sets of 72 and 96 judgments. Those validation sets have historical project exposure. The no-key subset excludes rows with exact training-context keys at widths of 1, 3, 5 or 9 sentences. The code reads the four development members of the score bank archive.

## Paired outcome

| Cohort | Base macro-F1 | Pair residual macro-F1 | Base pair errors | Pair residual pair errors |
|---|---:|---:|---:|---:|
| 48 selector judgments | 73.4791 | 73.6023 | 875 | 855 |
| 72 confirmation judgments | 72.5727 | 72.6100 | 1,506 | 1,489 |
| 96 confirmation judgments | 72.3857 | 72.3523 | 2,702 | 2,709 |
| Pooled confirmation | 72.4448 | 72.4386 | 4,208 | 4,198 |
| Pooled no-key rows | 71.7314 | 71.7075 | 2,808 | 2,813 |

The protected pair residual was the only family to meet the selector requirements. Its pooled paired 95% macro-F1 interval was −0.062 to +0.057 points and its pair change interval was −54 to +32. The equal-norm shuffled control produced 72.4386 macro-F1 and 4,194 pooled pair errors. The fixed partitions gave inconsistent confirmation results.

The protected pair residual changed 359 of 37,279 pooled confirmation labels. Its 169 fixes, 159 new errors, and 31 changes from one wrong role to another give 47.1% correction precision. Its four-case-fold training calibration estimated 49.9%. On the 72-judgment set, the high-margin bin corrected 35 of 52 changes. On the 96-judgment set, it corrected 31 of 74. The validation performance varied across cohorts even at higher switch scores.

The follow-up reused the fixed partitions and calculated an expected per-role macro-F1 benefit for changing the incumbent label. The fitting confusion matrix supplies the class supports and benefits. All 108 configurations retained the base predictions on the selector. The zero write therefore remained in effect for the two confirmation cohorts.

## Replay

From `work/judgment-role`, use the existing development score bank and the frozen no-key files.

```bash
export CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
trial=weight-edit-continuation/forest-memory
.venv/bin/python "$trial/run_forest.py" fit
.venv/bin/python "$trial/run_forest.py" score
.venv/bin/python "$trial/audit.py"
.venv/bin/python "$trial/action_readout.py" fit
.venv/bin/python "$trial/action_readout.py" score
```

`freeze.json` records the selector decisions, fitted tensors and cross-fitted precision calibration. `results.json` contains the separate cohort scores. `runtime-audit.json` verifies stored value norms and reproduces the predictions for all 45,221 development rows through the NumPy entry point in `forest_runtime.py`. It also verifies that each sampled judgment receives the same predictions in isolation. `pooled-confirmation.json` includes the paired pooled counts. The `action` directory contains the follow-up protocol, freeze and scores.
