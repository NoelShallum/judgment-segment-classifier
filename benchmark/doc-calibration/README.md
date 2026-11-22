# Case-aware document score calibration

This CPU experiment adds closed-form residual corrections to the two-checkpoint document model’s final seven role scores. The fit uses 43,401 annotated sentences in 192 official training judgments. The underlying two-checkpoint readout was also written using these judgments. The case-disjoint selector has 48 judgments. The two confirmation cohorts have 72 and 96 judgments and appeared in previous validation work.

The score banks preserve the pinned document classifier’s actual final decision on each validation row. Selection gives equal weight to macro-F1 on all rows and rows with no exact training-context key. It also checks the `None`–`Reasoning` error counts on both sets. The exploratory variants include a constant bias, score-only affine mappings, case-aware affine mappings, donor-expert features, and pair-only corrections. Matched controls shuffle case summaries within length bands or permute the fitting roles.

The frozen constant bias was the strongest candidate. It increased pooled validation macro-F1 from 72.445 to 72.695. The paired interval was +0.106 to +0.414 points. There were 171 fewer `None → Reasoning` and 171 more `Reasoning → None` errors, for 4,208 swaps in both versions. On the no-key subset, macro-F1 improved from 71.731 to 71.955 and pair swaps increased from 2,808 to 2,822. The case-aware controls yielded comparable results. The study used training and validation score banks.

The static bias can be fused into the native final linear decoder by changing the last row of `coef`. The serialized patch changes seven values. Its archived score replay matches all 45,221 validation labels from the selected bias version.

## Reproduction

With the existing public score bundle and protocol in place, run the following from `work/judgment-role`.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
trial=weight-edit-continuation/doc-calibration
.venv/bin/python "$trial/calibrate.py" fit
.venv/bin/python "$trial/calibrate.py" score
.venv/bin/python "$trial/audit.py"
```

`results.json` contains each cohort and comparator. `pooled-confirmation.json` contains the judgment-paired estimates. `native-audit.json` records the patch checksum, tensor changes, and replay checks. `entrypoint-verification.json` records a complete CPU forward on the public example using the serialized decoder. `margin-headroom.json` records a hypothetical correction at the selected bias’s effective pair-margin size.
