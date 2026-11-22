# Routed seven-role ModernBERT NLI weight edit

The premise contains a marked target sentence and up to one neighboring sentence on each side. The same seven role hypotheses are evaluated as actual text pairs. The incumbent remains the frozen two-checkpoint whole-judgment classifier. The train-only router spends its budget on disagreements and small score margins, with explicit places reserved for None–Reasoning, Facts–Issue, and petitioner–respondent cases. The route, maximum budget, context and hypotheses are fixed before fitting the NLI readout. [Public pinned NLI checkpoint](https://huggingface.co/tasksource/ModernBERT-base-nli/tree/de4ab7e77845098b7fab7f6ab9d370ddff27b19c)

The donor has 149,607,171 parameters and a 3 × 768 classifier. Apache-2.0 is declared in the model card. The pinned config's `classifier_bias` field is false, while the tested Transformers implementation instantiates and loads the checkpoint's three-value classifier bias. The experiment preserves the actual loaded bias. All role evidence uses the relative, across-hypothesis centered entailment–contradiction margin, where any shared affine intercept cancels.

One least-squares solve per weight candidate estimates a direction from the 192 routed public training judgments. The edit adds half of the direction to the entailment weight row and subtracts half from the contradiction row. The freeze records the current document model, the candidate, the tuned source NLI donor, and norm-matched label shuffles. The two confirmation cohorts each contain complete judgments and are distinct from the 48 selector cases. Earlier project experiments used the same validation cohorts. These commands take train and validation inputs only.

## Results

| Whole judgments | Current macro-F1 | Edited macro-F1 | Current swaps | Edited swaps |
|---|---:|---:|---:|---:|
| 48 selection | 73.479 | 73.523 | 875 | 868 |
| 72 confirmation | 72.573 | 72.595 | 1,506 | 1,505 |
| 96 confirmation | 72.386 | 72.409 | 2,702 | 2,707 |
| 168 confirmation pooled | 72.445 | 72.470 | 4,208 | 4,212 |

This is a small shift across 37,279 confirmation sentences. With the edit, None-to-Reasoning decreased from 1,950 to 1,929 while Reasoning-to-None increased from 2,258 to 2,283. The case-bootstrap 95% interval for the macro-F1 change was −0.026 to +0.083 points. The unchanged NLI donor at the same operating point scored 72.409 with 4,219 swaps. Both norm-matched shuffled edits scored approximately 72.468 with 4,199 swaps. On sentences without an exact training key, the edit moved macro-F1 from 71.731 to 71.728 and swaps from 2,808 to 2,816. The selected gate could reach only 385 of the 4,208 incumbent swaps. The edit missed the paired confirmation criterion and the official test remained sealed.

The single T4 captured 36,533 pairs across fit, selection, and confirmation in 243.41 seconds. Inputs ran to 148 tokens and the target was truncated to fit in 252 of the 5,219 contexts. The FP32 head matched direct GPU logits exactly. Recomputed source logits from the exported activations matched within 2.15 × 10⁻⁶ on CPU. The head fit and selection took 11.12 seconds on CPU, followed by 6.13 seconds for both full confirmation cohorts and the paired bootstrap.

## Replay from `work/judgment-role`

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python weight-edit-continuation/entailment-edit/prepare.py
.venv/bin/python weight-edit-continuation/entailment-edit/dispatch.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python weight-edit-continuation/entailment-edit/solve.py preregister
```

The coordinator uses `dispatch.py --launch` against the existing `attoflow-role-context-t4` session. The capture validates the pinned checkpoint and a single Tesla T4. It saves the exact native head inputs and native logit parity receipt. The output contains only routed public author train and validation assets. Copy `entailment-captures.zip` to this directory once `capture.exit` is `0` and verify the archive checksum reported at the end of `capture.log`.

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python weight-edit-continuation/entailment-edit/solve.py fit
OPENBLAS_NUM_THREADS=1 .venv/bin/python weight-edit-continuation/entailment-edit/solve.py confirmation
OPENBLAS_NUM_THREADS=1 .venv/bin/python weight-edit-continuation/entailment-edit/verify_weights.py
```
