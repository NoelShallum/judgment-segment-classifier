# Discourse weights for the seven-role judgment classifier

This experiment adds text-derived evidence to the frozen two-checkpoint LegalSeg scores for every supplied sentence of a judgment. It tracks direct and aliased speakers, reported prior judgments, the present court, and the source, continuation and close of quotations. It also uses recognizable headings and evidence from up to 24 preceding and six following sentences. Relative judgment phase and within-judgment cue frequency affect the weights.

The weights use three closed-form estimates.

- The emission table measures the log odds for each role when a cue is present relative to the same judgment phase.
- The transition table measures the change in association between adjacent roles when a quotation or speaker cue occurs. A CPU Viterbi pass uses those scores as edge potentials.
- A residual readout uses the named cue features and association scores to correct the frozen classifier’s evidence on the 192 official training judgments with score banks.

The associative tables use a prespecified hash sample of 1,536 official training judgments. The earlier full scan ended when its process was interrupted after 3,840 judgments and before it saved its sufficient statistics. The recovery run checkpoints each block. The model is compared with zero correction, local-only forms, and matched-norm tables built by rotating training labels inside each judgment.

Selection uses 48 judgments and gives equal weight to full-set and no-exact-training-key macro-F1, with guardrails on pair errors and accuracy. The 72 and 96 confirmation cases are case-disjoint from selection and from the residual fit. They appeared in earlier project validation studies. All results use published sentence divisions. The official test member of the score archive stays outside this study.

A direct scan of all public training and validation text fields found zero ASCII or curly quotation markers, apostrophes, backticks, guillemets or trailing colons. It found 17,851 training sentences and 5,128 validation sentences with textual report introductions such as “as follows.” The parser activated an inferred quote source on 3,321 of the 37,279 confirmation sentences. This is evidence about inferred reports and attributions. Evaluation of explicit quotation boundaries requires text with quotation markers.

The selector chose the context residual with pair-only edits. Pooled confirmation macro-F1 moved from 72.445 to 72.467, with a paired 95% interval of −0.009 to +0.058 points. The pair count went from 4,208 to 4,182, compared with 4,174 for the matched shuffled control. The candidate scored lower than the incumbent on the 96-judgment cohort. The confirmation results did not qualify for an official replay.

## Reproduce

Use the existing LegalSeg workspace and frozen semantic score archive.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
trial=weight-edit-continuation/discourse-memory
.venv/bin/python "$trial/associations.py" --workers 4 --judgments 1536
.venv/bin/python "$trial/study.py" fit
.venv/bin/python "$trial/study.py" score
.venv/bin/python "$trial/audit.py" all
```

`training.json` lists the actual training support, sampling rule, feature order and timing. `freeze.json` pins the selector’s decisions before the confirmation cohorts are read. `results.json` and `pooled-confirmation.json` include judgment-level intervals, both confusion directions, no-key results, controls and role-boundary F1. `native-replay.json` checks the serialized weights with fresh text features from complete validation judgments.
