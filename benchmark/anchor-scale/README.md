# Original model features at expanded training scale

This experiment measures the value of the original whole-judgment model's predictions as inputs to a larger contextual classifier. Every arm uses the same two contextual encoders, compact features, and fit cases. A fixed analytic ridge solve writes a new seven-column readout. The three arms use the contextual encoders alone, add the original model's labels before its reviewer, or add the original model's final labels from the T4 reviewer. The existing `doc-semantic` classifier supplies the incumbent scores and remains a candidate at selection.

The fit prefixes use the first 256 and 512 authorized public LegalSeg training judgments in `protocol.json`, in order. Each arm and prefix uses the same fixed 48 judgments to choose ridge strength, rare-role weighting, and its mixture with incumbent scores. The objective gives equal weight to ordinary macro-F1 and macro-F1 on sentences without an exact training text key. Pair errors and accuracy break ties. The model weights and selection outcome are written before scoring the separate 72- and 96-judgment confirmation cohorts.

## Preparing the input

Run `prepare_anchor.py capture --cases 256` or `--cases 512` with the workspace Python environment. The capture uses the original document-model route, requires exact public train text and label alignment against the compact input, and checks original case fingerprints before reusing historical word encodings. It writes atomic 64-judgment blocks and can resume from verified blocks.

`capture-parity.json` compares the accelerated and original CPU captures on eight prefix judgments. All 1,752 pre-review labels and 354 ordered reviewer rows and token windows agree. The maximum difference is `2.39e-7` for structured features and `1.14e-7` for the reviewer prior. Using the cached word encodings took 7.07 seconds against 9.25 seconds for the original capture on these eight cases.

`pre-review-proxy.json` measures the difference between the original pre-review and final labels. Their agreement across the original fit, selector, and confirmation cohorts is 98.82–99.03%. Every exact-reviewed comparison uses actual final reviewer predictions for the expanded training rows as well as validation.

## Building and evaluating

Run `pack_anchor.py pack --blocks 4` or `--blocks 8` to assemble the T4 input packet. It contains the pinned review head, inputs for the original review rule, and a three-case replay control from the archived original T4 results. Run `t4_review.py --checkpoint /content/reviewer2/donor.safetensors` after extraction. The worker verifies the checkpoint, hardware, input checksums, saved label parity, and reviewer score tolerance before capturing the expanded rows.

Use the existing compact training banks and fixed basis at `/content/reviewer2/scale`. Run `fit_anchor.py --scale /content/reviewer2/scale --reviewed returned --blocks 4` or `--blocks 8` on CPU. Keep separate directories for the 256- and 512-judgment outputs. `freeze.json`, the fitted `expanded-anchors.safetensors`, the staged moments, and the split predictions provide the replay evidence. The `stage_remote.py` wrapper runs both steps and packages these files with the returned reviewer receipt.

Run `verify_stages.py anchor-256-dev.zip anchor-512-dev.zip` from the workspace environment once the result archives are present. The audit checks payload hashes, original validation case and row identities, selector rankings, saved confusion matrices, and T4 reviewer parity. It reports paired 95% confidence intervals over the 168 confirmation judgments, including each direction of the None–Reasoning error and the no-exact-key subset.
