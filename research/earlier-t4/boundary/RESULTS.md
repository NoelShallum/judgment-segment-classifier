# GLiNER2.5 role-edit measurements

The FP32 checkpoint contains 73,881,879 parameters. The frozen edit changes all 81,920 values in three pre-existing weight matrices. Each additive update has numerical rank six. The supervised fit used 1,011 sentences in 12 judgments and used direct numerical linear algebra without backpropagation.

| Checkpoint tensor | Dimensions | Updated values | Edit Frobenius norm | Shuffled-control norm |
|---|---:|---:|---:|---:|
| `boundary_head.boundary_query_head.start_boundary_projection.weight` | 128 × 128 | 16,384 | 35.8681 | 35.8681 |
| `boundary_head.boundary_query_head.end_boundary_projection.weight` | 128 × 128 | 16,384 | 31.1993 | 31.1993 |
| `boundary_head.boundary_query_head.inside_text_projection.weight` | 128 × 384 | 49,152 | 46.4664 | 46.4664 |

The five training-selector judgments contained 447 rows. The selected joint solve at λ = 0.03 and gain 1 scored 0.3039 macro F1, versus 0.1715 for unchanged weights. The selected normal-equation relative residual was 0.00124 after 400 numerical iterations.

Reloading the published checkpoint and replacing its actual tensors reproduced the cached logits on a three-sentence fixture within 2.95e-06 for the selected edit and 1.76e-06 for the shuffled control. All argmax labels matched.

| Completed cohort | Method | Rows | Sentence macro F1 | Sentence accuracy | Exact labeled span F1 |
|---|---|---:|---:|---:|---:|
| Internal training-split check | native | 421 | 0.1258 | 0.1971 | 0.0299 |
| Internal training-split check | edit | 421 | 0.2744 | 0.4394 | 0.0508 |
| Internal training-split check | shuffled | 421 | 0.0858 | 0.1995 | 0.0205 |
| First four shared validation judgments | native | 841 | 0.1543 | 0.2105 | 0.0334 |
| First four shared validation judgments | edit | 841 | 0.2236 | 0.3769 | 0.0308 |
| First four shared validation judgments | shuffled | 841 | 0.1217 | 0.2200 | 0.0296 |
| First four shared validation judgments | incumbent | 841 | 0.6943 | 0.7669 | 0.3310 |
| All sixteen shared validation judgments | native | 3,558 | 0.1632 | 0.2102 | 0.0351 |
| All sixteen shared validation judgments | edit | 3,558 | 0.2632 | 0.3749 | 0.0418 |
| All sixteen shared validation judgments | shuffled | 3,558 | 0.1431 | 0.2203 | 0.0272 |
| All sixteen shared validation judgments | incumbent | 3,558 | 0.6988 | 0.7080 | 0.4294 |

## Paired uncertainty on sixteen judgments

| Edit minus comparator | Macro F1 difference | 95% judgment-bootstrap interval |
|---|---:|---:|
| native | +0.1000 | [+0.0792, +0.1168] |
| shuffled | +0.1201 | [+0.1019, +0.1416] |
| incumbent | -0.4356 | [-0.5153, -0.3874] |

## Repairing the incumbent spans

The short-run experiment accepts each proposed span of up to three sentences lying strictly within an incumbent run. The separate relocation experiment moves each incumbent boundary by up to two sentences according to the GLiNER scores. The oracle rows use the human labels to measure available headroom.

| Method | Sentence macro F1 | Exact span precision | Exact span recall | Exact span F1 | Short span recall |
|---|---:|---:|---:|---:|---:|
| incumbent | 0.6988 | 0.4545 | 0.4069 | 0.4294 | 0.4354 |
| edit | 0.2632 | 0.0312 | 0.0631 | 0.0418 | 0.0843 |
| edit_insert | 0.4910 | 0.1697 | 0.3540 | 0.2294 | 0.4593 |
| edit_relocate | 0.6325 | 0.2239 | 0.2004 | 0.2115 | 0.2430 |
| edit_oracle_insert | 0.7003 | 0.4586 | 0.4171 | 0.4369 | 0.4494 |
| oracle_relocate | 0.7099 | 0.5398 | 0.4832 | 0.5099 | 0.5169 |

| GLiNER weights | Proposals | Correct proposals | Eligible missed gold spans |
|---|---:|---:|---:|
| native | 1168 | 4 | 105 |
| edit | 743 | 7 | 105 |
| shuffled | 1064 | 12 | 105 |

Paired 95% case-bootstrap intervals for exact span F1 change were [-0.2424, -0.1607] for GLiNER-edited insertions and [-0.2860, -0.1644] for GLiNER-edited relocation.

## Publisher API on isolated sentences

A separate, preselected sample includes 16 noncontiguous sentences from each of the first four judgments. The publisher API receives one sentence at a time. The span decoder and incumbent use their existing contextual predictions on those same row IDs.

| Method | Sentence macro F1 | Sentence accuracy |
|---|---:|---:|
| publisher_classify_text | 0.1032 | 0.1094 |
| native | 0.1489 | 0.2031 |
| edit | 0.2420 | 0.3750 |
| shuffled | 0.0966 | 0.1875 |
| incumbent | 0.6011 | 0.7188 |

All main methods receive the corpus sentence boundaries. Fits, selection and primary confirmation use different judgment IDs. The first four validation judgments are included in the sixteen-judgment cohort. Native denotes a compulsory seven-role decode of the native span marginals. The labeled exact-span metric requires matching both endpoints and the role of a contiguous run of sentences. The incumbent is the unchanged lane from the shared case manifest.
