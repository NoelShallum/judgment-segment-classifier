"""Assemble a ready-to-load model, portable 7-row edit, and sourced evaluation records."""
import json
import os
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from weight_edit import LABELS, ROOT, file_hash, tensor

OUT = ROOT.parents[1] / "outputs/judgment-role"
MODEL = OUT / "model"
KIT = ROOT / "portable-edit"
SOURCE = ROOT / "checkpoints/selected-direct"


def install(src, dest):
    dest.parent.mkdir(exist_ok=True, parents=True)
    if dest.exists() and os.path.samefile(src, dest):
        return
    if dest.exists() and file_hash(src) == file_hash(dest):
        return
    stage = dest.with_name(dest.name + ".partial")
    if stage.exists():
        stage.unlink()
    try:
        os.link(src, stage)
    except OSError:
        shutil.copyfile(src, stage)
    stage.replace(dest)


def write(path, contents):
    path.parent.mkdir(exist_ok=True, parents=True)
    stage = path.with_name(path.name + ".partial")
    stage.write_text(contents, encoding="utf-8")
    stage.replace(path)


def main():
    for folder in (MODEL, KIT):
        folder.mkdir(exist_ok=True, parents=True)
    check = json.loads((ROOT / "results/distribution-check.json").read_text())
    report = json.loads((ROOT / "results/program-validation.json").read_text())
    c = report["confirmation"]
    rows = [f"| `{k}` | {c['base']['f1'][k]:.3f} | {c['selected']['f1'][k]:.3f} |"
            for k in LABELS]
    text = """# AttoFlow judgment rhetorical roles

This experimental checkpoint assigns one dominant rhetorical role to an English sentence or paragraph from an Indian judgment. The seven candidates are `none`, `facts`, `issue`, `petitioner_argument`, `respondent_argument`, `reasoning` and `decision`. The input contains only the text the developer supplies.

## Run it locally

Use Python 3.11 and install the dependencies with `python -m pip install -r requirements.txt`. If you received the portable edit archive, extract it and first run `python rebuild.py`. That step fetches the public 438 MB original checkpoint once and verifies both checksums. You can use an existing copy with `python rebuild.py --source /path/to/original.safetensors`. The complete model directory is ready for inference immediately.

```bash
python predict.py "Notice was served on 3 May. We therefore allow the appeal."
```

Verified output for this paragraph is `{"label": "decision"}`. Call `JudgmentRoleClassifier().predict(text)` to reuse the loaded model across requests. Whitespace-only input raises a validation error. The same method accepts a sentence or a paragraph. Long inputs use overlapping 510-token windows across the entire excerpt. The classifier combines their logits by nonduplicated token coverage to return one role. The full FP32 weights contain 109,487,623 parameters and occupy 437,974,028 bytes. The BERT encoder has 12 layers, hidden width 768, and 12 attention heads.

## How the direct edit works

The LegalSeg current-sentence InLegalBERT supplies a pooled 768-dimensional representation and a seven-row classification head. Six authored anchors describe each role. For each output row, we compute the difference between its positive anchor centroid and a predefined set of confusable role centroids. We add a multiple of this direction divided by its squared length to that row. A bias centers this change between the two centroids. A second bias term uses the logarithm of the class counts from the official training split. The experiment writes 5,376 classifier weights and seven biases to a copy of the original safetensors file. It uses zero gradient steps.

We selected the anchor margin and the train-prior coefficient from a predeclared 20-setting grid on 1,120 sampled validation sentences from 601 cases. The selected margin was 4 and the coefficient was 0.65. Six example anchors for each of seven categories and their contrasting categories are included in the reproducibility scripts. This is an activation-informed edit to an existing output layer. The linked weight-space work motivated the intervention. [Clinical weight-edit study](https://arxiv.org/abs/2611.05341) · [Horizon Labs](https://horizon-labs.ai/)

## Evidence on untouched validation judgments

We froze the edited weights, then applied both the original and the edited model to 1,120 sentences from 371 other judgments. Both samples contain 160 sentences per class. Within each partition, gold-class sampling weights recover the observed sentence prevalence. The calculated scores estimate performance on that validation partition.

| Measure | Original | Edited |
|---|---:|---:|
| Seven-class macro-F1 | 0.4054 | 0.4450 |
| Accuracy | 0.4987 | 0.5026 |

The macro-F1 gain was 0.0395. A paired bootstrap resampling whole judgments produced a 95% interval of 0.0132 to 0.0656 for that gain. Reasoning improved the most. The `none` and `decision` role F1 scores decreased.

| Role | Original F1 | Edited F1 |
|---|---:|---:|
""" + "\n".join(rows) + """

## What the evidence covers

LegalSeg annotates individual sentences. Multi-sentence and mixed-role paragraph accuracy, calibration, and transfer to another jurisdiction require separate gold evaluation. On a hand-written smoke test, the sentence “The question for determination is whether the appeal is within time.” returned `none`. The sentence “The evidence supports that inference because the two accounts agree.” also returned `none`. These observed errors matter for a developer relying on issue or reasoning labels.

The published standalone current-sentence predictions score 0.49095 macro-F1 on all 149,881 official test rows. This is a different evaluation population from the sampled validation results above. The LegalSeg paper reports 0.77 for a hierarchical model with document context. The case-aligned named hierarchical predictions we located score 0.59827 on 149,825 retained sentences from all 712 cases using the authors' preprocessing. We have not resolved the difference between that file and the paper result. The evidence above supports a validation improvement over the original weights. An overall state-of-the-art claim would require comparable full-test evidence. [LegalSeg paper](https://aclanthology.org/2025.findings-naacl.63/) · [Standalone predictions](https://huggingface.co/datasets/L-NLProc/LegalSeg_InLegalBERT_Predictions) · [Hierarchical predictions](https://huggingface.co/datasets/L-NLProc/LegalSeg_Hier_BiLSTM_CRF_Predictions)

The team archive contains the original and modified checkpoint hashes, scores by role, the raw confusion matrices, edit programs, public download links and the checksums of the official CSVs. The verified standalone model reproduces the programmed logit shift to within 0.000002 across eight test inputs, including an overlength paragraph.

## Sources and attribution

Shubham Kumar Nigam, Tanmay Dubey, Govind Sharma, Noel Shallum, Kripabandhu Ghosh and Arnab Bhattacharya released LegalSeg and the current-sentence InLegalBERT checkpoint. The author checkpoint is tagged Apache 2.0. The InLegalBERT tokenizer is tagged MIT and credits the Law and AI group at IIT Kharagpur. The package includes both license texts. [LegalSeg repository](https://github.com/ShubhamKumarNigam/LegalSeg) · [Author weights](https://huggingface.co/L-NLProc/LegalSeg_InLegalBERT) · [Tokenizer](https://huggingface.co/law-ai/InLegalBERT) · [Original dataset](https://huggingface.co/datasets/L-NLProc/LegalSeg_CSV)
"""
    extras = []
    interim = ROOT / "results/direct-interim-test.json"
    if interim.is_file():
        v = json.loads(interim.read_text())
        extras.append(f"A label-independent random audit covered {v['rows']:,} official test sentences from {v['judgments']} judgments. On those exact rows, the published standalone predictions scored {v['published_release']['macro_f1']:.4f} macro-F1, the reloaded original weights scored {v['original_reloaded']['macro_f1']:.4f} and the direct edit scored {v['edited']['macro_f1']:.4f}. The loaded original agreed with {100*v['original_replay_matches_published_fraction']:.1f}% of published predictions. The publisher assigns the same checkpoint SHA-256 to both its standalone filename and its previous-gold-label filename. File identity cannot determine which training run produced the weights. [Author checkpoint files](https://huggingface.co/L-NLProc/LegalSeg_InLegalBERT/tree/44d7925e5ab422ecf04acd5dcbb0b62a6f8d8f53/InLegalBERT)")
    full_test = ROOT / "results/direct-official-test.json"
    if full_test.is_file():
        v = json.loads(full_test.read_text())
        extras.append(f"On the full {v['rows']:,}-row test split, the direct edit scored {v['frozen_model']['edited']['macro_f1']:.4f} macro-F1. The matching local replay scored {v['frozen_model']['official_encoder_replay']['macro_f1']:.4f}. The public prediction file scored {v['published_sentence_only']['macro_f1']:.4f}.")
    evidence = ROOT / "results/gemini-augmentation-validation.json"
    if evidence.is_file():
        v = json.loads(evidence.read_text()); m=v['confirm']
        extras.append(f"A separate supervised experiment used 1,792 gold training sentences. Gemini Decision 1 reviewed 168 newly authored fictional passages and agreed with {v['checkpoint']['synthetic_training_rows']} intended labels. Adding these reviewed examples moved macro-F1 on the 1,120-sentence held-out validation sample from {m['gold_only']['macro_f1']:.4f} to {m['gold_plus_reviewed']['macro_f1']:.4f}. The paired 95% case-bootstrap interval for the change was {m['paired_document_95pct_interval_gain'][0]:.4f} to {m['paired_document_95pct_interval_gain'][1]:.4f}. These examples were authored locally. The API supplied the independent choice judgment.")
    paragraph=ROOT/"results/prr-paragraph-transfer.json"
    if paragraph.is_file():
        v=json.loads(paragraph.read_text());q=v["all"]["models"]
        extras.append(f"On {v['all']['rows']:,} genuine expert-annotated Indian Supreme Court paragraphs, the original and direct models obtained {q['original']['four_role_macro_f1']:.4f} and {q['direct']['four_role_macro_f1']:.4f} four-role macro-F1 after a predefined taxonomy projection. Party arguments are merged. Issue and none count as errors. The corpus has only 22 Argument paragraphs. {v['paragraphs_truncated_to_last_510']} paragraphs exceeded the input window. The evidence JSON includes a conservative exact-text training-overlap check. [Nikita D, Dipti Rana and Rupa Mehta, paragraph corpus, CC BY 4.0]({v['dataset_url']})")
    if extras:
        text += "\n## Further benchmark checks\n\n" + "\n\n".join(extras) + "\n"
    write(KIT / "README.md", text)
    write(KIT / "NOTICE", "AttoFlow experimental direct classification-head edit.\n"
          "Modified from L-NLProc/LegalSeg_InLegalBERT by Nigam and collaborators.\n"
          "The modification changes classifier.weight and classifier.bias.\n"
          "Tokenizer derived from law-ai/InLegalBERT, Law and AI, IIT Kharagpur.\n"
          "See README.md for sources, study authors and training provenance.\n")
    write(KIT / "requirements.txt", "torch==2.8.0\ntransformers==4.53.3\nsafetensors>=0.5,<1\nnumpy>=1.26,<3\n")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "edit.json"):
        install(SOURCE / name, KIT / name)
    install(ROOT / "references/tokenizer/LICENSE-MIT.txt", KIT / "TOKENIZER-LICENSE.txt")
    install(Path("/usr/share/common-licenses/Apache-2.0"), KIT / "LICENSE")
    for name in ("predict.py", "windowing.py", "rebuild.py", "weight_edit.py"):
        install(ROOT / name, KIT / name)
    with (KIT / "head.npz").open("wb") as stream:
        np.savez_compressed(stream, **{name: tensor(SOURCE / "model.safetensors", name)
                                      for name in ("classifier.weight", "classifier.bias")})
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt",
                 "edit.json", "requirements.txt", "predict.py", "windowing.py", "README.md", "NOTICE", "LICENSE", "TOKENIZER-LICENSE.txt"):
        install(KIT / name, MODEL / name)
    install(SOURCE / "model.safetensors", MODEL / "model.safetensors")
    rec = KIT / "reproduce"
    for name in ("encoder.py", "role_fixtures.py", "program_trial.py", "weight_edit.py", "cache_full.py",
                 "score_official_test.py", "download_references.py", "benchmark_releases.py", "import_official.py",
                 "lexical_trial.py", "discriminant_trial.py", "verify_distribution.py", "predict.py", "windowing.py", "full_window_evidence.py", "associative_edit.py",
                 "transfer_trial.py", "interim_test.py", "synthetic_review.py", "synthetic_ablation.py",
                 "fetch_prediction_files.py", "audit_all_predictions.py", "paragraph_transfer.py",
                 "train_minilm.py", "score_minilm_test.py", "mixed_encoder_trial.py", "openvino_worker.py"):
        install(ROOT / name, rec / name)
    install(ROOT / "references/papers/additional-predictions-meta.json", rec / "references/papers/additional-predictions-meta.json")
    for path in (ROOT / "synthetic").glob("*.jsonl"):
        install(path, rec / "synthetic" / path.name)
    write(rec / "README.md", """# Reproduce the direct experiment

Use the package's Python dependencies and install `requests` and `scipy`. From this directory, run `python download_references.py` to obtain the pinned public model, tokenizer and prediction files. Place the three authorized LegalSeg CSVs in one directory and run `python import_official.py --downloads /path/to/downloads`. This checks all byte counts and digests.

Run `python program_trial.py` to select and validate the direct edit. Run `python verify_distribution.py` to compare the physical checkpoint and calculated logit changes. For the complete official test, run `python cache_full.py --split test` followed by `python score_official_test.py direct`. The first step saves resumable encoder blocks and supports `--device cuda` on compatible hardware. `python benchmark_releases.py` audits the authors' public predictions.

For the conventional supervised comparison, install `fasttext==0.9.3` and run `python lexical_trial.py` followed by `python score_official_test.py lexical`. Two-thread fastText training can vary between runs. `python discriminant_trial.py` fits a shrinkage discriminant head on 256 labeled training excerpts per role using the frozen encoder.

Run `python transfer_trial.py` to score the India and UK three-role transfer slices. `python interim_test.py` reconstructs the pre-frozen uniform test audit. `python fetch_prediction_files.py` followed by `python audit_all_predictions.py` checks the broader set of archived public baselines. Their git objects or LFS digests are checked before scoring.

The original fictional passages and the recorded Gemini choice checks are included under `synthetic/`. After the gold-only discriminant trial, `python synthetic_ablation.py` reruns the supervised synthetic-data comparison without new API calls. To review your own copy from scratch, place the pre-existing reviews in another directory first, set a project key in the process environment, and run `python synthetic_review.py --live --limit 168`. This submits only the included fictional examples to the official Gemini Decisions endpoint.

Run `python paragraph_transfer.py` for the distinct four-role gold paragraph assessment. It downloads checksum-verified public archives from the authors' Mendeley Data release and compares normalized substantive sentences against the authorized LegalSeg training copy locally.

To reproduce the other encoder experiments, run `python download_references.py --experimental-parents`. Then `python mixed_encoder_trial.py` compares physical weight interpolation and `python train_minilm.py` trains the compact supervised challenger. After training finishes, run `python score_minilm_test.py` for the full official test. The default portable evaluation path uses PyTorch. An optional Intel FP32 inference path uses `openvino==2024.6.0` with a compatible local driver.
""")
    for path in sorted((ROOT / "results").glob("*.json")):
        install(path, KIT / "evidence" / path.name)
    install(ROOT / "references/official/verified-local.json", KIT / "evidence" / "official-csv-checksums.json")
    install(ROOT / "public-manifest.json", KIT / "evidence" / "public-reference-manifest.json")
    for filename in ("AttoFlow-judgment-role-results.pdf",):
        if (OUT/filename).is_file():
            install(OUT/filename,KIT/"evidence"/filename)
    final = OUT / "AttoFlow-judgment-role-edit.zip"
    stage = final.with_suffix(".part")
    with ZipFile(stage, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(KIT.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, path.relative_to(KIT))
    with ZipFile(stage) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Invalid edit archive")
    stage.replace(final)
    assert file_hash(MODEL / "model.safetensors") == check["target_sha256"]
    print(f"{final.name} {final.stat().st_size} bytes, head payload {(KIT / 'head.npz').stat().st_size} bytes")


if __name__ == "__main__":
    main()
