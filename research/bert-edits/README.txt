AttoFlow | Native InLegalBERT role-classification edits
22 November 2026

Build one of six local seven-class models from the original 109,487,623-parameter
InLegalBERT checkpoint. The model accepts an English sentence or paragraph from
an Indian judgment and returns one role for the entire supplied text. The
download contains small weight patches, the tokenizer, a CPU inference script
and the measurements in the accompanying report. Python 3.11 was used for QA.

Quick start

Run the commands from the directory containing this file. The pinned 438 MB
donor downloads directly from the authors. The build verifies its SHA-256 and
creates a new directory that Transformers can load offline.

  python3 -m venv .venv
  . .venv/bin/activate
  python -m pip install -r requirements.txt
  curl -fL 'https://huggingface.co/L-NLProc/LegalSeg_InLegalBERT/resolve/44d7925e5ab422ecf04acd5dcbb0b62a6f8d8f53/InLegalBERT/InLegalBERT%28i%29model.safetensors' -o donor.safetensors
  python build_model.py ffn-28 donor.safetensors ./role-model
  python classify.py ./role-model "For these reasons, the appeal is dismissed."

Supply a paragraph as the quoted argument or pipe UTF-8 text through stdin.
The script returns a JSON object with the label, number of content WordPieces
and number of windows. Each window fits 512 positions including two special
tokens. Longer text uses overlapping windows and a token-weighted average of
the raw seven-way scores. The defaults use two CPU threads and FP32 weights.

Available variants

  ffn-28                 Rewrite 28 final-block feed-forward neurons.
  ffn-112                Rewrite 112 neurons using role and role-pair examples.
  head-balanced          Refit the native classifier on balanced role examples.
  combined-balanced      Apply the 112-neuron patch and a head fitted to it.
  head-train-priors      Fit the head with the published training class counts.
  combined-train-priors  Apply the 112-neuron patch and the corresponding
                         prevalence-weighted head.

The 28-neuron patch had the highest observed macro-F1 among the new excerpt
edits on the naturally sampled validation sentences. Its gain over the donor
has a 95% interval spanning zero and it reduced accuracy. The balanced
combination had the highest observed score on the equal-role sample that
motivated that follow-up. The family has separate checkpoints for these
different operating conditions. All edits use public seven-role labels.

Weights and evidence

variants.json records the donor, patch digests and patch order.
model_files/LICENSE and model_files/TOKENIZER-LICENSE.txt contain the supplied
upstream license texts. Each build writes its own edit-receipt.json.

The evidence directory contains per-row predictions, evaluation cohorts,
the native tensor recipes, comparison code, metrics and checksums. The two
experiment archives preserve the paths and manifests used during evaluation.
whole-judgment-comparisons.json includes judgment-clustered confidence
intervals for every verified model. The document-experiment archive contains
the query and LayerNorm patches used with AttoFlow's separate whole-judgment
reviewer. Whole-judgment scores require the original hierarchy and referral
rule. The standalone invocation above uses one supplied excerpt.

Public references

Dataset and seven-role paper
https://huggingface.co/datasets/L-NLProc/LegalSeg_CSV
https://aclanthology.org/2025.findings-naacl.63/

Original classifier and tokenizer
https://huggingface.co/L-NLProc/LegalSeg_InLegalBERT
https://huggingface.co/law-ai/InLegalBERT
