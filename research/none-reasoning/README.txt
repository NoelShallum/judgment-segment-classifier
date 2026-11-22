ATTOFLOW | NONE–REASONING WEIGHT EDITS

These research checkpoints apply a frozen, gradient-free rank-one correction
to the None and Reasoning rows of the original InLegalBERT classification head.
The second variant includes the separately frozen 28-neuron feed-forward edit.
Both models return one of the seven LegalSeg rhetorical roles for a supplied
English sentence or paragraph.

The 4,900-sentence validation sample and the 96-judgment document sample were
previously inspected during this research. These outcomes are exploratory.
On the isolated sample, the donor head edit moved macro-F1 from 46.78 to 47.42
and the None/Reasoning errors from 835 to 812. On complete judgments, the same
edit moved macro-F1 from 68.83 to 68.80. Please use the evidence files for
the matched controls and full uncertainty estimates.

BUILD THE CLASSIFIER

Run the commands from the extracted directory with Python 3.11 or later.

  python3 -m venv .venv
  . .venv/bin/activate
  python -m pip install -r requirements.txt
  curl -fL 'https://huggingface.co/L-NLProc/LegalSeg_InLegalBERT/resolve/44d7925e5ab422ecf04acd5dcbb0b62a6f8d8f53/InLegalBERT/InLegalBERT%28i%29model.safetensors' -o donor.safetensors
  python build_model.py none-reasoning-donor donor.safetensors ./role-model
  python classify.py ./role-model "For these reasons, the appeal is dismissed."

Use none-reasoning-28 to build the second variant. The builder checks source
and patch hashes, verifies the final checkpoint digest, and creates a new
directory. The script accepts a full sentence or paragraph as one input and
returns exactly one role. Longer texts use 512-position windows with overlap
and combine their scores.

The evidence directory includes the frozen selector, numerical results,
per-example predictions, two matched-norm shuffled controls, CPU verification
and a separate whole-judgment T4 replay. The fitting and audit scripts refer to
the data and frozen feature archives in the prior AttoFlow benchmark bundle.
The standalone inference script uses the included tokenizer files.

LegalSeg paper https://aclanthology.org/2025.findings-naacl.63/
LegalSeg checkpoint https://huggingface.co/L-NLProc/LegalSeg_InLegalBERT
