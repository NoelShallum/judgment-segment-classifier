ATTOFLOW / SEVEN-ROLE LEGALSEG WEIGHT EDITS

Open the accompanying PDF for the full-test benchmark and the matched validation experiments. The experiments use the English seven-role corpus. None is a human annotation. Preserve whole judgment IDs when resampling. The shared score files report one label per nonblank existing sentence. Validation scores and train-derived pilot scores use different populations.

An immediately usable experiment is the portable-pooler directory. With Python 3.11+, numpy and safetensors installed, run
  python apply_pooler.py ORIGINAL.safetensors document.safetensors document.json edited-model.safetensors
from that directory. The donor link and full SHA-256 are in document.json. The script copies the original checkpoint, writes exactly two tensors, and checks the whole output file against the checkpoint verified in native PyTorch. Choose public.safetensors and public.json for the separate native-classifier variant. The document variant was evaluated with the frozen AttoFlow readout. Both are research candidates.

Each experiment directory contains its code and receipts. The UltraBERT and GLiDE directories identify their original public releases by revision. Some evaluation programs import preexisting AttoFlow research modules from work/judgment-role. Use those programs within the original project tree. The portable-pooler example is self-contained once its pinned donor has been downloaded. Per-model summaries describe controls and cohort sizes.

The portable-legal-hierarchy directory also contains a complete offline CPU classifier, with the Apache-2.0 base hierarchy, vocabulary and AttoFlow's document-conditioned bias. Install Python 3.11+, numpy, torch, safetensors and scikit-learn. Write a UTF-8 JSON array of ordered sentence strings to input.json and run python predict_document.py input.json --bias-output document-bias.safetensors. It returns each label and contiguous runs with exclusive end indices. A JSON string returns one label. The optional --method parameter accepts source for the original donor, retention for the native head plus 200 forget-gate entries, and retention-prior for both edits. The official scores apply to sentences in complete judgments. The README and model.json identify the original authors.

The gemini directory contains the matched-input comparison with previously recorded gemini-decision-1 answers. The per-item ledger contains case IDs, human labels, input hashes and predictions. The excerpt BERT specialist and the whole-judgment model use different frozen weights. The transitions directory includes all 48 actual native CRF matrices, the training method and held-out controls. The t4/device.json receipt is the hardware preflight for the T4-only Colab run.

The bert-t4 study contains verified Tesla T4 outputs for the native final-layer value edit and the independently refitted readouts. Its materialize.py recreates a full author-format safetensors checkpoint from the pinned donor and one returned study ZIP. The hierarchy and boundary directories include the reserved case controls and paired measurements prepared after resuming.

The bert-packed study includes the corrected results archive. The replay checked all twelve frozen selection scores after recovering the intended numeric order and writing every array contiguously. Use the corrected archive for the 48-case comparison. The full assessment preserves the same T4 features and row positions for the ordered, isolated and shuffled conditions.

The karpo directory preserves the completed T4 checkpoint, the frozen worker and the independent audit. The audit checks the physical readout weights, replays the saved model states against held-out predictions and recomputes the benchmark using the official labels.

Primary sources
https://aclanthology.org/2025.findings-naacl.63/
https://huggingface.co/datasets/L-NLProc/LegalSeg_CSV
https://huggingface.co/L-NLProc/LegalSeg_InLegalBERT
https://huggingface.co/L-NLProc/LegalSeg_Hier_BiLSTM-CRF
https://huggingface.co/dmis-lab/UltraBERT-150M
https://huggingface.co/fastino/GLiDE-0.4B
https://huggingface.co/fastino/gliner2.5-small-v1
https://aistudio.google.com/docs/decisions-api
