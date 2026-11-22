# AttoFlow LegalSeg reviewer

This package labels the sentences of an English Indian judgment in order. It returns one of the seven LegalSeg rhetorical roles per sentence and groups adjacent same-role labels into spans. It includes the frozen AttoFlow document pipeline and the edited reviewer. The separate public InLegalBERT donor is approximately 418 MiB.

## Install and run

The verified runtime uses Linux, Python 3.11, the CPU PyTorch 2.8.0 wheel and a C++17 `g++` compiler. From the extracted archive, run:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
python -m pip install -r requirements.txt
python assets.py verify
python assets.py install
python classify.py --input example.json
```

The installer downloads the pinned public donor without credentials and verifies its SHA-256. If you already have that exact donor file, pass `--donor /path/to/donor.safetensors` to `classify.py` or run `python assets.py install --from-file /path/to/donor.safetensors`. The package verifies the donor, component files and individual reviewer tensors before inference. Inference uses local files, and the first run compiles the bundled sequence decoder into `runtime/cache`.

Supply a UTF-8 JSON array with the nonblank sentences of one judgment in order. `labels` returns a label for each item. Each entry in `spans` uses a zero-based `start_sentence` and an exclusive `end_sentence`. `referrals` lists the indices sent to InLegalBERT. The document pipeline sees all supplied sentences. InLegalBERT reviews the highest-uncertainty 20% per judgment using windows of up to 96 wordpieces.

## Benchmark

On 149,868 nonblank sentences in the 712 official LegalSeg test judgments, the closed-form edit improved macro-F1 from 64.39 to 64.59 and accuracy from 66.00% to 66.11%. The macro-F1 gain had a paired 95% interval of +0.10 to +0.30 points. The project had previously examined the official test split. Candidate selection used separate judgments from official train and validation. The LegalSeg paper reports a 77.00 macro-F1 reference. [LegalSeg paper](https://aclanthology.org/2025.findings-naacl.63.pdf#page=7)

The edit writes a centered rank-six update into the reviewer’s seven-column decision matrix and bias. The official result is a locked replay of the confidence route with the historical FP32 representation cache. The CPU CLI recomputes the representations from the supplied sentences. The packaged validation check compares its actual outputs with saved T4 predictions on two whole judgments. To repeat the check with the pinned public `val.csv`, run:

```bash
python verify_validation.py --validation /path/to/val.csv --donor models/InLegalBERT.safetensors
```

Official scores, paired intervals, archived prediction arrays, validation metrics and the freeze receipts are under `evidence/`. The source for the weight write, native parity audit and official replay is under `research/`. Those research scripts expect the original project feature caches. The inference command requires only the files in this archive, the pinned donor and the documented CPU dependencies.

## Sources

- InLegalBERT donor and license. [Pinned LegalSeg InLegalBERT](https://huggingface.co/L-NLProc/LegalSeg_InLegalBERT/tree/44d7925e5ab422ecf04acd5dcbb0b62a6f8d8f53/InLegalBERT)
- Bundled LegalSeg BiLSTM-CRF tensors and vocabulary. [Pinned LegalSeg hierarchy](https://huggingface.co/L-NLProc/LegalSeg_Hier_BiLSTM-CRF/tree/837c2a85ce9a6df21c9e3ef6d1b908b23b66d985/Hier_BiLSTM-CRF)
- Validation CSV. [Public LegalSeg CSV release](https://huggingface.co/datasets/L-NLProc/LegalSeg_CSV/tree/main)
