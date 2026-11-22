"""Label one LegalSeg-style judgment supplied as ordered English sentences."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from assets import DEFAULT_DONOR, EDITED_SHA256, HERE, INCUMBENT_SHA256, sha, verify_donor, verify_release


def load_classifier(donor_file=DEFAULT_DONOR, head="edited"):
    """Return a CPU callable from a verified release and its pinned donor."""
    if head not in ("edited", "incumbent"):
        raise ValueError("Choose the edited or incumbent reviewer head")
    verify_release()
    checkpoint = verify_donor(donor_file)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    runtime = HERE / "runtime"
    sys.path.insert(0, str(runtime))

    import numpy as np
    import torch
    from safetensors import safe_open
    from safetensors.numpy import load_file
    from transformers import BertConfig, BertModel, BertTokenizerFast

    import mixedbread_referral_overlay as overlay
    import predict_pairwise_judgment as pairwise
    import sentence_bert_referral_write as bert
    import sentence_phrase_sequence as sequence
    import sentence_referral_utility_write as utility

    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    review_dir = runtime / "review"
    head_path = review_dir / f"{head}.safetensors"
    expected = EDITED_SHA256 if head == "edited" else INCUMBENT_SHA256
    if sha(head_path) != expected:
        raise ValueError(f"Unexpected {head} reviewer file")
    meta = json.loads((review_dir / "tensors.json").read_text(encoding="utf-8"))
    weights = load_file(head_path)
    if set(weights) != set(meta[head]):
        raise ValueError(f"Unexpected tensors in {head} reviewer")
    for name, tensor in weights.items():
        receipt = meta[head][name]
        if (list(tensor.shape) != receipt["shape"] or tensor.dtype.str != receipt["dtype"]
                or hashlib.sha256(tensor.tobytes(order="C")).hexdigest() != receipt["sha256"]):
            raise ValueError(f"Tensor SHA-256 or layout mismatch for {head}/{name}")

    tokenizer = BertTokenizerFast.from_pretrained(runtime / "references/tokenizer", local_files_only=True)
    config = BertConfig.from_json_file(str(review_dir / "bert-config.json"))
    config._attn_implementation = "eager"
    with safe_open(checkpoint, framework="pt", device="cpu") as donor:
        state = {name.removeprefix("bert."): donor.get_tensor(name) for name in donor.keys() if name.startswith("bert.")}
        author_weight = donor.get_tensor("classifier.weight")
        author_bias = donor.get_tensor("classifier.bias")
    encoder = BertModel(config)
    encoder.load_state_dict(state, strict=True, assign=True)
    encoder.eval().requires_grad_(False)
    del state
    rule = json.loads((review_dir / "review.json").read_text(encoding="utf-8"))
    gain, gap, veto = rule["rule"]
    if rule["rate"] != .20 or rule["max_tokens"] != bert.LIMIT or rule["mode"] != "hidden":
        raise ValueError("Unexpected route configuration for this release")
    parent = pairwise.predictor("direct")

    @torch.inference_mode()
    def review_pool(windows):
        pooled = []
        for start in range(0, len(windows), 24):
            rows = windows[start:start + 24]
            ids = torch.full((len(rows), bert.LIMIT), tokenizer.pad_token_id, dtype=torch.long)
            mask = torch.zeros_like(ids, dtype=torch.bool)
            for i, tokens in enumerate(rows):
                ids[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
                mask[i, :len(tokens)] = True
            pooled.append(encoder(ids, attention_mask=mask).pooler_output)
        return torch.cat(pooled)

    def predict(sentences):
        if not isinstance(sentences, list) or not sentences or any(not isinstance(s, str) or not s.strip() for s in sentences):
            raise ValueError("Supply a nonempty JSON array of nonblank sentence strings from one judgment")
        route = {}

        @torch.inference_mode()
        def revise(bank, lex, unary, initial, item, hidden, model, settings, vocab):
            base = parent(bank, lex, unary, initial)
            confidence = np.exp(unary - unary.max(1, keepdims=True))
            confidence /= confidence.sum(1, keepdims=True)
            risk = 1. - confidence[np.arange(len(base)), base]
            positions = utility.route(bank[1], risk, rule["rate"])
            row = bert.case_view(item, hidden, model, settings, vocab, positions)
            np.testing.assert_array_equal(row[2], bank[3])
            pool = review_pool(bert.pack_ids(tokenizer, [row], bert.MODES[1]))
            author_scores = (pool @ author_weight.T + author_bias).softmax(-1).numpy()
            aux = row[4]
            design = np.column_stack((aux, author_scores, author_scores * aux[:, 40:47], pool.numpy()))
            values = bert.response(design, weights)
            answer = overlay.edit(base, positions, row[5], values, gain, gap, row[2][positions] if veto else None)
            route.update(reviewed=len(positions), referrals=positions.tolist(), reviewer_changes=int(np.count_nonzero(answer != base)))
            return answer

        labels = sequence.run_input(sentences, False, "spans", revise_with_context=revise)
        return {**labels, **route, "head": head, "head_sha256": expected}

    return predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="A UTF-8 JSON file containing ordered sentences")
    parser.add_argument("--donor", default=DEFAULT_DONOR, type=Path)
    parser.add_argument("--head", choices=("edited", "incumbent"), default="edited")
    args = parser.parse_args()
    sentences = json.loads(args.input.read_text(encoding="utf-8-sig"))
    try:
        result = load_classifier(args.donor, args.head)(sentences)
    except (OSError, ValueError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
