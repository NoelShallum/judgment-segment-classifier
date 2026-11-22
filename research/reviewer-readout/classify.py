"""Classify one English LegalSeg-style judgment with the two contextual encoders on CPU."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from context_assets import HERE, MODEL_DIR, WEIGHTS, donors_at, sha, verify_release


LABELS = ("none", "facts", "issue", "petitioner_argument", "respondent_argument", "reasoning", "decision")
VIEW = {"past": (-2, -1, 0), "both": (-1, 0, 1)}


def read_sentences(path):
    values = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(values, list) or not values or any(not isinstance(s, str) or not s.strip() for s in values):
        raise ValueError("Supply a JSON array of nonblank sentence strings from one judgment")
    return values


def read_weights(directory, filename):
    from safetensors.numpy import load_file

    receipts = json.loads((directory / "tensors.json").read_text(encoding="utf-8"))[filename]
    params = load_file(directory / filename)
    if set(params) != set(receipts):
        raise ValueError(f"Unexpected tensors in {filename}")
    for name, tensor in params.items():
        info = receipts[name]
        digest = hashlib.sha256(tensor.tobytes(order="C")).hexdigest()
        if (list(tensor.shape), tensor.dtype.str, digest) != (info["shape"], info["dtype"], info["sha256"]):
            raise ValueError(f"Tensor shape, type or SHA-256 mismatch in {filename}/{name}")
    return params


def load_encoder(checkpoint, config_path):
    import torch
    from safetensors import safe_open
    from transformers import BertConfig, BertModel

    config = BertConfig.from_json_file(str(config_path))
    config._attn_implementation = "sdpa"
    encoder = BertModel(config)
    with safe_open(checkpoint, framework="pt", device="cpu") as model:
        state = {k.removeprefix("bert."): model.get_tensor(k).to(torch.float16) for k in model.keys() if k.startswith("bert.")}
        weight = model.get_tensor("classifier.weight").numpy().copy()
        bias = model.get_tensor("classifier.bias").numpy().copy()
    encoder.load_state_dict(state, strict=True, assign=True)
    return encoder.eval().requires_grad_(False), weight, bias


def encode_view(kind, input_ids, tokenizer, donor_file, window_fn, encoder_loader=None):
    import numpy as np
    import torch

    base_config = HERE / "base/runtime/review/bert-config.json"
    encoder, weight, bias = (encoder_loader or load_encoder)(donor_file, base_config)
    data = []
    for j in range(len(input_ids)):
        indices = [j + shift for shift in VIEW[kind]]
        roles = [input_ids[k] if 0 <= k < len(input_ids) else [] for k in indices]
        chunks = window_fn(roles, tokenizer.sep_token_id, tokenizer.cls_token_id)
        data.extend((len(tokens), j, share, tokens) for tokens, share in chunks)
    data.sort(key=lambda window: window[0], reverse=True)
    hidden = np.zeros((len(input_ids), encoder.config.hidden_size), np.float32)
    coverage = np.zeros(len(input_ids), np.float32)
    pos = 0
    with torch.inference_mode():
        while pos < len(data):
            width = (data[pos][0] + 7) // 8 * 8
            size = max(8, min(160, 11264 // data[pos][0]))
            part = data[pos:pos + size]
            pos += len(part)
            owner = np.array([row[1] for row in part], dtype=np.int64)
            weights = np.array([row[2] for row in part], dtype=np.float32)
            ids = torch.zeros((len(part), width), dtype=torch.long)
            for i, row in enumerate(part):
                ids[i, :row[0]] = torch.tensor(row[3], dtype=torch.long)
            rep = encoder(ids, attention_mask=(ids != 0)).pooler_output.float().numpy()
            np.add.at(hidden, owner, rep * weights[:, None])
            np.add.at(coverage, owner, weights)
    np.testing.assert_allclose(coverage, 1, atol=3e-6)
    stats = {"windows": len(data), "max_tokens": data[0][0], "additional_windows": len(data) - len(input_ids)}
    return SimpleNamespace(hidden=hidden.astype(np.float16), raw=hidden @ weight.T + bias), stats


def contiguous_spans(roles):
    spans = []
    start = 0
    for end, next_label in enumerate([*roles[1:], None], 1):
        if next_label != roles[start]:
            spans.append({"start_sentence": start, "end_sentence": end, "label": roles[start]})
            start = end
    return spans


def classify(sentences, donor_paths):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    sys.path.insert(0, str(HERE / "base"))
    sys.path.insert(0, str(HERE / "research"))

    import numpy as np
    import torch
    from transformers import BertTokenizerFast

    from base.classify import load_classifier
    from legal_encode import windows
    from legal_solve import basis, head_output
    from solve import decide, docbasis

    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    initial = load_classifier(donor_paths["incumbent"], head="incumbent")(sentences)
    base = np.asarray([LABELS.index(label) for label in initial["labels"]], dtype=np.int8)
    tokenizer = BertTokenizerFast.from_pretrained(HERE / "base/runtime/references/tokenizer", local_files_only=True)
    ids = tokenizer(sentences, add_special_tokens=False, return_attention_mask=False, verbose=False)["input_ids"]
    if tokenizer.pad_token_id != 0:
        raise ValueError("The pinned author models require padding token zero")
    weights = HERE / "weights"
    tensors = {filename: read_weights(weights, filename) for filename in WEIGHTS}
    data, streams, pca, stats = {}, {}, {}, {}
    for mode in VIEW:
        packet, stats[mode] = encode_view(mode, ids, tokenizer, donor_paths[mode], windows)
        packet.base = base
        packet.offsets = np.array([0, len(base)], dtype=np.int64)
        data[mode] = packet
        pca[mode] = tensors[mode + "-pca.safetensors"]
        streams[mode] = head_output(packet, tensors[mode + "-edited.safetensors"])
    doc_file = "doc-semantic.safetensors"
    doc_head = tensors[doc_file]
    freeze = json.loads((weights / "freeze.json").read_text(encoding="utf-8"))
    rule = freeze["cfg"][freeze["selected"]]
    features = basis(data, streams, pca, "semantic")
    labels = decide(docbasis(features, doc_head) @ doc_head["coef"], base, rule["anchor"], rule["margin"], rule["scope"])
    output = [LABELS[k] for k in labels]
    return {"labels": output, "spans": contiguous_spans(output), "incumbent_labels": initial["labels"],
            "backbone_windows": stats, "head_sha256": sha(weights / doc_file)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="An ordered JSON array of nonblank English sentences")
    parser.add_argument("--donor-dir", default=MODEL_DIR, type=Path)
    args = parser.parse_args()
    try:
        text = read_sentences(args.input)
        verify_release()
        result = classify(text, donors_at(args.donor_dir))
    except (ValueError, OSError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps(result, ensure_ascii=False))
