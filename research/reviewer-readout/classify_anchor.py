"""Apply the selector-locked LegalSeg anchor model to ordered English sentences on CPU."""

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import classify as context_runtime
from context_assets import HERE, MODEL_DIR, WEIGHTS, donors_at, sha, verify_release


LABELS = context_runtime.LABELS


def classify(sentences, donors, arithmetic):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    sys.path.insert(0, str(HERE / "base"))
    sys.path.insert(0, str(HERE / "research"))

    import numpy as np
    import torch
    from transformers import BertTokenizerFast

    from base.classify import load_classifier
    from fit_anchor import EYE, build_design
    from legal_encode import windows
    from legal_solve import basis, head_output
    from run_scale import design
    from solve import docbasis, probabilities

    from classify_fast import load_fast_encoder

    encoder_loader = load_fast_encoder if arithmetic == "fp32" else context_runtime.load_encoder
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    initial = load_classifier(donors["incumbent"], head="incumbent")(sentences, include_pre_review=True)
    parent = np.asarray([LABELS.index(label) for label in initial["labels"]], dtype=np.int8)
    previous = np.asarray([LABELS.index(label) for label in initial["pre_review_labels"]], dtype=np.int8)
    end = np.array([0, len(parent)], dtype=np.int64)
    tokenizer = BertTokenizerFast.from_pretrained(HERE / "base/runtime/references/tokenizer", local_files_only=True)
    input_ids = tokenizer(sentences, add_special_tokens=False, return_attention_mask=False, verbose=False)["input_ids"]
    if tokenizer.pad_token_id != 0:
        raise ValueError("The pinned author models require padding token zero")

    read = context_runtime.read_weights
    weights = HERE / "weights"
    inherited = {f: read(weights, f) for f in WEIGHTS}
    inherited_rule = json.loads((weights / "freeze.json").read_text(encoding="utf-8"))["cfg"]["doc-semantic"]
    if inherited_rule["scope"] != "all" or inherited_rule["margin"] != 0:
        raise ValueError("Unexpected contextual readout rule")
    pca, representation, edited, compact, window_stats = {}, {}, {}, {}, {}
    for mode in context_runtime.VIEW:
        packet, window_stats[mode] = context_runtime.encode_view(mode, input_ids, tokenizer, donors[mode], windows, encoder_loader)
        packet.base, packet.offsets = parent, end
        representation[mode] = packet
        pca[mode] = inherited[mode + "-pca.safetensors"]
        edited[mode] = head_output(packet, inherited[mode + "-edited.safetensors"])
        vector = np.clip((packet.hidden.astype(np.float32) - pca[mode]["mu"]) @ pca[mode]["proj"], -5, 5)
        compact[mode] = np.column_stack((packet.raw, edited[mode], vector)).astype(np.float32)
    inherited_head = inherited["doc-semantic.safetensors"]
    old_x = basis(representation, edited, pca, "semantic")
    incumbent = (docbasis(old_x, inherited_head) @ inherited_head["coef"]
                 + inherited_rule["anchor"] * EYE[parent])
    original = incumbent.argmax(1)

    location = HERE / "weights/anchor"
    choice = json.loads((location / "selection.json").read_text(encoding="utf-8"))
    freeze_path = location / "freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    arm, cases = choice["chosen_arm"], choice["chosen_cases"]
    if (arm, cases) != (freeze["selection"], freeze["cases"]):
        raise ValueError("The selected arm or fit prefix is inconsistent")
    selected = choice["stages"][str(cases)]
    if sha(freeze_path) != selected["freeze_sha256"]:
        raise ValueError("The saved selector freeze has changed")
    names = ("basis.safetensors", "expanded-anchors.safetensors")
    for filename, digest in zip(names, (freeze["basis_sha256"], freeze["weights_sha256"])):
        if sha(location / filename) != digest:
            raise ValueError(f"Unexpected file SHA-256 for the anchor model {filename}")
    plan, params = (read(location, name) for name in names)
    if arm == "doc-semantic":
        labels = original
    else:
        features = np.column_stack((compact["past"], compact["both"]))
        panel = SimpleNamespace(offsets=end, compact=features, pre=previous, reviewed=parent)
        x, indices = build_design(panel, plan, design, docbasis, probabilities, arm == "exact-reviewed")
        columns, coeff = params[arm + ".columns"], params[arm + ".coef"]
        np.testing.assert_array_equal(columns, indices[arm])
        mix = freeze["ridges"][arm]["settings"][2]
        labels = ((1. - mix) * incumbent + mix * (x[:, columns] @ coeff)).argmax(1)
    roles = [LABELS[y] for y in labels]
    return {"labels": roles, "spans": context_runtime.contiguous_spans(roles),
            "incumbent_labels": initial["labels"], "pre_review_labels": initial["pre_review_labels"],
            "context_labels": [LABELS[y] for y in original], "backbone_windows": window_stats,
            "encoding_arithmetic": arithmetic, "head_sha256": freeze["weights_sha256"],
            "fit_judgments": cases, "anchor_arm": arm}


if __name__ == "__main__":
    flags = argparse.ArgumentParser(description=__doc__)
    flags.add_argument("--input", required=True, type=Path)
    flags.add_argument("--donor-dir", default=MODEL_DIR, type=Path)
    flags.add_argument("--encoding", choices=("fp32", "fp16"), default="fp32")
    args = flags.parse_args()
    try:
        verify_release()
        text = context_runtime.read_sentences(args.input)
        result = classify(text, donors_at(args.donor_dir), args.encoding)
    except (OSError, ValueError) as error:
        flags.exit(2, f"{error}\n")
    print(json.dumps(result, ensure_ascii=False))
