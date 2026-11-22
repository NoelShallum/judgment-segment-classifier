"""Use the optional FP32 CPU path with the frozen context weights and FP16 pooling."""

import argparse
import json
from pathlib import Path

import classify as runtime
from context_assets import MODEL_DIR, donors_at, verify_release


def load_fast_encoder(checkpoint, config_path):
    """Run with FP32 arithmetic using the same half-rounded donor parameters."""
    import torch

    encoder, author_weight, author_bias = NATIVE_LOADER(checkpoint, config_path)
    encoder.float()

    def round_pool(module, inputs, output):
        return output.to(torch.float16)

    encoder.pooler.register_forward_hook(round_pool)
    return encoder, author_weight, author_bias


NATIVE_LOADER = runtime.load_encoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--donor-dir", default=MODEL_DIR, type=Path)
    args = parser.parse_args()
    try:
        sentences = runtime.read_sentences(args.input)
        verify_release()
        runtime.load_encoder = load_fast_encoder
        result = runtime.classify(sentences, donors_at(args.donor_dir))
    except (ValueError, OSError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps({**result, "encoding_arithmetic": "fp32 with half-rounded weights and pooler outputs"}, ensure_ascii=False))
