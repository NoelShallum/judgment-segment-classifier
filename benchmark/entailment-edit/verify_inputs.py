"""Smoke-test the actual public donor and all routed window encodings on CPU."""

from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from protocol import GROUPS, HYPOTHESES, MAX_TOKENS
from remote_capture import digest, input_records, models, public_checkpoint, read, write

HERE = Path(__file__).resolve().parent


def main():
    started = perf_counter()
    torch.set_num_threads(2)
    spec, checkpoint = public_checkpoint(local_files_only=True)
    model, encoder, head, source_bias, count = models(torch, spec, checkpoint, device="cpu")
    np.testing.assert_array_equal(head.numpy(), model.classifier.weight.detach().numpy())
    bias = model.classifier.bias.detach() if model.classifier.bias is not None else None
    if bias is not None:
        np.testing.assert_array_equal(bias.numpy(), source_bias.numpy())
    inputs = input_records(read("preparation.json"))
    hold = []
    samples = [(g, *inputs[g][len(inputs[g]) // 2]) for g in GROUPS]
    actual = []
    for group, row, text in samples:
        ids, _ = encoder.encode(*text)
        actual += ids
        hold.append((group, row, max(map(len, ids))))
    longest = max(map(len, actual))
    array = torch.full((len(actual), longest), encoder.padding, dtype=torch.int64)
    mask = torch.zeros_like(array)
    for i, ids in enumerate(actual):
        array[i, :len(ids)] = torch.tensor(ids)
        mask[i, :len(ids)] = 1
    act = []
    conn = model.classifier.register_forward_pre_hook(lambda _, args: act.append(args[0]))
    with torch.inference_mode():
        native = model(input_ids=array, attention_mask=mask).logits
        conn.remove()
        direct = torch.nn.functional.linear(act[0], head, bias)
    assert torch.isfinite(native).all() and torch.allclose(native, direct, rtol=1e-6, atol=1e-6)
    synthetic, was_truncated = encoder.encode("Earlier detail " * 500, "Notice issued " * 600, "Appeal dismissed " * 500)
    assert was_truncated and max(map(len, synthetic)) <= MAX_TOKENS
    pairs = {}
    for group, records in inputs.items():
        lengths, row_trunc = [], 0
        for _, items in records:
            candidates, short = encoder.encode(*items)
            assert len(candidates) == len(HYPOTHESES)
            lengths.extend(map(len, candidates))
            row_trunc += short
        pairs[group] = {"cases_and_rows_verified": True, "nli_pairs": len(lengths), "longest": max(lengths),
                        "median_pair_tokens": int(np.median(lengths)), "target_truncated": row_trunc}
    result = {"checkpoint_sha256": digest(checkpoint), "checkpoint_parameters": count,
              "effective_parameters": sum(p.numel() for p in model.parameters()),
              "head_shape": list(head.shape), "published_config_classifier_bias": spec["config"]["classifier_bias"],
              "checkpoint_classifier_bias": source_bias.tolist(), "effective_classifier_bias": bias.tolist() if bias is not None else None,
              "torch_version": str(torch.__version__),
              "production_inputs": pairs, "native_cpu_smoke_inputs": hold,
              "native_pairs_scored": len(array), "native_logits_max_abs_diff": float(torch.max(torch.abs(native - direct))),
              "wall_seconds": round(perf_counter() - started, 2)}
    write("native-cpu-input-qa.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
