"""Apply the frozen numerical edit to the native public model on CPU."""

from time import perf_counter

import numpy as np
from safetensors.torch import load_file
import torch

from protocol import GROUPS, digest
from remote_capture import input_records, models, public_checkpoint, read, write


def main():
    from solve import WEIGHTS

    started = perf_counter()
    torch.set_num_threads(2)
    lock = read("freeze.json")
    assert digest(WEIGHTS) == lock["weights_sha256"]
    spec, checkpoint = public_checkpoint(local_files_only=True)
    model, pair_encoder, native_weight, _, _ = models(torch, spec, checkpoint, device="cpu")
    edit = load_file(str(WEIGHTS), device="cpu")
    assert torch.equal(native_weight, edit["source"])
    deltas = edit["classifier.weight"] - native_weight
    assert torch.equal(edit["classifier.weight"][1], native_weight[1])
    assert torch.allclose(deltas[0], -deltas[2], rtol=3e-6, atol=2e-8)
    n_modified = int(torch.count_nonzero((deltas != 0).sum(1)))
    assert n_modified == 2
    if model.classifier.bias is not None:
        assert torch.equal(model.classifier.bias, edit["classifier.bias"])
    records = input_records(read("preparation.json"))
    samples = [(g, *records[g][len(records[g]) // 3]) for g in GROUPS]
    pairs = [pair for _, _, sample in samples for pair in pair_encoder.encode(*sample)[0]]
    width = max(map(len, pairs))
    toks = torch.full((len(pairs), width), pair_encoder.padding, dtype=torch.int64)
    mask = torch.zeros_like(toks)
    for i, p in enumerate(pairs):
        toks[i, :len(p)], mask[i, :len(p)] = torch.tensor(p), 1
    versions = {name: tensor._version for name, tensor in model.named_parameters()}
    hidden = []
    hook = model.classifier.register_forward_pre_hook(lambda _, a: hidden.append(a[0]))
    with torch.inference_mode():
        src = model(input_ids=toks, attention_mask=mask).logits
        model.classifier.weight.copy_(edit["classifier.weight"])
        after = model(input_ids=toks, attention_mask=mask).logits
        direct = torch.nn.functional.linear(hidden[0], edit["classifier.weight"], model.classifier.bias)
        hook.remove()
    assert torch.equal(hidden[0], hidden[1]) and torch.equal(src[:, 1], after[:, 1])
    assert torch.allclose(after, direct, rtol=1e-6, atol=1e-6)
    assert not torch.equal(src[:, 0], after[:, 0]) and not torch.equal(src[:, 2], after[:, 2])
    touched = [name for name, tensor in model.named_parameters() if tensor._version != versions[name]]
    assert touched == ["classifier.weight"]
    difference = ((after[:, 0] - after[:, 2]) - (src[:, 0] - src[:, 2])).reshape(-1, 7)
    adjusted = hidden[0] @ (deltas[0] - deltas[2])
    np.testing.assert_allclose(difference.flatten().numpy(), adjusted.numpy(), atol=2e-5, rtol=2e-5)
    result = {"donor_sha256": digest(checkpoint), "serialized_sha256": digest(WEIGHTS),
              "test_focus_rows": [[g, row] for g, row, _ in samples], "test_native_forward_pass_pairs": len(pairs),
              "changed_parameter_tensor": touched[0], "touched_rows": [0, 2], "unchanged_row": 1,
              "unchanged_classifier_bias": lock["native_preserved_bias"], "donor_backbone_exact_same": True,
              "exact_head_parity_max_difference": float(torch.max(torch.abs(after - direct))),
              "max_delta_logit": float(torch.max(torch.abs(difference))), "cpu_wall_seconds": round(perf_counter() - started, 2)}
    write("physical-readout-replay.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
