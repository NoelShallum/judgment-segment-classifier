"""Capture final classifier inputs on one verified T4 from a pinned public NLI donor."""

import gzip
import json
from pathlib import Path
import subprocess
from time import perf_counter
from zipfile import ZipFile

from protocol import GROUPS, PairEncoder, digest

HERE = Path(__file__).resolve().parent
GPU_NAMES = {"Tesla T4", "NVIDIA T4", "T4"}


def read(name):
    return json.loads((HERE / name).read_text())


def write(name, data):
    path = HERE / name
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def public_checkpoint(*, local_files_only=False):
    from huggingface_hub import hf_hub_download

    spec = read("donor.json")
    expected = dict(spec["file_sha256"], **{"model.safetensors": spec["remote_weights_sha256"]})
    for name, checksum in expected.items():
        path = Path(hf_hub_download(spec["repo"], name, revision=spec["revision"], token=False,
                                    local_files_only=local_files_only))
        if digest(path) != checksum:
            raise ValueError(f"Public checkpoint file {name} does not match pinned SHA-256")
    assert path.stat().st_size == spec["bytes"] and path.parent.name == spec["revision"]
    return spec, path


def check_gpu(torch):
    report = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                      "--format=csv,noheader"], text=True).strip().splitlines()
    n = torch.cuda.device_count()
    assert n == 1 and len(report) == 1 and report[0].split(",")[0] in GPU_NAMES
    assert torch.cuda.get_device_name(0) in GPU_NAMES
    assert torch.cuda.get_device_capability(0) == (7, 5)
    return report[0]


def input_records(prep):
    if any(digest(HERE / name) != signature for name, signature in prep["sha256"].items()
           if name != "router.safetensors"):
        raise ValueError("Input windows or plan changed after route lock")
    records = {g: [] for g in GROUPS}
    with gzip.open(HERE / "windows.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            group, row, snippet = json.loads(line)
            assert group in records and type(row) is int and row >= 0
            assert len(snippet) == 3 and all(type(text) is str for text in snippet) and snippet[1].strip()
            records[group].append((row, snippet))
    for group, entries in records.items():
        entries.sort(key=lambda record: record[0])
        assert len(entries) == len({row for row, _ in entries}) == prep["cohorts"][group]["routed_rows"]
    return records


def models(torch, spec, path, *, device="cuda"):
    from safetensors.torch import safe_open
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    model_dir = str(path.parent)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False, local_files_only=True, token=False)
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=False, local_files_only=True, token=False)
    for name in ("num_hidden_layers", "hidden_size", "num_attention_heads", "max_position_embeddings"):
        assert getattr(config, name) == spec["config"][name]
    assert config.model_type == "modernbert" and config.num_labels == 3 and not config.classifier_bias
    assert config.id2label == {0: "entailment", 1: "neutral", 2: "contradiction"}
    config.reference_compile = False
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, trust_remote_code=False, local_files_only=True, token=False, config=config,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32, attn_implementation="sdpa",
    ).eval().to(device)
    with safe_open(str(path), framework="pt", device="cpu") as archive:
        n_params = sum(archive.get_tensor(key).numel() for key in archive.keys())
        original = archive.get_tensor("classifier.weight").float()
        source_bias = archive.get_tensor("classifier.bias").float()
    assert original.shape == (3, config.hidden_size)
    assert source_bias.shape == (3,) and torch.isfinite(source_bias).all()
    model.classifier.to(dtype=torch.float32)
    with torch.no_grad():
        model.classifier.weight.copy_(original)
        if model.classifier.bias is not None:
            model.classifier.bias.copy_(source_bias)
    return model, PairEncoder(tokenizer), original, source_bias, n_params


def capture(np, torch, model, encoder, entries, group, inputs_hash):
    started = perf_counter()
    package = HERE / f"capture-{group}.npz"
    receipt_name = f"capture-{group}.json"
    receipt = HERE / receipt_name
    if package.exists() and receipt.exists():
        info = read(receipt_name)
        if info.get("input_sha256") == inputs_hash and info.get("output_sha256") == digest(package):
            print("Reused", group, info, flush=True)
            return info
    rows = np.asarray([row for row, _ in entries], dtype=np.int32)
    batch_inputs = []
    clipped = 0
    for i, (_, snippet) in enumerate(entries):
        pairs, trunc = encoder.encode(*snippet)
        clipped += trunc
        batch_inputs.extend((len(tokens), i * 7 + role, tokens) for role, tokens in enumerate(pairs))
    batch_inputs.sort(key=lambda record: record[0])
    observed = np.empty((len(batch_inputs), 768), dtype=np.float16)
    returned = np.empty((len(batch_inputs), 3), dtype=np.float32)
    prefix = []

    def hook(_, args):
        prefix.append(args[0])
        return (args[0].float(),)

    connection = model.classifier.register_forward_pre_hook(hook)
    try:
        parity, steps, start = 0., 0, 0
        with torch.inference_mode():
            while start < len(batch_inputs):
                end = min(start + 96, len(batch_inputs))
                while end > start + 1 and (end - start) * batch_inputs[end - 1][0] > 8192:
                    end -= 1
                values = batch_inputs[start:end]
                width = 8 * ((values[-1][0] + 7) // 8)
                ids = np.full((len(values), width), encoder.padding, dtype=np.int64)
                mask = np.zeros_like(ids)
                for i, (length, _, tokens) in enumerate(values):
                    ids[i, :length], mask[i, :length] = tokens, 1
                logits = model(input_ids=torch.from_numpy(ids).to("cuda"),
                               attention_mask=torch.from_numpy(mask).to("cuda")).logits
                assert len(prefix) == 1 and prefix[0].dtype == torch.float16
                linear = torch.nn.functional.linear(prefix[0].float(), model.classifier.weight, model.classifier.bias)
                parity = max(parity, torch.max(torch.abs(logits - linear)).item())
                slots = np.asarray([v[1] for v in values])
                observed[slots], returned[slots] = prefix.pop().cpu().numpy(), logits.cpu().numpy()
                steps += 1
                if steps % 40 == 0:
                    print(group, end, "/", len(batch_inputs), "pairs", round(perf_counter() - started, 1), "seconds", flush=True)
                start = end
    finally:
        connection.remove()
    assert parity <= .00005 and np.isfinite(observed).all() and np.isfinite(returned).all()
    part = package.with_suffix(".npz.part")
    with part.open("wb") as handle:
        np.savez_compressed(handle, rows=rows, hidden=observed.reshape(-1, 7, 768), logits=returned.reshape(-1, 7, 3))
    part.replace(package)
    info = {"input_sha256": inputs_hash, "output_sha256": digest(package), "rows": len(rows), "pairs": len(batch_inputs),
            "max_tokens": batch_inputs[-1][0], "target_truncated": clipped, "native_linear_max_diff": parity,
            "wall_seconds": round(perf_counter() - started, 2), "peak_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 3)}
    write(receipt_name, info)
    print(group, info, flush=True)
    return info


def main():
    import numpy as np
    import torch
    import transformers

    started = perf_counter()
    print("Verified one T4", check_gpu(torch), flush=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    gpu = check_gpu(torch)
    time_download = perf_counter()
    spec, weights = public_checkpoint()
    print("Verified pinned public checkpoint", spec["revision"], round(perf_counter() - time_download, 2), "seconds", flush=True)
    prep = read("preparation.json")
    records = input_records(prep)
    model, encoder, original, source_bias, n_params = models(torch, spec, weights)
    original_path = HERE / "native-head.npy"
    np.save(original_path, original.numpy(), allow_pickle=False)
    inputs_hash = digest(HERE / "input-manifest.json")
    captures = {group: capture(np, torch, model, encoder, records[group], group, inputs_hash) for group in GROUPS}
    result = {"gpu": gpu, "torch": str(torch.__version__), "transformers": transformers.__version__, "dtype_backbone": "float16",
              "dtype_classifier": "float32", "donor_sha256": spec["remote_weights_sha256"], "donor_revision": spec["revision"],
              "checkpoint_weight_count": n_params, "effective_parameter_count": sum(p.numel() for p in model.parameters()),
              "published_config_classifier_bias": spec["config"]["classifier_bias"],
              "checkpoint_classifier_bias": source_bias.tolist(),
              "effective_classifier_bias": model.classifier.bias.detach().cpu().tolist() if model.classifier.bias is not None else None,
              "native_head_sha256": digest(original_path), "target_limit": encoder.target_limit,
              "input_sha256": inputs_hash, "captures": captures, "wall_seconds": round(perf_counter() - started, 2)}
    write("capture-summary.json", result)
    archive = HERE / "entailment-captures.zip"
    stage = archive.with_suffix(".zip.part")
    sources = ["capture-summary.json", "native-head.npy", "input-manifest.json"] + [f"capture-{g}.npz" for g in GROUPS]
    with ZipFile(stage, "w") as package:
        for file in sources:
            package.write(HERE / file, file)
    stage.replace(archive)
    print("RESULT", archive, digest(archive), flush=True)


if __name__ == "__main__":
    main()
