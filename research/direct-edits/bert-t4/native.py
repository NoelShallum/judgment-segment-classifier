"""Capture exact final-layer CLS value inputs and reconstruct the native BERT tail."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
from safetensors.torch import load_file
import torch
from torch.nn import functional as F
from transformers import BertConfig, BertModel

VALUE = "bert.encoder.layer.11.attention.self.value"
HEADS = 12
WIDTH = 768
HEAD_DIM = WIDTH // HEADS


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, contents):
    p = Path(path)
    stage = p.with_suffix(".tmp")
    stage.write_text(json.dumps(contents, indent=2, allow_nan=False) + "\n")
    stage.replace(p)


def save_arrays(path, **values):
    stage = path.with_suffix(".tmp")
    with stage.open("wb") as f:
        np.savez(f, **values)
    stage.replace(path)


def require_t4():
    listing = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"], text=True
    ).strip().splitlines()
    names = [line.split(",")[0].strip() for line in listing]
    if len(names) != 1 or names[0] not in {"Tesla T4", "NVIDIA T4", "T4"}:
        raise RuntimeError(f"One NVIDIA T4 required. Assigned {names!r}.")
    if not torch.cuda.is_available() or torch.cuda.get_device_name() not in {"Tesla T4", "NVIDIA T4", "T4"}:
        raise RuntimeError("PyTorch could not verify the T4.")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {"devices": listing, "torch": str(torch.__version__), "cuda": torch.version.cuda}


def config_at(directory):
    return BertConfig.from_json_file(str(directory / "bert-config.json"))


def donor_at(directory, checkpoint, device):
    plan = read(directory / "review.json")
    if sha(checkpoint) != plan["model"]["sha256"]:
        raise ValueError("Incorrect InLegalBERT checkpoint.")
    cfg = config_at(directory)
    assert cfg.hidden_size == WIDTH and cfg.num_attention_heads == HEADS and cfg.num_hidden_layers == 12
    cfg._attn_implementation = "eager"
    model = BertModel(cfg)
    state = {k.removeprefix("bert."): v for k, v in load_file(checkpoint).items() if k.startswith("bert.")}
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval().requires_grad_(False).to(device)


class NativeTail:
    def __init__(self, model):
        self.model = model
        self.layer = model.encoder.layer[-1]
        self.attention = self.layer.attention.self

    def native_values(self, keys, weight=None, bias=None):
        layer = self.attention.value
        if weight is None:
            weight, bias = layer.weight, layer.bias
        val = torch.einsum("bhd,hvd->bhv", keys, weight.reshape(HEADS, HEAD_DIM, WIDTH))
        return (val + bias.reshape(1, HEADS, HEAD_DIM)).flatten(1)

    def pooled(self, cls, values):
        post_attention = self.layer.attention.output(values, cls)
        output = self.layer.output(self.layer.intermediate(post_attention), post_attention)
        return torch.tanh(self.model.pooler.dense(output))

    @torch.inference_mode()
    def capture(self, ids, valid):
        result = self.model(ids, attention_mask=valid, output_hidden_states=True)
        hidden = result.hidden_states[-2]
        cls = hidden[:, 0, :].contiguous()
        q = self.attention.query(cls).reshape(-1, HEADS, HEAD_DIM)
        k = self.attention.key(hidden).view(*hidden.shape[:2], HEADS, HEAD_DIM)
        affinity = torch.einsum("bhd,bthd->bht", q, k) / math.sqrt(HEAD_DIM)
        a = affinity.masked_fill(~valid[:, None, :], torch.finfo(affinity.dtype).min).softmax(-1)
        keys = torch.einsum("bht,btd->bhd", a, hidden)
        pooled = result.pooler_output
        reconstructed = self.pooled(cls, self.native_values(keys))
        err = (pooled - reconstructed).abs().max().item()
        if err > 5e-5:
            raise RuntimeError(f"Fixed-value reconstruction disagrees with the native model ({err:.3g}).")
        return {"keys": keys, "cls": cls, "pooled": pooled}, err


def batches(n, size):
    for start in range(0, n, size):
        yield slice(start, min(start + size, n))


def move(values, device, dtype=None):
    return torch.as_tensor(values, dtype=dtype, device=device)


def encode(directory, checkpoint, output, group, batch, device, take):
    started = time.perf_counter()
    device_note = require_t4() if device == "cuda" else {"devices": ["CPU for parity"]}
    plan = read(directory / "protocol.json")
    group_ids = plan["groups"][group]
    if take is not None:
        group_ids = group_ids[:take]
    tail = NativeTail(donor_at(directory, checkpoint, device))
    output.mkdir(exist_ok=True, parents=True)
    n = 0
    for case_id in group_ids:
        path = directory / "cases" / f"{case_id}.npz"
        case_sha = sha(path)
        feat = output / (case_id + ".npz")
        rec = output / (case_id + ".json")
        if feat.exists() and rec.exists():
            prev = read(rec)
            if (prev["case_sha256"], prev["sha256"], prev["backend"]) == (case_sha, sha(feat), device_note):
                n += prev["windows"]
                continue
        with np.load(path, allow_pickle=False) as x:
            assert x["case_sha256"].item() == plan["case_text_sha256"][case_id]
            ids, valid = x["ids"], x["mask"]
        parts, errors = [], []
        for b in batches(len(ids), batch):
            bank, err = tail.capture(move(ids[b], device, torch.long), move(valid[b], device, torch.bool))
            parts.append({name: value.cpu().numpy() for name, value in bank.items()})
            errors.append(err)
        save_arrays(feat, **{name: np.concatenate([p[name] for p in parts]) for name in parts[0]})
        write(rec, {"id": case_id, "backend": device_note, "case_sha256": case_sha, "sha256": sha(feat),
                    "windows": len(ids), "maximum_native_error": max(errors)})
        n += len(ids)
        print(f"{group} {case_id} · {len(ids)} reviews · {n} cumulative · native error {max(errors):.2g}", flush=True)
    print(f"encoded {group}: {len(group_ids)} cases, {n} reviews, {time.perf_counter()-started:.1f}s", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("group", choices=("fit", "select", "confirm", "shared"))
    p.add_argument("--batch", type=int, default=96)
    p.add_argument("--take", type=int)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    a = p.parse_args()
    torch.set_num_threads(1)
    encode(a.directory, a.checkpoint, a.output, a.group, a.batch, a.device, a.take)


if __name__ == "__main__":
    main()
