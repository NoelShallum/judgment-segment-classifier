"""Extract training content states on one CPU core, keeping one encoder layer resident at a time."""
import argparse
import fcntl
import hashlib
import json
import time
from contextlib import ExitStack

import numpy as np
import torch
from safetensors import safe_open
from tokenizers import Tokenizer
from transformers import ModernBertConfig
from transformers.modeling_utils import no_init_weights
from transformers.models.modernbert.modeling_modernbert import ModernBertEncoderLayer

from attention_write import normal
from cohort import case_rows
from fetch import MODEL, ROOT, sha
from readout_study import raw_mask


def blocks(records, tokenizer, protocol):
    prefix = [50281] + tokenizer.encode(protocol["prefix"], add_special_tokens=False).ids
    anchor = tokenizer.encode(protocol["anchor"], add_special_tokens=False).ids
    tokens = list(prefix)
    rows, cues, positions, offsets = [], [], [], [0]
    for row, (text, _) in enumerate(records):
        ids, keep = raw_mask(tokenizer, text)
        if len(ids) > 7999 - len(prefix) - len(anchor) - 1:
            raise ValueError("A new overlength population requires the registered retention policy")
        if rows and (len(rows) == 8 or len(tokens) + len(ids) + len(anchor) + 1 > 512):
            yield tokens + [50282], rows, cues, positions, offsets
            tokens = list(prefix)
            rows, cues, positions, offsets = [], [], [], [0]
        rows.append(row)
        positions.extend((len(tokens) + np.flatnonzero(keep)).tolist())
        offsets.append(len(positions))
        tokens.extend(ids.tolist())
        tokens.extend(anchor)
        cues.append(len(tokens) - 1)
    if rows:
        yield tokens + [50282], rows, cues, positions, offsets


def write_npz(path, **fields):
    temp = path.with_suffix(".part")
    with temp.open("wb") as f:
        np.savez(f, **fields)
    temp.replace(path)


def run(wave_size, maximum):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    plan = json.loads((ROOT / "packed-full-plan.json").read_text())
    protocol = json.loads((ROOT / "packed-plan.json").read_text())
    tag = sha(ROOT / "content-plan.json")
    case_ids = plan["fixed_fit_ids"][:maximum]
    docs = case_rows("train", set(case_ids))
    layout_root = ROOT / "features/ov-GPU-f32/packed-ordered"
    out = ROOT / "content-hidden"
    locks = out / ".locks"
    locks.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    cfg = ModernBertConfig.from_json_file(str(MODEL / "config.json"))
    cfg._attn_implementation = "sdpa"
    cfg.reference_compile = False
    weight_file = MODEL / "model-00001-of-00001.safetensors"
    with torch.inference_mode(), safe_open(str(weight_file), "pt", device="cpu") as weights:
        for offset in range(0, len(case_ids), wave_size):
            started = time.monotonic()
            with ExitStack() as active_locks:
                planned, references, pooled, layout_hash = {}, {}, {}, {}
                for cid in case_ids[offset:offset + wave_size]:
                    lock = active_locks.enter_context((locks / ("train-" + cid)).open("a"))
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                    summary = out / ("train-" + cid + ".json")
                    if summary.exists():
                        done = json.loads(summary.read_text())
                        if done["plan"] != tag or done["file"] != sha(summary.with_suffix(".npz")):
                            raise ValueError("Unexpected saved feature recipe or digest")
                        continue
                    layout = list(blocks(docs[cid], tokenizer, protocol))
                    evidence = json.loads((layout_root / "layouts" / ("train-" + cid + ".json")).read_text())
                    packed = [(ids, cues, rows) for ids, rows, cues, _, _ in layout]
                    digest = hashlib.sha256(json.dumps(packed).encode()).hexdigest()
                    assert digest == evidence["sha256_input_ids_cues_targets"]
                    assert evidence["invalid_label_counterfactual_identical"]
                    layout_hash[cid] = digest
                    with np.load(layout_root / ("train-" + cid + ".npz")) as ref:
                        assert ref["idx"].tolist() == list(range(len(docs[cid])))
                        references[cid] = ref["hidden"]
                    pooled[cid] = np.empty_like(references[cid])
                    for bix, entry in enumerate(layout):
                        planned[cid, bix] = entry
                if not planned:
                    continue

                embedding = weights.get_tensor("model.embeddings.tok_embeddings.weight")
                embed_norm = weights.get_tensor("model.embeddings.norm.weight").float().numpy()
                final_norm = weights.get_tensor("model.final_norm.weight").float().numpy()
                buffers, cached = {}, {}
                for key, (ids, rows, cues, positions, seg) in planned.items():
                    cid, bix = key
                    prior = ROOT / "attention-write/cpu-probes" / (cid + ".npz")
                    if bix == 0 and prior.exists():
                        with np.load(prior) as z:
                            assert (z["input_ids"].tolist(), z["cue_positions"].tolist(), z["rows"].tolist()) == (ids, cues, rows)
                            cached[key] = torch.from_numpy(z["before_terminal_residual"])[None]
                        continue
                    buffers[key] = torch.from_numpy(normal(embedding[ids].float().numpy(), embed_norm))[None]
                del embedding
                masks = {}
                for ids, *_ in planned.values():
                    length = len(ids)
                    if length in masks:
                        continue
                    pos = torch.arange(length)[None]
                    attention = torch.zeros((1, 1, length, length))
                    local = attention.masked_fill((pos - pos.T).abs() > 64, -3e38)
                    masks[length] = (pos, attention, local)

                drift = dict.fromkeys(pooled, 0.0)
                words = dict.fromkeys(pooled, 0)
                for layer_no in range(28):
                    with no_init_weights():
                        layer = ModernBertEncoderLayer(cfg, layer_no)
                    prefix = f"model.layers.{layer_no}."
                    layer.load_state_dict({key: weights.get_tensor(prefix + key).float() for key, _ in layer.named_parameters()})
                    layer.eval()
                    if layer_no == 27:
                        buffers.update(cached)
                    for key, z in list(buffers.items()):
                        ip, am, lm = masks[z.shape[1]]
                        h = layer(z, position_ids=ip, attention_mask=am, sliding_window_mask=lm)[0]
                        if layer_no < 27:
                            buffers[key] = h
                            continue
                        cid, bix = key
                        ids, rows, cues, pos, parts = planned[key]
                        norm = normal(h[0].numpy(), final_norm)
                        drift[cid] = max(drift[cid], float(np.max(np.abs(norm[cues] - references[cid][rows]))))
                        if drift[cid] > 1.e-3:
                            raise ValueError(("CPU reference diverged", cid, bix, drift[cid]))
                        word = norm[pos]
                        for j, r in enumerate(rows):
                            pooled[cid][r] = word[parts[j]:parts[j + 1]].mean(0)
                        words[cid] += len(word)
                        store = out / "token-states" / "train" / cid
                        store.mkdir(parents=True, exist_ok=True)
                        pos = np.asarray(pos, np.int32)
                        ids = np.asarray(ids, np.int32)
                        write_npz(store / f"{bix:04d}.npz", rows=rows, offsets=parts, hidden=word, token_ids=ids[pos],
                                  input_ids=ids, content_positions=pos, cue_positions=cues, before_terminal_residual=z[0].numpy())
                    del layer
                    if layer_no in (0, 9, 18, 27):
                        print("training wave", list(pooled), "layer", layer_no, "blocks", len(planned),
                              "seconds", round(time.monotonic() - started, 1), flush=True)
                for cid in pooled:
                    target = out / ("train-" + cid + ".npz")
                    write_npz(target, positions=np.arange(len(pooled[cid])), content_mean=pooled[cid], input_hash=layout_hash[cid])
                    summary = dict(file=sha(target), plan=tag, backend="torch-f32-cpu-streamed-layers",
                                   blocks=sum(key[0] == cid for key in planned), rows=len(pooled[cid]),
                                   content_tokens=words[cid], max_known_cue_abs_error=drift[cid],
                                   total_wave_seconds=time.monotonic() - started,
                                   full_precision_token_state_folder=str((out / "token-states" / "train" / cid).resolve()))
                    target.with_suffix(".json").write_text(json.dumps(summary, indent=2))
                    print("saved all training rows", cid, summary["rows"], "max cue drift", drift[cid], flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--wave-size", type=int, default=2)
    cli.add_argument("--maximum", type=int, default=12)
    args = cli.parse_args()
    run(args.wave_size, args.maximum)
