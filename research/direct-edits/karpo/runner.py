"""Use the publisher's Qwen reference forward on a verified NVIDIA T4."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import monotonic

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from assets import HERE, MODEL, READOUT_SHA, SHARD, UPSTREAM, WEIGHT_SHA, assets, checksum, read, write


PUBLISHER_PATH = HERE / "publisher"
TRANSFORMERS_VERSION = "5.21.0"
if not PUBLISHER_PATH.exists():
    PUBLISHER_PATH = HERE / ("karpo-decision-" + UPSTREAM) / "src"
sys.path.insert(0, str(PUBLISHER_PATH))


class TextProcessor:
    """Text-only adapter for the same checkpoint tokenizer and chat template."""
    def __init__(self, directory):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        self.tokenizer.padding_side = "left"
        self.chat_template = (directory / "chat_template.jinja").read_text()

    def apply_chat_template(self, messages, **kw):
        return self.tokenizer.apply_chat_template(messages, chat_template=self.chat_template, **kw)


def input_encoder(assets_dir, native_processor=False):
    from karpo_decision.config import DecisionConfig
    from karpo_decision.encoding import Encoder
    from karpo_decision.templates import template
    config = DecisionConfig.load(assets_dir)
    assert not config.backbone and len(config.codes) == 255 and config.readout_dtype == "float32"
    if native_processor:
        from karpo_decision.families.qwen import load_processor
        processor = load_processor(assets_dir, config.max_pixels)
    else:
        processor = TextProcessor(assets_dir)
    for code, token_id in zip(config.codes, config.token_ids, strict=True):
        assert processor.tokenizer.encode(code, add_special_tokens=False) == [token_id]
    return config, Encoder(processor, config.codes, template(config.template), config.marker)


def request(texts, indices, plan, order=None, course=None, replacement=None):
    canon = tuple(range(7))
    options = canon if order is None else order
    asked = tuple(indices) if course is None else tuple(course)
    assert sorted(asked) == sorted(indices)
    starts, stop = min(indices), max(indices) + 1
    state = "Ordered excerpts from one Indian court judgment. Each line begins with its original sentence number.\n"
    state += "\n".join(f"[{j + 1:05d}] " + (replacement if j == starts and replacement is not None else texts[j])
                        for j in range(starts, stop))
    choices = {plan["roles"][k]: plan["descriptions"][k] for k in options}
    questions = {str(j): {"type": "choice", "instructions": (
        f"Choose the rhetorical role of sentence [{j + 1:05d}] in the court judgment. "
        "Consider the speaker and the current court in the supplied state."), "criteria": choices} for j in asked}
    return {"state": state, "questions": questions}


def encode(encoder, query):
    result = encoder.plan(query, max_length=None)
    assert len(result) == 1
    encoded = result[0]
    assert encoded.keys == tuple(query["questions"]) and tuple(encoded.counts) == (7,) * len(encoded.keys)
    assert all(int(encoded.input_ids[i]) == encoder.marker_id for i in encoded.markers)
    return encoded


def clip_single(encoder, texts, i, plan):
    tok = encoder.processor.tokenizer
    original = tok.encode(texts[i], add_special_tokens=False)
    hard = plan["single_question_hard_cap"]
    empty = encode(encoder, request(texts, [i], plan, replacement=""))
    count = min(len(original), hard - empty.length - 16)
    while count > 64:
        old = tok.decode(original[:count // 2], clean_up_tokenization_spaces=False)
        new = tok.decode(original[-(count-count // 2):], clean_up_tokenization_spaces=False)
        alternative = old + "\n[Middle of overlength sentence omitted]\n" + new
        packed = encode(encoder, request(texts, [i], plan, replacement=alternative))
        if packed.length <= hard:
            return packed, alternative, {"original_text_tokens": len(original), "retained_source_tokens": count,
                                         "original_text_sha256": sha256(texts[i].encode()).hexdigest()}
        count -= max(16, packed.length - hard)
    raise ValueError(f"Unable to fit sentence {i} without losing its meaningful ends")


def blocks(encoder, texts, plan):
    start = 0
    while start < len(texts):
        count = min(plan["maximum_questions_per_request"], len(texts) - start)
        while True:
            positions = list(range(start, start + count))
            current = encode(encoder, request(texts, positions, plan))
            if current.length <= plan["normal_token_cap"] or count == 1:
                break
            count -= 1
        if current.length > plan["single_question_hard_cap"]:
            current, replacement, clip = clip_single(encoder, texts, start, plan)
        else:
            replacement, clip = None, None
        assert current.length <= plan["single_question_hard_cap"]
        yield positions, current, replacement, clip
        start += count


def digest(encoding):
    h = sha256()
    h.update(json.dumps([encoding.keys, encoding.counts]).encode())
    h.update(encoding.input_ids.contiguous().numpy().tobytes())
    h.update(encoding.markers.contiguous().numpy().tobytes())
    return h.hexdigest()


def verify_t4():
    from importlib.metadata import version
    from packaging.version import Version
    import torch
    from require_t4 import require_t4

    names = require_t4()
    assert len(names) == 1 and torch.cuda.device_count() == 1 and "T4" in torch.cuda.get_device_name(0), names
    assert torch.cuda.get_device_capability(0) == (7, 5)
    assert Version(torch.__version__) >= Version("2.14")
    assert Version(version("transformers")) == Version(TRANSFORMERS_VERSION)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    facts = {"nvidia_smi": names, "torch": torch.__version__, "cuda": torch.version.cuda,
             "transformers": version("transformers"), "device_capability": [7, 5], "compute_dtype": "torch.float32"}
    return facts


def load_stack(probe, config):
    import torch
    from karpo_decision.families.qwen import load_backbone
    from karpo_decision.families.qwen.reference import reference_hidden
    from karpo_decision.model.readout import Readout
    from safetensors import safe_open
    from safetensors.torch import load_file

    assert probe["device_capability"] == [7, 5]
    assert checksum(MODEL / "readout.safetensors") == READOUT_SHA
    with safe_open(MODEL / SHARD, "pt", device="cpu") as shard:
        sizes = [shard.get_slice(k).get_shape() for k in shard.keys()]
        dtype = {shard.get_slice(k).get_dtype() for k in shard.keys()}
    import math
    assert sum(math.prod(s) for s in sizes) == 852985920 and dtype == {"BF16"}
    weight = load_file(str(MODEL / "readout.safetensors"))["weight"]
    assert weight.shape == (255, 1024) and weight.dtype == torch.float32
    backbone = load_backbone(MODEL, torch.float32).eval().requires_grad_(False).to("cuda")
    assert {p.dtype for p in backbone.parameters()} == {torch.float32}
    assert sum(p.numel() for p in backbone.parameters()) == 852985920
    readout = Readout.from_rows(weight, dtype=torch.float32).eval().requires_grad_(False).to("cuda")
    return backbone, readout, reference_hidden


def forward(stack, encoder, config, encoded):
    import torch
    backbone, readout, fn = stack
    start = monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        states = fn(backbone, [encoded], config.attention, encoder.pad_id)
        original = readout(states)[:, :7]
    torch.cuda.synchronize()
    hidden = states.float().cpu().numpy()
    logits = original.float().cpu().numpy()
    assert hidden.shape == (len(encoded.keys), 1024) and logits.shape == (len(encoded.keys), 7)
    assert torch.isfinite(states).all() and torch.isfinite(original).all()
    fact = {"seconds": monotonic() - start, "gpu_peak_bytes": int(torch.cuda.max_memory_allocated()),
            "tokens": encoded.length, "markers": encoded.markers.tolist(), "digest": digest(encoded)}
    return hidden, logits, fact


def record_pack(packed, positions, order, clip):
    return {"targets": list(positions), "option_order": list(order), "input_ids": packed.input_ids.tolist(),
            "markers": packed.markers.tolist(), "digest": digest(packed), "clip": clip}


def diagnostics_specs(encoder, key, case, rows, plan, replacement, clip):
    canonical = tuple(range(7))
    rng = int(sha256(key.encode()).hexdigest()[:8], 16)
    rotate = (rng % 6) + 1
    order = canonical[rotate:] + canonical[:rotate]
    a = request(case, rows, plan, order=order, replacement=replacement)
    b = request(case, rows, plan, course=rows[::-1], replacement=replacement)
    assert a["state"] == b["state"] == request(case, rows, plan, replacement=replacement)["state"]
    return {"rotated-options": record_pack(encode(encoder, a), rows, order, clip),
            "reversed-questions": record_pack(encode(encoder, b), rows[::-1], canonical, clip)}


def token_only(encoder, texts, plan, destination):
    report = {}
    protocol_sha = checksum(HERE / "protocol.json")
    canonical = tuple(range(7))
    for split in ("train", "val"):
        report[split] = {}
        for case, vals in texts[split].items():
            staged = list(blocks(encoder, vals, plan))
            assert [row for keys, *_ in staged for row in keys] == list(range(len(vals)))
            ready = destination / "token-layout" / split / f"{case}.json"
            ready.parent.mkdir(exist_ok=True, parents=True)
            controls = {}
            if split == "val":
                rows, _, replacement, clip = staged[0]
                controls = diagnostics_specs(encoder, case, vals, rows, plan, replacement, clip)
                assert all(len(p["input_ids"]) <= plan["single_question_hard_cap"] for p in controls.values())
            record = {"protocol_sha256": protocol_sha, "normal": [record_pack(x, rows, canonical, clip) for rows, x, _, clip in staged],
                      "variants": controls}
            write(ready, record)
            report[split][case] = {"blocks": len(staged), "total_tokens": sum(pack.length for _, pack, _, _ in staged),
                                   "maximum_tokens": max(pack.length for _, pack, _, _ in staged),
                                   "clipped_rows": [keys[0] for keys, _, _, clip in staged if clip],
                                   "layout_sha256": checksum(ready)}
            print("layout", split, case, len(vals), report[split][case]["blocks"], report[split][case]["total_tokens"], flush=True)
    write(destination / "token-plan.json", report)
    print("tokenized", {s: {"blocks": sum(r["blocks"] for r in v.values()), "tokens": sum(r["total_tokens"] for r in v.values()),
                            "maximum_tokens": max(r["maximum_tokens"] for r in v.values())} for s, v in report.items()}, flush=True)


def smoke(stack, encoder, config, texts, plan, output, facts):
    key, rows = next(iter(texts["val"].items()))
    states = [encode(encoder, request(rows, [0], plan)), encode(encoder, request(rows, [0, 1], plan))]
    first = next(blocks(encoder, rows, plan))
    states.append(first[1])
    check = []
    for packed in states:
        x, y, info = forward(stack, encoder, config, packed)
        check.append({**info, "keys": list(packed.keys), "predicted": y.argmax(1).tolist(),
                      "state_norm": float((x ** 2).sum() ** .5)})
    write(output / "smoke.json", {"case": key, "backend": facts, "attention": config.attention,
                                  "template": config.template, "result": check, "weight_sha256": WEIGHT_SHA,
                                  "readout_sha256": READOUT_SHA, "choice_temperature": config.temperatures["choice"]})
    print("T4 native backward-free smoke", check, flush=True)


def store_block(path, facts, stack, encoder, config, packed, positions, order, clip):
    import numpy as np
    stamp = digest(packed)
    info_path = path.with_suffix(".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and info_path.exists():
        receipt = read(info_path)
        assert receipt["parameter_source"] == WEIGHT_SHA and receipt["digest"] == stamp
        assert receipt["sha256"] == checksum(path) and receipt["backend"] == facts
        assert receipt["target_rows"] == list(positions) and receipt["option_order"] == list(order)
        return receipt
    hidden, logits, info = forward(stack, encoder, config, packed)
    new = path.with_suffix(".part")
    with new.open("wb") as stream:
        np.savez_compressed(stream, hidden=hidden, logits=logits, targets=np.asarray(positions, np.int64),
                            option_order=np.asarray(order, np.int8), input_ids=packed.input_ids.numpy(),
                            marker_positions=packed.markers.numpy())
    new.replace(path)
    receipt = {**info, "sha256": checksum(path), "parameter_source": WEIGHT_SHA, "backend": facts,
               "target_rows": list(positions), "option_order": list(order), "clip": clip}
    write(info_path, receipt)
    return receipt


def unpack(pack, encoder, plan):
    import torch
    from karpo_decision.encoding import Encoded
    rows = pack["targets"]
    tokens = torch.tensor(pack["input_ids"], dtype=torch.long)
    markers = torch.tensor(pack["markers"], dtype=torch.long)
    assert len(markers) == len(rows) and tokens[markers].eq(encoder.marker_id).all()
    answer = Encoded(keys=tuple(map(str, rows)), counts=(7,) * len(rows), input_ids=tokens, markers=markers,
                     mm_token_type_ids=torch.zeros_like(tokens))
    assert answer.length <= plan["single_question_hard_cap"] and digest(answer) == pack["digest"]
    return answer


def measure_cases(stack, encoder, config, texts, plan, facts, output, split, keys, budget, diagnostics=False):
    started = monotonic()
    canonical = tuple(range(7))
    for key in keys:
        case = texts[split][key]
        logged = []
        layout = HERE / "token-layout" / split / f"{key}.json"
        if layout.exists():
            stat = read(HERE / "token-plan.json")
            assert checksum(layout) == stat[split][key]["layout_sha256"]
            supplied = read(layout)
            assert supplied["protocol_sha256"] == facts["protocol_sha256"]
            if diagnostics:
                requests = supplied["variants"]
            else:
                requests = {f"{j:04d}": v for j, v in enumerate(supplied["normal"])}
        else:
            on_site = blocks(encoder, case, plan)
            if diagnostics:
                rows, _, replacement, clip = next(on_site)
                requests = diagnostics_specs(encoder, key, case, rows, plan, replacement, clip)
            else:
                requests = {f"{j:04d}": record_pack(pack, rows, canonical, clip) for j, (rows, pack, _, clip) in enumerate(on_site)}
        for name, secured in requests.items():
            base = output / "states" / split / key
            packed = unpack(secured, encoder, plan)
            p = base / (name + ".npz")
            stored = store_block(p, facts, stack, encoder, config, packed, secured["targets"], secured["option_order"], secured["clip"])
            logged.append(stored)
            if budget and monotonic() - started >= budget:
                print("time budget reached, committed", split, key, name, flush=True)
                return False
        if not diagnostics:
            assert [t for item in logged for t in item["target_rows"]] == list(range(len(case)))
        write((output / "states" / split / key) / ("sensitivity.json" if diagnostics else "complete.json"),
              {"blocks": len(logged), "rows": sum(len(r["target_rows"]) for r in logged),
               "tokens": sum(r["tokens"] for r in logged), "seconds": sum(r["seconds"] for r in logged)})
        print(split, key, len(case), "inputs finished", len(logged), "blocks /", round(monotonic() - started, 1), "s", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("tokens", "smoke", "train", "four", "validation", "sensitivity", "four-sensitivity"))
    parser.add_argument("--out", type=Path, default=HERE / "run")
    parser.add_argument("--budget-seconds", type=float, default=0.)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plan = read(HERE / "protocol.json")
    assert checksum(HERE / "inputs.json") == plan["input_sha256"]
    texts = read(HERE / "inputs.json")
    if args.phase == "tokens":
        config, encoder = input_encoder(assets(False))
        token_only(encoder, texts, plan, args.out)
        return
    facts = verify_t4()
    facts.update(protocol_sha256=checksum(HERE / "protocol.json"), runner_sha256=checksum(Path(__file__)))
    write(args.out / "backend.json", facts)
    assets(True)
    config, encoder = input_encoder(MODEL)
    key = plan["validation_cases"][0]
    native_probe = request(texts["val"][key], [0, 1], plan)
    original = encode(encoder, native_probe)
    try:
        _, native_encoder = input_encoder(MODEL, native_processor=True)
    except (ImportError, RuntimeError, OSError) as error:
        proof = {"native_processor_check": "optional image dependencies unavailable", "reason": type(error).__name__,
                 "text_fallback_digest": digest(original)}
    else:
        reference = encode(native_encoder, native_probe)
        assert digest(reference) == digest(original), "Publisher text prompt and lightweight tokenizer differ"
        proof = {"native_processor_check": "exact token and marker parity", "digest": digest(original)}
        encoder = native_encoder
    write(args.out / "text-encoder.json", proof)
    stack = load_stack(facts, config)
    if args.phase == "smoke":
        smoke(stack, encoder, config, texts, plan, args.out, facts)
        return
    if not (args.out / "smoke.json").exists():
        raise ValueError("Run the T4 smoke before continuing")
    split = "train" if args.phase == "train" else "val"
    if split == "train":
        keys = plan["fit_cases"]
    else:
        keys = plan["fallback_validation_first_four_cases"] if args.phase.startswith("four") else plan["validation_cases"]
    measure_cases(stack, encoder, config, texts, plan, facts, args.out, split, keys, args.budget_seconds,
                  diagnostics="sensitivity" in args.phase)


if __name__ == "__main__":
    main()
