"""Recover exact final incumbent labels from prepared review windows on one Tesla T4."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

from fit import Readout, prob
from native import donor_at, require_t4, sha


HERE = Path(__file__).resolve().parent


def write(path, item):
    staged = path.with_suffix(".part")
    staged.write_text(json.dumps(item, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    staged.replace(path)


@torch.inference_mode()
def run_bank(model, head, data, batch):
    ids, masks = data["ids"], data["mask"]
    assert ids.shape == masks.shape and ids.shape[1] == 96
    assert ids.dtype == np.int32 and masks.dtype == np.bool_ and ids.min() >= 0 and ids.max() < 30522
    assert len(ids) == len(data["positions"]) == len(data["aux"]) and np.isfinite(data["aux"]).all()
    assert np.isin(data["base"], np.arange(7)).all() and data["positions"].max() < len(data["base"])
    estimates = []
    for at in range(0, len(ids), batch):
        x = torch.as_tensor(ids[at:at + batch], device="cuda", dtype=torch.long)
        mask = torch.as_tensor(masks[at:at + batch], device="cuda", dtype=torch.bool)
        pool = model(x, attention_mask=mask).pooler_output
        structure = torch.as_tensor(data["aux"][at:at + batch], device="cuda")
        estimates.append(prob(head.raw(structure, pool)).cpu())
    score = torch.cat(estimates)
    view = SimpleNamespace(parent=data["base"], review=data["positions"], prior=data["prior"], original=data["original"])
    prediction = head.review(view, score)
    assert np.isin(np.flatnonzero(prediction != data["base"]), data["positions"]).all()
    return prediction.astype(np.int8), score.numpy()


def main(checkpoint, destination):
    hardware = require_t4()
    torch.set_num_threads(1)
    destination.mkdir(parents=True, exist_ok=True)
    receipt = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in receipt["files"].items():
        assert sha(HERE / name) == digest
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    model = donor_at(HERE, checkpoint, "cuda")
    head = Readout(HERE, "cuda")
    with np.load(HERE / "control.npz", allow_pickle=False) as control:
        tested, scores = run_bank(model, head, control, 96)
        np.testing.assert_array_equal(tested, control["reference"])
        np.testing.assert_allclose(scores, control["reference_scores"], rtol=0., atol=2e-5)
        check = {"cases": control["case_ids"].tolist(), "rows": len(tested), "reviewed": len(scores),
                 "reference_differences": 0, "maximum_absolute_score_error": float(np.max(np.abs(scores - control["reference_scores"])))}
    checked = {"device": hardware, "model_sha256": sha(checkpoint), "head_sha256": sha(HERE / "head.safetensors"),
               "fixed_control": check, "input_manifest_sha256": sha(HERE / "manifest.json"), "shards": {}}
    for member in receipt["shards"]:
        start = perf_counter()
        block = int(Path(member).stem.split("-")[1])
        source = HERE / member
        target = destination / f"reviewed-{block:02d}.npz"
        with np.load(source, allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays["case_ids"], protocol["case_ids"][64 * block:64 * (block + 1)])
            actual, values = run_bank(model, head, arrays, 96)
            staged = target.with_suffix(".part")
            with staged.open("wb") as stream:
                np.savez_compressed(stream, reference=actual, review_scores=values.astype(np.float32),
                                    **{k: arrays[k] for k in ("case_ids", "offsets", "gold", "original_row", "positions", "base")})
            staged.replace(target)
            checked["shards"][target.name] = {"input_sha256": sha(source), "file_sha256": sha(target),
                                             "rows": int(len(actual)), "reviewed": int(len(values)),
                                             "changed_from_pre_review": int(np.count_nonzero(actual != arrays["base"])),
                                             "seconds": perf_counter() - start}
        write(destination / "runtime-receipt.json", checked)
        print(target, checked["shards"][target.name], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=Path("/content/reviewer2/donor.safetensors"))
    ap.add_argument("--destination", type=Path, default=HERE / "returned")
    args = ap.parse_args()
    main(args.checkpoint, args.destination)
