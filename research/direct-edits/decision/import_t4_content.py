"""Verify and import the T4 result into an isolated bank, preserving CPU feature provenance."""
import json
import zipfile
from pathlib import Path

import numpy as np

from fetch import ROOT, sha
from weights import savej


def run():
    artifact = ROOT.parent / "t4/glide-contextual-return.zip"
    device = json.loads((ROOT.parent / "t4/device.json").read_text())
    assert device["nvidia_smi"] and all("T4" in s.split(",")[0] for s in device["nvidia_smi"])
    plan = json.loads((ROOT / "packed-full-plan.json").read_text())
    expected = [(split, cid) for split, field in [("train", "fixed_fit_ids"), ("val", "fixed_validation_ids")]
                for cid in plan[field]]
    requirements = {"content-hidden/" + split + "-" + cid + ext for split, cid in expected for ext in (".json", ".npz")}
    dest = ROOT / "content-T4"
    dest.mkdir(exist_ok=True)
    with zipfile.ZipFile(artifact) as z:
        assert z.testzip() is None
        assert requirements <= set(z.namelist())
        for name in sorted(requirements):
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Invalid member name")
            path = dest / name
            path.parent.mkdir(exist_ok=True, parents=True)
            data = z.read(name)
            if path.exists():
                assert path.read_bytes() == data
            else:
                path.write_bytes(data)
    reference = ROOT / "features/ov-GPU-f32/packed-ordered"
    cohorts = {}
    max_overlap = 0.
    parity_rows = 0
    for split, cid in expected:
        tag = split + "-" + cid
        item = dest / "content-hidden" / (tag + ".npz")
        receipt = json.loads(item.with_suffix(".json").read_text())
        layout = json.loads((reference / "layouts" / (tag + ".json")).read_text())
        assert "T4" in receipt["backend"] and receipt["plan"] == sha(ROOT / "content-plan.json")
        assert sha(item) == receipt["file"] and receipt["max_known_cue_abs_error"] <= .002
        with np.load(item, allow_pickle=False) as a:
            assert str(a["input_hash"]) == layout["sha256_input_ids_cues_targets"]
            assert a["positions"].tolist() == list(range(layout["rows"]))
            assert a["content_mean"].shape == (layout["rows"], 1024) and np.isfinite(a["content_mean"]).all()
            cpu = ROOT / "content-hidden" / (tag + ".npz")
            if cpu.is_file():
                with np.load(cpu) as b:
                    assert str(a["input_hash"]) == str(b["input_hash"])
                    max_overlap = max(max_overlap, float(np.abs(a["content_mean"] - b["content_mean"]).max()))
                    parity_rows += len(a["positions"])
        record = cohorts.setdefault(split, {"cases": 0, "rows": 0, "blocks": 0, "content_tokens": 0, "max_abs_cue_drift": 0., "seconds": 0., "case_receipts": {}})
        record["cases"] += 1
        for k in ["rows", "blocks", "content_tokens", "seconds"]:
            record[k] += receipt[k]
        record["max_abs_cue_drift"] = max(record["max_abs_cue_drift"], receipt["max_known_cue_abs_error"])
        record["case_receipts"][cid] = sha(item.with_suffix(".json"))
    assert (cohorts["train"]["rows"], cohorts["val"]["rows"]) == (357, 3558)
    assert (cohorts["train"]["blocks"], cohorts["val"]["blocks"]) == (51, 460)
    out = {"zip_sha256": sha(artifact), "device": device, "cohorts": cohorts, "overlapping_independent_cpu_mean_rows": parity_rows,
           "max_abs_cpu_content_drift": max_overlap, "checked": "CRC, plan hash, each case file hash, all earlier input-layout hashes and positions, finite 1024-wide means; every originally cached FP32 answer cue was compared during the source CUDA forward."}
    savej(dest / "verified.json", out)
    print("Verified T4 means", {k: (v["cases"], v["rows"], v["blocks"], v["max_abs_cue_drift"]) for k,v in cohorts.items()},
          "CPU actual-word overlap", parity_rows, max_overlap, flush=True)


if __name__ == "__main__":
    run()
