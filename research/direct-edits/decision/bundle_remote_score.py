"""Package small analysis inputs so full-precision token states can stay on the T4 worker."""
import hashlib
import json
import zipfile

from fetch import ROOT, sha


plan = json.loads((ROOT / "packed-full-plan.json").read_text())
files = {s: ROOT / s for s in [
    "require_t4.py", "remote_score.py", "readout_study.py", "joint_content.py", "weights.py",
    "cohort.py", "compare.py", "per_head.py", "attention_write.py", "attention_eval.py",
    "content-cue-traincv-plan.json", "content-plan.json", "attention-plan.json",
    "weights/ov-GPU-f32/packed-ordered/whole16/roles-readout.safetensors",
    "weights/ov-GPU-f32/packed-ordered/whole16/predictions.json",
]}
for split, ids in [("train", plan["fixed_fit_ids"]), ("val", plan["fixed_validation_ids"])]:
    for cid in ids:
        layout = "features/ov-GPU-f32/packed-ordered/layouts/" + split + "-" + cid + ".json"
        files[layout] = ROOT / layout
for cid in plan["fixed_validation_ids"]:
    files["portable-incumbent/" + cid + ".json"] = ROOT.parent / "pooler/shared-cases" / (cid + ".json")
dst = ROOT / "glide-t4-control-scoring.zip"
with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as pack:
    audit = {}
    for name, f in files.items():
        data = f.read_bytes()
        # Exclude all other incumbent/exploratory models. Only the frozen
        # seven-way comparator is needed on this model's prespecified cohort.
        if name.startswith("portable-incumbent/"):
            source = json.loads(data)
            data = json.dumps({"id": source["id"], "manifest_sha256": source["manifest_sha256"],
                               "predictions": {"unchanged": source["predictions"]["unchanged"]}}).encode()
        audit[name] = hashlib.sha256(data).hexdigest()
        pack.writestr(name, data)
    pack.writestr("remote-score-sources.json", json.dumps(audit))
print(dst.name, dst.stat().st_size, sha(dst))
