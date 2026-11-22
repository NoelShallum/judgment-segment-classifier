"""Score full-precision pooled or last-attention features in-place on the verified T4 VM."""
import argparse
import json
import zipfile

from require_t4 import require_t4


def main(mode):
    print("Verified nvidia-smi", require_t4(), flush=True)
    import numpy as np
    from fetch import ROOT, sha
    # Keep the bulk token vectors on the worker. These small files reconstruct
    # only the original answer-cue bank consumed by the frozen scoring scripts.
    dest = ROOT / "features/ov-GPU-f32/packed-ordered"
    dest.mkdir(parents=True, exist_ok=True)
    audit = json.loads((ROOT / "remote-score-sources.json").read_text())
    for name, digest in audit.items():
        assert sha(ROOT / name) == digest, name
    for f in (ROOT / "portable-cues").glob("*.npz"):
        layout = json.loads((dest / "layouts" / (f.stem + ".json")).read_text())
        with np.load(f) as ref:
            assert str(ref["layout_sha256"]) == layout["sha256_input_ids_cues_targets"]
            if not (dest / f.name).exists():
                with (dest / f.name).open("wb") as out:
                    h = ref["hidden"]
                    np.savez(out, hidden=h, idx=np.arange(len(h)))
    if mode == "means":
        from readout_study import run
        from joint_content import fit, validate
        run(True)
        fit()
        validate()
    elif mode == "diagnose":
        from attention_write import training
        from per_head import evaluate
        training(False)
        evaluate("train")
    elif mode == "freeze-q":
        from attention_write import training
        training(True)
    elif mode == "evaluate-q":
        from attention_eval import evaluate
        evaluate()
    elif mode == "archive":
        with zipfile.ZipFile(ROOT / "glide-t4-control-results.zip", "w", compression=zipfile.ZIP_DEFLATED) as pack:
            for folder in ("readout-study", "attention-write"):
                for f in (ROOT / folder).rglob("*"):
                    if f.is_file() and f.suffix in (".safetensors", ".json"):
                        pack.write(f, str(f.relative_to(ROOT)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["means", "diagnose", "freeze-q", "evaluate-q", "archive"])
    args = p.parse_args()
    main(args.mode)
