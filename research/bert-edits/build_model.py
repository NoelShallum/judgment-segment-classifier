"""Materialize one of the frozen standalone classifiers, keeping the donor intact."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from safetensors.numpy import load_file

from weight_edit import file_hash, patched_checkpoint, tensor


HERE = Path(__file__).resolve().parent
PREFIX = "bert.encoder.layer.11."


def neuron_edits(source, patch):
    slots = patch["slots"]
    if slots.ndim != 1 or not np.issubdtype(slots.dtype, np.integer):
        raise ValueError("Neuron indices must be a vector of integers.")
    if len(np.unique(slots)) != len(slots) or np.any((slots < 0) | (slots >= 3072)):
        raise ValueError("Repeated or out-of-range neuron index.")
    targets = {
        PREFIX + "intermediate.dense.weight": ("intermediate_weight", (slots, slice(None))),
        PREFIX + "intermediate.dense.bias": ("intermediate_bias", (slots,)),
        PREFIX + "output.dense.weight": ("output_weight", (slice(None), slots)),
    }
    changes = {}
    for name, (field, index) in targets.items():
        weight = tensor(source, name)
        edit = patch[field]
        if edit.shape != weight[index].shape or edit.dtype != np.float32 or not np.isfinite(edit).all():
            raise ValueError("Incompatible neuron patch for " + field)
        weight[index] = edit
        changes[name] = weight
    return changes


def main():
    manifest = json.loads((HERE / "variants.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=manifest["variants"])
    parser.add_argument("donor", type=Path, help="Original public InLegalBERT safetensors file")
    parser.add_argument("output", type=Path, help="New directory for a local Transformers model")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("Choose a new output directory.")
    if file_hash(args.donor) != manifest["donor_sha256"]:
        parser.error("The donor differs from the pinned LegalSeg file.")
    selection = manifest["variants"][args.variant]
    changes = {}
    for filename in selection["patches"]:
        path = HERE / "patches" / filename
        if file_hash(path) != manifest["patch_sha256"][filename]:
            raise ValueError("Checksum mismatch for " + filename)
        patch = load_file(str(path))
        if set(patch) == {"slots", "intermediate_bias", "intermediate_weight", "output_weight"}:
            replacement = neuron_edits(args.donor, patch)
        elif set(patch) == {"classifier.weight", "classifier.bias"}:
            replacement = patch
        else:
            raise ValueError("Unknown patch format")
        if changes.keys() & replacement.keys():
            raise ValueError("Two patches replace the same weight")
        changes.update(replacement)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
        staged = Path(scratch) / "classifier"
        staged.mkdir()
        model = staged / "model.safetensors"
        changed = patched_checkpoint(args.donor, model, changes)
        final_hash = file_hash(model)
        if selection.get("checkpoint_sha256") and selection["checkpoint_sha256"] != final_hash:
            raise ValueError("The materialized checkpoint has an unexpected checksum.")
        for filename in manifest["support_files"]:
            shutil.copy2(HERE / "model_files" / filename, staged / filename)
        receipt = dict(variant=args.variant, donor_sha256=manifest["donor_sha256"],
                       checkpoint_sha256=final_hash, changed=changed)
        (staged / "edit-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        staged.rename(output)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
