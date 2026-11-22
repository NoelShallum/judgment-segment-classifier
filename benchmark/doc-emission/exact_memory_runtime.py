"""Compact CPU runtime for the two locked exact-context associative reads."""

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

import exact_memory as m

ARTIFACT = m.HERE / "compact-memory.safetensors"
MANIFEST = m.HERE / "compact-memory.json"


def compile_weights(path, proof):
    p = m.table()
    frozen = m.read(m.FREEZE)
    policies = {k: frozen["arms"][k]["parameters"] for k in m.TYPES}
    widths = sorted({w for ladder, _, _ in policies.values() for w in ladder})
    saved = {"train.complete_judgments": m.byte_matrix(p["train.complete_judgments"])}
    for w in widths:
        values = p[f"{w}.weight"]
        role = values.argmax(1)
        support = values.sum(1)
        fidelity = values[np.arange(len(values)), role] / support.clip(min=1)
        code = np.zeros(len(values), dtype=np.uint8)
        for i, name in enumerate(m.TYPES):
            sizes, votes, agreement = policies[name]
            if w not in sizes:
                continue
            accepted = (support >= votes) & (fidelity >= agreement)
            if name == "pair-only":
                accepted &= np.isin(role, m.PAIR)
            code[accepted] |= ((role[accepted] + 1) << (i * 3)).astype(np.uint8)
        keep = code != 0
        saved[f"{w}.keys"] = m.byte_matrix(p[f"{w}.keys"][keep])
        saved[f"{w}.roles"] = code[keep]
    tmp = path.with_suffix(".part")
    save_file(saved, str(tmp))
    tmp.replace(path)
    m.write(proof, {"weights_sha256": m.digest(path), "bytes": path.stat().st_size, "profiles": policies,
                    "source_training_weights_sha256": m.digest(m.WEIGHTS), "freeze_sha256": m.digest(m.FREEZE),
                    "label_order": m.LABELS,
                    "implementation": "Exact key lookup over the active sparse output weights, with each eligible output label quantized to three bits."})


class ExactRoleMemory:
    def __init__(self, path=ARTIFACT, manifest=MANIFEST):
        self.proof = m.read(manifest)
        assert m.digest(path) == self.proof["weights_sha256"]
        self.source = load_file(str(path))
        for k in self.source:
            if k.endswith(".keys") or k == "train.complete_judgments":
                self.source[k] = m.packed_keys(self.source[k])

    def apply(self, sentences, base, method="all-seven"):
        if not sentences or any(not isinstance(t, str) or not t.strip() for t in sentences):
            raise ValueError("Pass the nonblank sentences of one complete judgment.")
        current = np.asarray(base, dtype=np.int8)
        if current.shape != (len(sentences),) or not np.isin(current, range(7)).all():
            raise ValueError("Supply one of the seven role indexes for each sentence.")
        signature = m.sentence_signatures(sentences)
        result = current.copy()
        use = np.zeros(len(result), dtype=np.int8)
        doc = np.asarray([m.whole(signature)], dtype=m.KEY)
        if m.search(self.source["train.complete_judgments"], doc)[1][0]:
            return result, use
        bits = m.TYPES.index(method) * 3
        for w in self.proof["profiles"][method][0]:
            q = m.signature_windows(signature, w)
            ix, exact = m.search(self.source[f"{w}.keys"], q)
            code = ((self.source[f"{w}.roles"][ix] >> bits) & 7).astype(np.int8)
            valid = (use == 0) & exact & (code != 0)
            if method == "pair-only":
                valid &= np.isin(current, m.PAIR)
            result[valid] = code[valid] - 1
            use[valid] = w
        return result, use


def verify(model, groups):
    # Compare the compressed native-forward path from original strings and saved role inputs.
    counts = {}
    for group in groups:
        path = m.HERE / (group + "-predictions.npz")
        with np.load(path, allow_pickle=False) as data:
            keys = data["case_ids"].tolist()
            offsets = data["offsets"]
            keyset = set(keys)
            by_id = {k: rows for k, rows, _ in m.cases("test" if group == "official" else "val") if k in keyset}
            arms = [name for name in data if "/" in name and "." not in name]
            for name in arms:
                base, profile = name.split("/")
                base_labels = data[base]
                result = []
                sizes = []
                for k, a, b in zip(keys, offsets, offsets[1:]):
                    scores, used = model.apply(by_id[k], base_labels[a:b], profile)
                    result.append(scores)
                    sizes.append(used)
                np.testing.assert_array_equal(np.concatenate(result), data[name])
                np.testing.assert_array_equal(np.concatenate(sizes), data[name + ".context_width"])
            counts[group] = {"documents": len(keys), "sentences": int(offsets[-1]), "arms": len(arms),
                             "errors": 0, "prediction_file_sha256": m.digest(path)}
    m.write(m.HERE / "compact-parity.json", {"weights_sha256": model.proof["weights_sha256"],
            "cohorts": counts, "device": "CPU", "test_used_for_edits": False})
    print(counts, flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("operation", choices=("compile", "verify", "predict"))
    cli.add_argument("--input", type=Path)
    cli.add_argument("--weights", type=Path, default=ARTIFACT)
    cli.add_argument("--manifest", type=Path, default=MANIFEST)
    cli.add_argument("--profile", choices=m.TYPES, default="all-seven")
    cli.add_argument("--include-official", action="store_true")
    args = cli.parse_args()
    if args.operation == "compile":
        compile_weights(args.weights, args.manifest)
        return
    model = ExactRoleMemory(args.weights, args.manifest)
    if args.operation == "verify":
        groups = ("confirm", "fresh")
        verify(model, (*groups, "official") if args.include_official else groups)
        return
    if not args.input:
        cli.error("predict requires a JSON file with sentences and existing model labels")
    v = json.loads(args.input.read_text())
    base = [m.TO_ID[role] for role in v["labels"]]
    predicted, used = model.apply(v["sentences"], base, args.profile)
    print(json.dumps({"labels": [m.LABELS[i] for i in predicted], "matched_context_width": used.tolist()}))


if __name__ == "__main__":
    main()
