"""Assemble the standalone CPU LegalSeg anchor package from verified model archives."""

import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
from shutil import copyfile
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from safetensors.numpy import load_file

from fit_anchor import HERE, hash_file, rank, read, write
from verify_stages import digest, npz


WORKSPACE = HERE.parents[3]
OUTPUT = WORKSPACE / "outputs/judgment-role"
CONTEXT = OUTPUT / "AttoFlow-LegalSeg-context-model.zip"
CONTEXT_SHA = "d0c2a77f38d6c622c2c2c4421cccf4ebc4c37905faa01e28d6027b326c8a5c32"
CASES = {"6045": "confirm", "5922": "fresh"}
LABELS = ("none", "facts", "issue", "petitioner_argument", "respondent_argument", "reasoning", "decision")


def patch_file(path, replacements):
    source = path.read_text(encoding="utf-8")
    for original, change in replacements:
        assert source.count(original) == 1
        source = source.replace(original, change)
    path.write_text(source, encoding="utf-8")


def tensor_receipt(path):
    return {k: {"shape": list(p.shape), "dtype": p.dtype.str, "sha256": sha256(p.tobytes(order="C")).hexdigest()}
            for k, p in load_file(path).items()}


def exact_archive(archive):
    manifest = json.loads(archive.read("SHA256SUMS.json"))
    assert set(manifest) == set(archive.namelist()) - {"SHA256SUMS.json"}
    for n, value in manifest.items():
        assert digest(archive, n) == value, n


def stage_select(dev):
    fit = json.loads(dev.read("head-fit/freeze.json"))
    arm, size = fit["selection"], fit["cases"]
    q = (fit["incumbent_selector"] if arm == "doc-semantic" else fit["ridges"][arm]["selection"])
    return {"chosen_cases": size, "chosen_arm": arm, "preview": True,
            "stages": {str(size): {"freeze_sha256": digest(dev, "head-fit/freeze.json"), "quality": rank(q)}}}


def bind_cpu_receipt(package, receipt, candidate, arm, cases):
    proof = read(receipt)
    with ZipFile(candidate) as tested:
        assert proof["release_manifest_sha256"] == digest(tested, "SHA256SUMS.json")
        manifest = json.loads(tested.read("SHA256SUMS.json"))
        runtime = ("classify_anchor.py", "context_assets.py", "classify.py", "classify_fast.py", "requirements.txt",
                   "base/classify.py", "base/assets.py", "research/fit_anchor.py", "research/legal_solve.py", "research/legal_encode.py",
                   "research/run_scale.py", "research/scale_encode.py", "research/solve.py")
        paths = [name for name in manifest if name in runtime or name.startswith(("weights/", "base/runtime/"))]
        rule = "weights/anchor/selection.json"
        paths.remove(rule)
        for name in paths:
            assert digest(tested, name) == manifest[name] == hash_file(package / name), name
        before = json.loads(tested.read(rule))
        after = read(package / rule)
        for choice in (before, after):
            assert choice["chosen_cases"] == cases and choice["chosen_arm"] == arm
        assert before["stages"][str(cases)]["freeze_sha256"] == after["stages"][str(cases)]["freeze_sha256"]
    write(package / "anchor-evidence/cpu-runtime-binding.json",
          {"tested_archive_sha256": hash_file(candidate), "same_selected_model_and_rule": True,
           "runtime_file_sha256": {n: manifest[n] for n in paths}})


def package(readout, official, receipt, candidate, dest):
    assert hash_file(CONTEXT) == CONTEXT_SHA
    with TemporaryDirectory(dir=HERE, prefix="build-release-") as directory, ZipFile(readout) as dev:
        h = Path(directory)
        exact_archive(dev)
        with ZipFile(CONTEXT) as z:
            assert all(not PurePosixPath(m).is_absolute() and ".." not in PurePosixPath(m).parts for m in z.namelist())
            assert z.testzip() is None
            z.extractall(h)
        context_notes = h / "CONTEXT_README.md"
        (h / "README.md").replace(context_notes)
        anchor = h / "weights/anchor"
        anchor.mkdir()
        evidence = h / "anchor-evidence"
        evidence.mkdir()
        for filename in ("expanded-anchors.safetensors", "freeze.json"):
            (anchor / filename).write_bytes(dev.read("head-fit/" + filename))
        fit = read(anchor / "freeze.json")
        assert fit["weights_sha256"] == hash_file(anchor / "expanded-anchors.safetensors")
        if official is None:
            choice = stage_select(dev)
            with ZipFile(HERE.parent / "scale/stage-512.zip") as z:
                (anchor / "basis.safetensors").write_bytes(z.read("basis.safetensors"))
        else:
            with ZipFile(official) as z:
                exact_archive(z)
                choice = json.loads(z.read("official-lock.json"))
                official_scores = json.loads(z.read("official-results.json"))
                assert official_scores["lock_sha256"] == digest(z, "official-lock.json")
                base_name = "basis.safetensors"
                (anchor / base_name).write_bytes(z.read(base_name))
                for f in ("official-lock.json", "official-results.json", "official-predictions.npz"):
                    (evidence / f).write_bytes(z.read(f))
                assert tuple(official_scores[k] for k in ("chosen_cases", "chosen_arm")) == (fit["cases"], fit["selection"])
                assert official_scores["weights_sha256"] == fit["weights_sha256"]
                assert official_scores["predictions_sha256"] == hash_file(evidence / "official-predictions.npz")
        assert fit["basis_sha256"] == hash_file(anchor / "basis.safetensors")
        assert choice["chosen_cases"] == fit["cases"] and choice["chosen_arm"] == fit["selection"]
        assert choice["stages"][str(fit["cases"])]["freeze_sha256"] == hash_file(anchor / "freeze.json")
        write(anchor / "selection.json", choice)
        write(anchor / "tensors.json", {name: tensor_receipt(anchor / name) for name in ("expanded-anchors.safetensors", "basis.safetensors")})
        for filename in ("results.json", "select-predictions.npz", "confirm-predictions.npz", "fresh-predictions.npz"):
            (evidence / filename).write_bytes(dev.read("head-fit/" + filename))
        for filename in ("classify_anchor.py", "verify_anchor_cli.py"):
            copyfile(HERE / filename, h / filename)
        for filename in ("fit_anchor.py", "anchor_official.py", "verify_stages.py", "prepare_anchor.py", "t4_review.py"):
            copyfile(HERE / filename, h / "research" / filename)
        for filename in ("run_scale.py", "scale_encode.py"):
            copyfile(HERE.parent / "scale" / filename, h / "research" / filename)

        base = h / "base/classify.py"
        patch_file(base, (
            ("    import mixedbread_referral_overlay as overlay\n", "    import mixedbread_referral_overlay as overlay\n"
             "    from hier_model import LABELS\n"),
            ("def predict(sentences):", "def predict(sentences, include_pre_review=False):"),
            ("            base = parent(bank, lex, unary, initial)\n", "            base = parent(bank, lex, unary, initial)\n"
             "            if include_pre_review:\n"
             "                route['pre_review_labels'] = [LABELS[i] for i in base]\n"),
        ))
        patch_file(h / "classify.py", (
            ("def encode_view(kind, input_ids, tokenizer, donor_file, window_fn):",
             "def encode_view(kind, input_ids, tokenizer, donor_file, window_fn, encoder_loader=None):"),
            ("    encoder, weight, bias = load_encoder(donor_file, base_config)",
             "    encoder, weight, bias = (encoder_loader or load_encoder)(donor_file, base_config)"),
        ))
        base_manifest = h / "base/SHA256SUMS.json"
        entries = read(base_manifest)
        entries["classify.py"] = hash_file(base)
        write(base_manifest, entries)
        expected = {}
        arm = fit["selection"]
        for case, group in CASES.items():
            ground = npz(dev, f"head-fit/{group}-predictions.npz")
            with np.load(HERE / f"validation-{group}.npz", allow_pickle=False) as prior:
                for col in ("case_ids", "offsets", "gold"):
                    np.testing.assert_array_equal(ground[col], prior[col])
                i = list(ground["case_ids"]).index(case)
                a, b = ground["offsets"][i:i + 2]
                cols = {"labels": ground[arm], "context_labels": ground["doc-semantic"],
                        "pre_review_labels": prior["pre"], "incumbent_labels": prior["reviewed"]}
                expected[case] = {"cohort": group, "predictions_sha256": digest(dev, f"head-fit/{group}-predictions.npz"),
                                  **{k: [LABELS[y] for y in values[a:b]] for k, values in cols.items()}}
        pinned = read(h / "evidence/qa-plan.json")["validation_sha256"]
        write(evidence / "qa-expected.json", {"validation_sha256": pinned, "cases": expected, "arm": arm,
                                             "fit_judgments": fit["cases"], "model_sha256": fit["weights_sha256"]})
        if receipt is not None:
            qa = read(receipt)
            assert qa["model_sha256"] == fit["weights_sha256"] and qa["validation_sha256"] == pinned
            assert set(qa["tests"]) == set(expected)
            assert all(not changes for record in qa["tests"].values() for changes in record["mismatches"].values())
            bind_cpu_receipt(h, receipt, candidate, arm, fit["cases"])
            copyfile(receipt, evidence / "cpu-qa.json")

        notes = (HERE / "ANCHOR_RELEASE.md").read_text(encoding="utf-8")
        intro = f"The 48-judgment selector chose the `{arm}` readout fitted on {fit['cases']:,} public LegalSeg training judgments."
        if official is not None:
            m = official_scores["metrics"]["selected"]
            intro += f" On the matched official test, it scored {m['macro_f1']:.2%} macro-F1 and {m['accuracy']:.2%} accuracy across 149,868 sentences."
        (h / "README.md").write_text(notes.replace("{{MODEL_SUMMARY}}", intro), encoding="utf-8")
        outer = h / "SHA256SUMS.json"
        files = sorted(p for p in h.rglob("*") if p.is_file() and p != outer)
        write(outer, {p.relative_to(h).as_posix(): hash_file(p) for p in files})
        staged = dest.with_suffix(".part")
        with ZipFile(staged, "w", ZIP_DEFLATED, compresslevel=3) as archive:
            for path in (outer, *files):
                archive.write(path, path.relative_to(h).as_posix())
        with ZipFile(staged) as archive:
            exact_archive(archive)
        staged.replace(dest)
        print(dest, dest.stat().st_size, hash_file(dest), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", required=True, type=Path)
    parser.add_argument("--official", type=Path)
    parser.add_argument("--qa", type=Path)
    parser.add_argument("--qa-archive", type=Path, default=HERE / "anchor-preview.zip")
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    package(a.dev, a.official, a.qa, a.qa_archive, a.out)
