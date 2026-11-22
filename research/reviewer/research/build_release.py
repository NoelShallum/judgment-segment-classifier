"""Assemble, audit and write the portable reviewer archive from this workspace."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from safetensors.numpy import load_file

from assets import HERE, EDITED_SHA256, INCUMBENT_SHA256, sha


RUN = HERE.parent
ROOT = HERE.parents[2]
STAGING = HERE / "portable"
REPORTS = ROOT / "results"
FILES = ROOT / "checkpoints/hierarchical"
OLD = ROOT / "live-nov22/bert-t4"
OUTPUT = ROOT.parents[1] / "outputs/judgment-role"
DESTINATION = OUTPUT / "AttoFlow-LegalSeg-reviewer-64-59.zip"
ROLES = ("none", "facts", "issue", "petitioner_argument", "respondent_argument", "reasoning", "decision")
RUNTIME_ROOTS = ("predict_pairwise_judgment", "sentence_bert_referral_write", "sentence_phrase_sequence",
                 "sentence_referral_utility_write", "mixedbread_referral_overlay")


def write(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def copy(src, dest):
    output = STAGING / dest
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, output)


def runtime_sources():
    sources = {}
    for base in (ROOT, ROOT / "astra-role-edge", ROOT / "astra-run-penalty"):
        for path in base.glob("*.py"):
            sources.setdefault(path.stem, path)
    queue = set(RUNTIME_ROOTS)
    selected = set()
    while queue:
        name = queue.pop()
        if name in selected or name not in sources:
            continue
        selected.add(name)
        code = ast.parse(sources[name].read_text(encoding="utf-8"))
        for op in code.body:
            if isinstance(op, ast.ImportFrom) and op.module:
                queue.add(op.module.split(".")[0])
            elif isinstance(op, ast.Import):
                queue.update(target.name.split(".")[0] for target in op.names)
    for name in sorted(selected):
        origin = sources[name]
        copy(origin, Path("runtime") / origin.relative_to(ROOT))
    copy(ROOT / "packed_role_decode.cpp", "runtime/packed_role_decode.cpp")
    (STAGING / "runtime/cache").mkdir(exist_ok=True)
    (STAGING / "runtime/cache/README.txt").write_text("The runtime writes its compiled CPU sequence decoder here.\n", encoding="utf-8")


def runtime_weights():
    for name in ("sentence-document-calibrated-portable.safetensors", "sentence-ngram-residual-memory.safetensors",
                 "sentence-phrase-readout-vocabulary.json", "sentence-phrase-readout-lexical.safetensors",
                 "sentence-phrase-readout-1.safetensors", "sentence-phrase-sequence.safetensors",
                 "sentence-pairwise-repair.safetensors"):
        copy(FILES / name, "runtime/checkpoints/hierarchical/" + name)
    for tag in ("sentence-phrase-sequence", "sentence-phrase-sequence-referral", "sentence-phrase-readout",
                "sentence-phrase-readout-lexical", "sentence-dense-readout", "sentence-bert-referral", "sentence-pairwise-repair"):
        name = tag + "-frozen.json"
        copy(REPORTS / name, "runtime/results/" + name)
    for name in ("hier.safetensors", "word2idx.json"):
        source = ROOT / "references/hier" / name
        assert sha(source) == sha(OUTPUT / "hierarchical-model" / name)
        copy(source, "runtime/references/hier/" + name)
    for name in ("LICENSE", "NOTICE"):
        copy(OUTPUT / "hierarchical-model" / name, "runtime/references/hier/" + name)
    for name in ("vocab.txt", "tokenizer_config.json", "special_tokens_map.json", "LICENSE-MIT.txt"):
        copy(ROOT / "references/tokenizer" / name, "runtime/references/tokenizer/" + name)
    copy(ROOT / "astra-role-edge/vocab.json", "runtime/astra-role-edge/vocab.json")
    for name in ("bert-config.json", "review.json"):
        copy(OLD / name, "runtime/review/" + name)
    copy(OUTPUT / "model/LICENSE", "runtime/review/LICENSE")
    copy(OUTPUT / "model/NOTICE", "runtime/review/NOTICE")
    for name, src, expected in (("incumbent", OLD / "head.safetensors", INCUMBENT_SHA256),
                                ("edited", RUN / "all-role.safetensors", EDITED_SHA256)):
        assert sha(src) == expected
        copy(src, f"runtime/review/{name}.safetensors")
    meta = {}
    for variant in ("incumbent", "edited"):
        weights = load_file(STAGING / f"runtime/review/{variant}.safetensors")
        meta[variant] = {name: {"shape": list(w.shape), "dtype": w.dtype.str,
                                   "sha256": hashlib.sha256(w.tobytes(order="C")).hexdigest()}
                         for name, w in sorted(weights.items())}
    write(STAGING / "runtime/review/tensors.json", meta)


def research_files():
    for name in ("official-results.json", "official-audit.json", "official-freeze.json", "official-predictions.npz",
                 "results.json", "native-audit.json", "freeze.json", "protocol.json", "pooled-confirmation.json"):
        copy(RUN / name, "evidence/" + name)
    for name in ("reviewer_head.py", "audit_native.py", "official_readout.py"):
        copy(RUN / name, "research/" + name)
    copy(HERE / "build_release.py", "research/build_release.py")
    copy(RUN.parent / "t4-return/pool-receipt.json", "evidence/pool-receipt.json")
    val = ROOT / "references/official/val.csv"
    cases = {}
    for group, key in (("confirm", "6045"), ("fresh", "5922")):
        with np.load(RUN / (group + "-predictions.npz"), allow_pickle=False) as saved:
            num = list(saved["case_ids"]).index(key)
            start, end = map(int, saved["offsets"][num:num + 2])
            review = saved["review"]
            cases[key] = {"cohort": group, "incumbent": [ROLES[i] for i in saved["incumbent"][start:end]],
                          "edited": [ROLES[i] for i in saved["all-role"][start:end]],
                          "referrals": (review[(review >= start) & (review < end)] - start).tolist(),
                          "source_predictions_sha256": sha(RUN / (group + "-predictions.npz"))}
            assert cases[key]["incumbent"] != cases[key]["edited"]
    write(STAGING / "evidence/validation-fixtures.json", {"csv_sha256": sha(val), "judgments": cases})


def assemble():
    for name in ("assets.py", "classify.py", "verify_validation.py", "requirements.txt", "README.md", "example.json"):
        copy(HERE / name, name)
    runtime_sources()
    runtime_weights()
    research_files()


def archive(destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = {p.relative_to(STAGING).as_posix(): p for p in STAGING.rglob("*")
             if p.is_file() and p.name != "SHA256SUMS.json" and "__pycache__" not in p.parts}
    sums = {name: sha(path) for name, path in sorted(files.items())}
    write(STAGING / "SHA256SUMS.json", sums)
    temporary = destination.with_suffix(".part")
    with ZipFile(temporary, "w", ZIP_DEFLATED, compresslevel=4) as bundle:
        for name, path in sorted(files.items()):
            bundle.write(path, name)
        bundle.write(STAGING / "SHA256SUMS.json", "SHA256SUMS.json")
    with ZipFile(temporary) as bundle:
        assert bundle.testzip() is None and set(bundle.namelist()) == set(files) | {"SHA256SUMS.json"}
        for name, expected in sums.items():
            with bundle.open(name) as f:
                assert hashlib.file_digest(f, "sha256").hexdigest() == expected
    temporary.replace(destination)
    print(destination, destination.stat().st_size, sha(destination), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("assemble", "archive"))
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    args = parser.parse_args()
    if args.command == "assemble":
        assemble()
    else:
        archive(args.destination)
