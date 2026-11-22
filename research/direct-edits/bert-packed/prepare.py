"""Freeze natural, case-disjoint LegalSeg cohorts and their untruncated wordpiece IDs."""

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from transformers import BertTokenizerFast

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "bert-t4"))
sys.path.insert(1, str(ROOT))
from native import read, save_arrays, sha, write
from program_trial import TO_ID
from weight_edit import LABELS

SALT = "attoflow-inlegalbert-ordered-wordpiece-packs-v1"
CSV = ROOT / "references/official"
MANIFEST = HERE.parent / "shared-confirmation.json"


def get_cases(split, selected=None):
    rows = defaultdict(list)
    with (CSV / (split + ".csv")).open(encoding="utf-8-sig", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            key = row["Index"]
            text = row["Text"]
            if (selected is None or key in selected) and text.strip():
                rows[key].append((i, text, TO_ID[row["Label"]]))
    return rows


def fingerprint(rows):
    h = hashlib.sha256()
    for _, text, _ in rows:
        h.update(" ".join(text.casefold().split()).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def rank(name, choices):
    return sorted(choices, key=lambda key:hashlib.sha256(f"{SALT}|{name}|{key}".encode()).digest())


def choose(rows, pool, name, count, exclude):
    selected = []
    for key in rank(name, pool):
        digest = fingerprint(rows[key])
        if digest in exclude:
            continue
        selected.append(key)
        exclude.add(digest)
        if len(selected) == count:
            return selected
    raise ValueError(f"Only {len(selected)} distinct judgments in {name}.")


def freeze():
    path = HERE / "protocol.json"
    manifest = read(MANIFEST)
    assert manifest["official_csv_sha256"] == sha(CSV / "val.csv")
    common = manifest["extended_validation_judgment_ids"]
    assert common[:16] == manifest["small_validation_judgment_ids"] and len(common) == 48
    if path.exists():
        plan = read(path)
        assert plan["shared_manifest_sha256"] == sha(MANIFEST)
        assert plan["train_csv_sha256"] == sha(CSV / "train.csv")
        ids = set(plan["groups"]["fit"])
        return plan, get_cases("train", ids), get_cases("val", set(plan["groups"]["select"] + common))
    v = get_cases("val")
    all_train = [str(i) for i in range(4984)]
    t = get_cases("train", set(rank("fit", all_train)[:384]))
    fit = choose(t, t, "fit", 96, {fingerprint(doc) for doc in v.values()})
    withheld = set(common)
    reviewed = read(HERE.parent / "bert-t4/protocol.json")["groups"]
    withheld.update(reviewed["select"] + reviewed["confirm"])
    for path_, field in (("hierarchy/scale/plan.json", "validation"), ("hierarchy/chain-program/plan.json", "validation"),
                         ("hierarchy/case-head-program/plan.json", "validation")):
        p = HERE.parent / path_
        if p.exists():
            withheld.update(read(p)[field])
    select = choose(v, set(v) - withheld, "select", 32, {fingerprint(v[k]) for k in common})
    groups = dict(fit=fit, select=select, shared=common)
    assert len(set(fit + select + common)) == 96 + 32 + 48
    source = t | v
    donor = read(HERE.parent / "bert-t4/review.json")["model"]
    plan = dict(
        version=SALT, source_model=donor, roles=list(LABELS), groups=groups,
        rows={k:sum(len(source[j]) for j in ids) for k,ids in groups.items()},
        roles_per_group={k:np.bincount([row[2] for key in ids for row in source[key]], minlength=7).tolist() for k,ids in groups.items()},
        text_sha256={k:fingerprint(source[k]) for k in (fit + select + common)},
        train_csv_sha256=sha(CSV / "train.csv"), val_csv_sha256=sha(CSV / "val.csv"),
        shared_manifest_sha256=sha(MANIFEST), fit_case_limit=96, select_case_limit=32,
        independent_selection="Validation decisions are fixed before scoring shared cases. The train and select draws use natural case-level samples. An exact whole-case normalized text deduplication filter excludes fit matches anywhere in official validation and local case overlap. Earlier research may have included parts of validation.",
        context="Concatenate original sentence wordpieces with one SEP each. 510 content slots plus CLS and SEP, stride 382. Retain each original wordpiece once from its most central window. Swap the order of full sentences for the shuffled control. Encode each in isolation with the same wordpieces for the isolated control.",
        pooling="Mean of each target's retained wordpieces, separately in BERT blocks 11 and 12. Source-classifier control uses normal CLS pooling on an isolated sentence, using the first 510 wordpieces for long sentences.",
        labels_source="LegalSeg human seven-way annotations, blank texts excluded", gradient_steps=0,
        benchmark="Macro-F1 pooled over all sentences and exact same-boundary span F1 on common16/48. No official test in this package.")
    assert all(min(support) > 0 for support in plan["roles_per_group"].values())
    write(path, plan)
    return plan, t, v


def main():
    plan, train, val = freeze()
    t = BertTokenizerFast.from_pretrained(ROOT / "references/tokenizer", local_files_only=True)
    group = HERE / "cases"
    group.mkdir(exist_ok=True)
    for name, keys in plan["groups"].items():
        src = train if name == "fit" else val
        for key in keys:
            path = group / (key + ".npz")
            if path.exists():
                with np.load(path, allow_pickle=False) as cached:
                    assert cached["fingerprint"].item() == plan["text_sha256"][key]
                continue
            csv_rows, texts, labels = zip(*src[key])
            assert fingerprint(src[key]) == plan["text_sha256"][key]
            tokens = t(list(texts), add_special_tokens=False, truncation=False,
                       return_attention_mask=False, return_token_type_ids=False, verbose=False)["input_ids"]
            tokens = [x or [t.unk_token_id] for x in tokens]
            integer = np.concatenate(tokens).astype(np.int32)
            ends = np.r_[0, np.cumsum([len(x) for x in tokens])].astype(np.int32)
            current = []
            if name == "shared":
                known = read(HERE.parent / "pooler/shared-cases" / f"{key}.json")
                assert known["id"] == key and known["rows"] == len(ends) - 1
                assert known["manifest_sha256"] == plan["shared_manifest_sha256"]
                current = known["predictions"]["unchanged"]
                assert len(current) == len(labels)
            save_arrays(path, tokens=integer, ends=ends, rows=np.asarray(csv_rows, np.int32),
                        gold=np.asarray(labels, np.int8), incumbent=np.asarray(current, np.int8),
                        special=np.asarray([t.cls_token_id, t.sep_token_id, t.pad_token_id], np.int32),
                        fingerprint=plan["text_sha256"][key])
        print("Frozen", name, len(keys), plan["rows"][name], flush=True)
    write(HERE / "review.json", dict(model=plan["source_model"], protocol_sha256=sha(HERE / "protocol.json")))


if __name__ == "__main__":
    main()
