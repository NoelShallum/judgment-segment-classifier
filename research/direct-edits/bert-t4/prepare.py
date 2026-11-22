"""Freeze case groups and extract the deployed review route for the value-weight study."""
import argparse
from collections import defaultdict
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from transformers import BertTokenizerFast

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import mixedbread_referral_overlay as overlay
import predict_pairwise_judgment as pair
from program_trial import TO_ID
import sentence_bert_referral_write as bert
import sentence_phrase_sequence as sequence
import sentence_referral_utility_write as utility
from weight_edit import LABELS, file_hash, tensor

RULE_REVISION = "attoflow-inlegalbert-last-value-20261122-v1"
ROUTE_LIMIT = .20
CSV = ROOT / "references/official"
RESULTS = ROOT / "results"
CASE_DIR = HERE / "cases"
DONOR_PATH = "InLegalBERT/InLegalBERT(i)model.safetensors"
PRIVATE_CONFIG = HERE / "review.json"


def write_json(path, value):
    interim = path.with_suffix(".tmp")
    interim.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    interim.replace(path)


def read_json(path):
    return json.loads(path.read_text())


def cases(split, selected=None):
    parts = defaultdict(list)
    with (CSV / (split + ".csv")).open(encoding="utf-8-sig", newline="") as f:
        for i, item in enumerate(csv.DictReader(f)):
            doc = item["Index"]
            if (selected is None or doc in selected) and item["Text"].strip():
                parts[doc].append((i, item["Text"], TO_ID[item["Label"]]))
    return parts


def fingerprint(rows):
    h = hashlib.sha256()
    for _, text, _ in rows:
        h.update(" ".join(text.casefold().split()).encode())
        h.update(b"\0")
    return h.hexdigest()


def order(name, candidates):
    return sorted(candidates, key=lambda doc: hashlib.sha256(f"{RULE_REVISION}|{name}|{doc}".encode()).digest())


def make_protocol():
    path = HERE / "protocol.json"
    if path.exists():
        cfg = read_json(path)
        for split, digest in cfg["csv_sha256"].items():
            assert digest == file_hash(CSV / (split + ".csv"))
        return cfg
    val = cases("val")
    training = cases("train")
    seen = set()
    for p in (RESULTS / "sentence-bert-referral-frozen.json", RESULTS / "mixedbread-referral-overlay-frozen.json",
              RESULTS / "sentence-document-graph-frozen.json", RESULTS / "sentence-graph-holdout-frozen.json",
              HERE.parent / "hierarchy/scale/plan.json", HERE.parent / "hierarchy/scale/quadratic/plan.json"):
        v = read_json(p)
        for key in ("cases", "fit_cases", "selection_cases", "train_cases", "validation"):
            if key in v and isinstance(v[key], list):
                seen.update(str(x) for x in v[key])
    previous = set(read_json(HERE.parent / "shared-confirmation.json")["extended_validation_judgment_ids"])
    heldout = set(val) - seen - previous

    def choose(pool, key, size, forbidden):
        result = []
        for doc in order(key, pool):
            rows = pool[doc]
            digest = fingerprint(rows)
            if digest in forbidden:
                continue
            forbidden.add(digest)
            result.append(doc)
            if len(result) == size:
                return result
        raise ValueError(f"Only {len(result)} distinct judgments found for {key}")

    # A uniform judgment sample retains the natural seven-way distribution and case lengths.
    val_fingerprints = {fingerprint(r) for r in val.values()}
    train_fingerprints = set(val_fingerprints)
    train = choose(training, "train", 384, train_fingerprints)
    remaining = {k: val[k] for k in heldout}
    heldout_fingerprints = {fingerprint(val[p]) for p in previous} | {fingerprint(training[p]) for p in train}
    selection = choose(remaining, "select", 48, heldout_fingerprints)
    confirmation = choose({k: val[k] for k in heldout if k not in selection}, "confirm", 72, heldout_fingerprints)
    shared = read_json(HERE.parent / "shared-confirmation.json")["extended_validation_judgment_ids"]
    groups = dict(fit=train, select=selection, confirm=confirmation, shared=shared)
    assert sum(map(len, groups.values())) == len(set().union(*map(set, groups.values())))
    allcases = training | val
    receipt = dict(version=RULE_REVISION, role_order=list(LABELS), groups=groups,
                   prefix_for_quick_fit=96,
                   rows={name: sum(len(allcases[k]) for k in ids) for name, ids in groups.items()},
                   case_text_sha256={k: fingerprint(allcases[k]) for ids in groups.values() for k in ids},
                   csv_sha256={s: file_hash(CSV / (s + ".csv")) for s in ("train", "val")},
                   selection="Set case IDs before computing any new edit outcome. Exact case text duplicates excluded across selected sets and fit versus full validation. Prior reviews may have used some validation cases.",
                   target_tensor="bert.encoder.layer.11.attention.self.value.weight",
                   target_bias="bert.encoder.layer.11.attention.self.value.bias", ground_truth="Public seven-role annotations only",
                   route="Original 20% confidence positions, previous two and target, up to 96 tokens.",
                   method="A ridge solve from observed attention-weighted inputs and a fixed native layer Jacobian; native edit, fixed-head, matched-norm shuffled, and independently refitted head comparators.",
                   primary_comparison="Whole-judgment macro-F1 on 72 frozen confirmation judgments versus the actual deployed reviewer. Judge score against all unchanged, independently refitted and shuffled controls; selection labels never used in fitting. Shared 48 is an additional cross-model comparison.")
    write_json(path, receipt)
    return receipt


def setup(protocol):
    author_head = HERE / "donor-classifier.npz"
    np.savez(author_head, weight=tensor(bert.ORIGINAL, "classifier.weight"), bias=tensor(bert.ORIGINAL, "classifier.bias"))
    plan = bert.read(bert.TAG + "-frozen")
    gate = bert.read(overlay.TAG + "-frozen")
    recipe_path = bert.OUT / (bert.TAG + "-2.safetensors")
    assert plan["checkpoints"][overlay.BERT] == file_hash(recipe_path)
    assert plan["backbone_weights_sha256"] == file_hash(bert.ORIGINAL)
    mode = plan["frozen_modes"][overlay.BERT][0]
    assert mode == "hidden"
    config = {
        "model": {"repo": "L-NLProc/LegalSeg_InLegalBERT", "revision": "44d7925e5ab422ecf04acd5dcbb0b62a6f8d8f53",
                  "path": DONOR_PATH, "sha256": plan["backbone_weights_sha256"]},
        "existing_head_sha256": file_hash(recipe_path), "rule": gate["arms"]["BERT only"]["choice"], "mode": mode,
        "max_tokens": bert.LIMIT, "rate": ROUTE_LIMIT, "protocol_sha256": file_hash(HERE / "protocol.json")}
    write_json(PRIVATE_CONFIG, config)


def route_capture(text, tokenizer, parent):
    result = {}

    def capture(bank, lex, unary, initial, item, hidden, model, cfg, vocab):
        base = parent(bank, lex, unary, initial)
        confidence = np.exp(unary - unary.max(1, keepdims=True))
        confidence /= confidence.sum(1, keepdims=True)
        positions = utility.route(bank[1], 1.-confidence[np.arange(len(base)), base], ROUTE_LIMIT)
        row = bert.case_view(item, hidden, model, cfg, vocab, positions)
        np.testing.assert_array_equal(row[2], bank[3])
        encoded = bert.pack_ids(tokenizer, [row], bert.MODES[1])
        matrix = np.full((len(encoded), bert.LIMIT), tokenizer.pad_token_id, dtype=np.int32)
        mask = np.zeros(matrix.shape, dtype=bool)
        for i, token_ids in enumerate(encoded):
            matrix[i, :len(token_ids)] = token_ids
            mask[i, :len(token_ids)] = True
        result.update(ids=matrix, mask=mask, positions=positions, base=base, original=row[2][positions], aux=row[4], prior=row[5])
        return base

    sequence.run_input(list(text), False, "spans", revise_with_context=capture)
    return result


def collect(protocol, groups):
    sequence.phrase.load_predictor = lru_cache(maxsize=1)(sequence.phrase.load_predictor)
    parent = pair.predictor("direct")
    tokenizer = BertTokenizerFast.from_pretrained(ROOT / "references/tokenizer", local_files_only=True)
    shared_dir = HERE.parent / "pooler/shared-cases"
    sources = {split: cases(split, set(ids)) for split, ids in (
        ("train", protocol["groups"]["fit"]), ("val", sum([protocol["groups"][g] for g in ("select", "confirm", "shared")], [])))}
    CASE_DIR.mkdir(exist_ok=True)
    for group, subset in groups:
        keys = protocol["groups"][group]
        if subset:
            keys = keys[subset[0]:subset[1]]
        for index, key in enumerate(keys, 1):
            existing = CASE_DIR / (key + ".npz")
            if existing.exists():
                with np.load(existing, allow_pickle=False) as saved:
                    assert saved["case_sha256"].item() == protocol["case_text_sha256"][key]
                continue
            records = sources["train" if group == "fit" else "val"][key]
            src_idx, text, gold = zip(*records)
            result = route_capture(text, tokenizer, parent)
            assert len(result["base"]) == len(gold)
            ref = shared_dir / (key + ".json")
            if group == "shared":
                historical = read_json(ref)
                np.testing.assert_array_equal(result["positions"], historical["predictions"]["positions"])
                result["reference_incumbent"] = np.asarray(historical["predictions"]["unchanged"], dtype=np.int8)
            temporary = existing.with_suffix(".tmp")
            with temporary.open("wb") as f:
                np.savez_compressed(f, **result, gold=np.asarray(gold, np.int8), original_row=np.asarray(src_idx, np.int64),
                                    case_sha256=protocol["case_text_sha256"][key])
            temporary.replace(existing)
            if index % 12 == 0 or index == len(keys) or index == 1:
                print(group, index, len(keys), "judgment", key, "rows", len(gold), "reviews", len(result["ids"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "priority", "all"), default="all")
    args = parser.parse_args()
    protocol = make_protocol()
    setup(protocol)
    print("fixed case counts", {k: len(v) for k, v in protocol["groups"].items()}, protocol["rows"], flush=True)
    if args.phase == "smoke":
        groups = [("shared", (0, 2)), ("fit", (0, 2))]
    elif args.phase == "priority":
        groups = [("shared", (0, 16)), ("fit", (0, 96)), ("select", None)]
    else:
        groups = [("shared", None), ("select", None), ("confirm", None), ("fit", None)]
    collect(protocol, groups)


if __name__ == "__main__":
    main()
