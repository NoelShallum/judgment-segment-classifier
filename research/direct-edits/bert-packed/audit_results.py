"""Audit exact shared-case alignment and measure complementary errors without tuning."""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from encode import HERE
from fit import confusion, metrics
from native import read, sha, write


def aligned(z,n,plan):
    data = np.load(BytesIO(z.read(f"shared{n}-predictions.npz")),allow_pickle=False)
    keys = data["keys"].tolist()
    assert keys == plan["groups"]["shared"][:n]
    y = []
    rows = []
    old = []
    ends = [0]
    for key in keys:
        with np.load(HERE/"cases"/(key+".npz"),allow_pickle=False) as f:
            y.extend(f["gold"])
            rows.extend(f["rows"])
            old.extend(f["incumbent"])
            ends.append(len(y))
    for tag,truth in (("csv_rows",rows),("gold",y),("current_20pct",old),("offsets",ends)):
        np.testing.assert_array_equal(data[tag],truth)
    receipt = json.loads(z.read(f"shared{n}.json"))
    report_bytes = z.read(f"shared{n}-predictions.npz")
    assert hashlib.sha256(report_bytes).hexdigest() == receipt["prediction_sha256"]
    assert receipt["rows"] == len(y) and receipt["judgments"] == n
    return data,receipt


def compare(data, receipt):
    y = data["gold"]
    base = data["current_20pct"]
    ends = data["offsets"]
    rows = {}
    for name, item in receipt["scores"].items():
        p = data[name]
        score = metrics(confusion(y,p))
        assert score["macro_f1"] == item["macro_f1"] and score["accuracy"] == item["accuracy"]
        helpful = (p == y) & (base != y)
        harmful = (p != y) & (base == y)
        case_delta = [int(helpful[a:b].sum()) - int(harmful[a:b].sum()) for a,b in zip(ends,ends[1:])]
        oracle = np.where(helpful,p,base)
        rows[name] = dict(
            macro_f1=score["macro_f1"], accuracy=score["accuracy"], fixed=int(helpful.sum()), broken=int(harmful.sum()),
            cases_with_any_unique_fixes=sum(bool(helpful[a:b].any()) for a,b in zip(ends,ends[1:])),
            cases_more_fixes_than_losses=sum(x>0 for x in case_delta),
            oracle_choose_only_true_fixes_macro_f1=metrics(confusion(y,oracle))["macro_f1"],
            fixes_by_human_class=np.bincount(y[helpful],minlength=7).tolist(),
            losses_by_human_class=np.bincount(y[harmful],minlength=7).tolist())
    return rows


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("archive",type=Path)
    args = cli.parse_args()
    plan = read(HERE/"protocol.json")
    result = {}
    with ZipFile(args.archive) as z:
        assert z.testzip() is None
        assert hashlib.sha256(z.read("protocol.json")).hexdigest() == sha(HERE/"protocol.json")
        device = json.loads(z.read("device.json"))
        assert len(device["devices"]) == 1 and "T4" in device["devices"][0]
        freeze = json.loads(z.read("freeze.json"))
        assert freeze["serialized_weights_roundtrip"], "Original packed replay has a serialization error."
        assert freeze["base_checkpoint_sha256"] == plan["source_model"]["sha256"]
        assert freeze["weights_sha256"] == hashlib.sha256(z.read("analytic-heads.safetensors")).hexdigest()
        assert freeze["parent_classifier"]["patch_sha256"] == hashlib.sha256(z.read("ordered-native-head.safetensors")).hexdigest()
        s16,r16 = aligned(z,16,plan)
        s48,r48 = aligned(z,48,plan)
        for key in r16["scores"]:
            np.testing.assert_array_equal(s16[key],s48[key][:len(s16[key])])
        result["shared16"] = compare(s16,r16)
        result["shared48"] = compare(s48,r48)
        for group in plan["groups"].values():
            for key in group:
                row = json.loads(z.read("feature-receipts/"+key+".json"))
                assert row["judgment"] == key and row["backend"] == device
                assert row["case_sha256"] == sha(HERE/"cases"/(key+".npz"))
                assert row["cost"]["ordered"] == row["cost"]["shuffled"]
                assert len({v["retained_wordpieces"] for v in row["cost"].values()}) == 1
    write(HERE/"result-audit.json",dict(
        archive_sha256=sha(args.archive), complete=True, device=device, by_cohort=result,
        caveat="Oracle uses gold labels to express the maximum gain from retaining observed complementary fixes. It is not a fitted hybrid. Shared outcomes have been used only for scoring and this audit."))
    print("verified",r16["rows"],r48["rows"],len(result["shared48"]),"matched arms",flush=True)


if __name__ == "__main__":
    main()
