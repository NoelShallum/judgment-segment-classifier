"""Preregister the joint whole-case head and stage checked native hierarchy states."""

from pathlib import Path
import time
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch

from prepare import HERE, ROOT, fingerprint, get_cases
from native import read, save_arrays, sha, write
from hier_model import HierModel

HIER = ROOT / "references/hier"
SCALE = HERE.parent / "hierarchy/scale"
WORK = HERE / "hybrid-input"


def lock():
    WORK.mkdir(exist_ok=True)
    path = WORK / "hybrid-protocol.json"
    schedule = HERE / "protocol.json"
    source = HIER / "hier.safetensors"
    if not path.exists():
        p = read(schedule)
        write(path, dict(
            packed_protocol_sha256=sha(schedule), native_hierarchy_sha256=sha(source),
            hierarchy_vocabulary_sha256=sha(HIER/"word2idx.json"),
            groups=p["groups"], rows=p["rows"], role_order=p["roles"], gradient_steps=0,
            design="Write one supervised ridge head to the concatenation of the native 200-dimensional full-judgment hierarchy state and 768-dimensional pooled BERT target state. Each joint head factors into two native classifier matrices and one common intercept.",
            comparator="Using the same hierarchy state, replace ordered BERT with identically sized and independently fitted shuffled BERT. The exact-hyperparameter companion refits with the ordered setting fixed. Include hierarchy-only, frozen packed ordered/shuffled, source hierarchy, frozen 20% incumbent, and a case-independent convex blend of separately fitted logits.",
            view_sources=["ordered","shuffled"], last_layer_fraction=[0.,.5,1.],
            inverse_frequency=[0.,.35], ridge=[.03,.30], transition_strength=[0.,.08,.18,.36],
            blend_weights_for_bert=[0.,.25,.5,.75,1.],
            selection="For each joint view and hierarchy-only head, choose among the fixed grid on 32 selector judgments by equal-weight macro-F1 then accuracy. For convex logits blend choose the two constant coefficients on the same selector. No learned gate. No shared outcomes used during selection.",
            primary="One prediction on every aligned shared sentence. Report shared16 and nested shared48. Compare ordered joint vs shuffled joint, exact-hyperparameter shuffled, hierarchy-only, packed ordered native and current 20%. Include whole-case bootstrap, exact role spans on supplied sentences and complementary corrections.",
            hierarchy="Published native CPU eval forward (zero hidden initialization). Same 200-dimensional sentence representation for all arms; no edited forget gates. SHA and row IDs checked at capture.",
            official_test_evaluated=False,
            transport="Only native 200-dimensional float32 states, ordering and integrity receipts. Reuse the FP32 packed BERT cache on the T4 host and solve with its CPU. No additional GPU inference."))
    plan = read(path)
    assert plan["packed_protocol_sha256"] == sha(schedule)
    assert plan["native_hierarchy_sha256"] == sha(source)
    assert plan["hierarchy_vocabulary_sha256"] == sha(HIER/"word2idx.json")
    return plan


def capture(plan):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model = HierModel(HIER)
    meta = read(SCALE/"plan.json")
    orig = read(HERE/"protocol.json")
    assert meta["source"] == orig["train_csv_sha256"] and meta["donor"] == plan["native_hierarchy_sha256"]
    groups = orig["groups"]
    train = get_cases("train",set(groups["fit"]))
    val = get_cases("val",set(groups["select"] + groups["shared"]))
    dest = WORK/"states"
    dest.mkdir(exist_ok=True)
    ledgers = {}
    save_arrays(WORK/"author-head.npz",weight=model.weight,bias=model.bias,transitions=model.transitions)
    for split,ids in groups.items():
        start = time.perf_counter()
        src = train if split == "fit" else val
        count = 0
        from_cache = 0
        for i,key in enumerate(ids):
            data = HERE/"cases"/(key+".npz")
            output = dest/(key+".npz")
            ledger = dest/(key+".json")
            signature = dict(document=key, source_case_sha256=sha(data),
                             donor_sha256=plan["native_hierarchy_sha256"], protocol_sha256=sha(WORK/"hybrid-protocol.json"))
            if output.exists() and ledger.exists():
                proof = read(ledger)
                if all(proof[k] == v for k,v in signature.items()) and proof["states_file_sha256"] == sha(output):
                    ledgers[key] = proof
                    count += proof["sentences"]
                    from_cache += int(proof["train_word_cache"])
                    continue
            rows,text,labels = zip(*src[key])
            assert fingerprint(src[key]) == orig["text_sha256"][key]
            with np.load(data,allow_pickle=False) as npz:
                np.testing.assert_array_equal(npz["rows"],rows)
                np.testing.assert_array_equal(npz["gold"],labels)
            old = SCALE/"words"/(key+".npy")
            reuse = split == "fit" and old.exists()
            if reuse:
                values = np.load(old,allow_pickle=False)
                assert values.shape == (len(text),200)
                chosen = [0,len(text)//2,len(text)-1]
                np.testing.assert_allclose(model.words([text[j] for j in chosen]).numpy(),values[chosen],atol=4e-5,rtol=0.)
                x = torch.from_numpy(values)
            else:
                x = model.words(text)
            result = model.contextualize(x)
            assert result.shape == (len(text),200) and np.isfinite(result).all()
            save_arrays(output,context=result,rows=np.asarray(rows,np.int32))
            proof = dict(**signature,states_file_sha256=sha(output),sentences=len(rows),train_word_cache=bool(reuse))
            write(ledger,proof)
            ledgers[key] = proof
            count += len(rows)
            from_cache += int(reuse)
            if i % 16 == 0:
                print("native",split,i+1,"of",len(ids),"sentences",count,flush=True)
        assert count == plan["rows"][split]
        print("native",split,len(ids),count,"used checked cached word states for",from_cache,
              "judgments, elapsed",round(time.perf_counter()-start,1),flush=True)
    write(WORK/"receipts.json",dict(cases=ledgers,author_head_sha256=sha(WORK/"author-head.npz")))


def package():
    archive = HERE/"hierarchy-features.zip"
    stage = archive.with_suffix(".part")
    with ZipFile(stage,"w",compression=ZIP_DEFLATED,compresslevel=2) as z:
        for name in ("hybrid-protocol.json","author-head.npz","receipts.json"):
            z.write(WORK/name,name)
        for f in sorted((WORK/"states").glob("*.npz")):
            z.write(f,"states/"+f.name)
    stage.replace(archive)
    print(archive,archive.stat().st_size,sha(archive),flush=True)


if __name__ == "__main__":
    capture(lock())
    package()
