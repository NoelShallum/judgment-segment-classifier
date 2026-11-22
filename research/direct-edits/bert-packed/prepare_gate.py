"""Freeze case-disjoint routing cohorts and stage exact incumbent and alternative inputs."""

from functools import lru_cache
import importlib.util
from pathlib import Path
import sys
import time
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
from transformers import BertTokenizerFast

from prepare import HERE, ROOT, choose, fingerprint, get_cases
from native import read, save_arrays, sha, write
from hier_model import HierModel

GATE = HERE/"gate-input"
REVIEW = HERE.parent/"bert-t4"
CSV = ROOT/"references/official"
HIER = ROOT/"references/hier"


def freeze():
    GATE.mkdir(exist_ok=True)
    out = GATE/"gate-protocol.json"
    packed = read(HERE/"protocol.json")
    reviews = read(REVIEW/"protocol.json")
    sources = (CSV/"train.csv",CSV/"val.csv")
    assert [sha(path) for path in sources] == [packed["train_csv_sha256"],packed["val_csv_sha256"]]
    if out.exists():
        plan = read(out)
        assert plan["packed_manifest_sha256"] == sha(HERE/"protocol.json")
        assert plan["incumbent_route_manifest_sha256"] == sha(REVIEW/"protocol.json")
        t = get_cases("train",set(plan["groups"]["fit"]))
        v = get_cases("val",set(plan["groups"]["select"]+plan["groups"]["confirm"]))
        return plan,t,v
    t = get_cases("train",set(reviews["groups"]["fit"] + packed["groups"]["fit"]))
    v = get_cases("val")
    pq = packed["groups"]
    used = {fingerprint(t[k]) for k in pq["fit"]}
    used.update(fingerprint(v[k]) for k in pq["select"]+pq["shared"])
    used.update(fingerprint(rows) for rows in v.values())
    training = set(reviews["groups"]["fit"]) - set(pq["fit"])
    fit = choose(t,training,"case-disjoint-gate-train-v1",128,used)
    used = set(fingerprint(t[k]) for k in fit+pq["fit"])
    used.update(fingerprint(v[k]) for k in pq["select"]+pq["shared"])
    select = choose(v,set(reviews["groups"]["select"]) - set(pq["select"]+pq["shared"]),
                    "case-disjoint-gate-selection-v1",32,used)
    exclude = set(pq["select"] + pq["shared"])
    for key in ("select","confirm","shared"):
        exclude.update(reviews["groups"][key])
    for relative,field in (("hierarchy/scale/plan.json","validation"),("hierarchy/scale/quadratic/plan.json","cases"),
                           ("hierarchy/chain-program/plan.json","validation"),("hierarchy/case-head-program/plan.json","validation"),
                           ("hierarchy/uniform-retention-prospective.json","ids"),("boundary/protocol.json","validation")):
        f = HERE.parent/relative
        if f.exists():
            exclude.update(str(i) for i in read(f).get(field,[]))
    confirm = choose(v,set(v)-exclude,"case-disjoint-gate-confirmation-v1",64,used)
    groups = dict(fit=fit,select=select,confirm=confirm)
    assert set(fit+select+confirm).isdisjoint(pq["fit"]+pq["select"]+pq["shared"])
    combined = t | v
    plan = dict(
        groups=groups,rows={g:sum(len(combined[k]) for k in ids) for g,ids in groups.items()},
        label_support={g:np.bincount([row[2] for k in ids for row in combined[k]],minlength=7).tolist() for g,ids in groups.items()},
        text_fingerprints={k:fingerprint(combined[k]) for ids in groups.values() for k in ids},
        packed_manifest_sha256=sha(HERE/"protocol.json"),incumbent_route_manifest_sha256=sha(REVIEW/"protocol.json"),
        hierarchy_sha256=sha(HIER/"hier.safetensors"),train_csv_sha256=sha(sources[0]),val_csv_sha256=sha(sources[1]),
        fit_scope="128 natural original training cases from the independently fixed 384-case review study, excluding all text-matched packed/hybrid fits and all validation text. Only disagreement-label outcomes enter the gate fit. Both candidates were frozen on disjoint documents.",
        selectors="32 label-blind cached review selection cases. Choose between ordered packed classifier with its position/lexical coefficients and the frozen hierarchy/ordered BERT logits blend. Choose gates only here. Keep 64 new hashed validation judgments outside the listed recent experiments for confirmation.",
        model="Direct ridge regression on the three-state correctness difference between frozen candidate and exact deployed current system. Context and decision features are built solely from model outputs and real input structure. Only disagreements can switch.",
        coarse_control="Same rule and grid. Give it both output labels, actual 20%-review flag, candidate margin, and coarse current confidence. The current confidence is the explicit reviewer vote margin where reviewed and hierarchy source probability support difference elsewhere.",
        context_control="Same-sized feature write. Shuffle richer ordered state/neighbor features within exact current/candidate label pairs in the gated fit cohort. Keep label pair, confidence and targets intact. Select on unshuffled real selectors with the same fixed grid.",
        role_weights=[0.,.35],ridges=[.03,.3],gate_margins=[0.,.05,.1,.2,.35,.5],
        per_case_caps=[.025,.05,.1,.2],projection_seed=83611,projection_width=32,
        selection_guard="Choose maximum seven-role F1 subject to candidate accuracy within 0.003 of current. Baseline is included.",
        advance="Confirm lower paired 95% macro-F1 bound above zero versus the current incumbent, point accuracy at least as good, and higher point macro-F1 than matched coarse confidence and shuffled controls. Use no official test scores in this pipeline.",
        official_test_evaluated=False,gradient_steps=0)
    assert all(min(v)>0 for v in plan["label_support"].values())
    write(out,plan)
    return plan,t,v


def load_review_prepare():
    spec = importlib.util.spec_from_file_location("review_prepare_for_gate",REVIEW/"prepare.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def data(plan,train,val):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    token = BertTokenizerFast.from_pretrained(ROOT/"references/tokenizer",local_files_only=True)
    hierarchy = HierModel(HIER)
    review = load_review_prepare()
    review.sequence.phrase.load_predictor = lru_cache(maxsize=1)(review.sequence.phrase.load_predictor)
    parent = review.pair.predictor("direct")
    stage = GATE/"cases"
    stage.mkdir(exist_ok=True)
    checks = {}
    features = {}
    save_arrays(GATE/"hierarchy-author-head.npz",weight=hierarchy.weight,bias=hierarchy.bias,
                transitions=hierarchy.transitions)
    for group,ids in plan["groups"].items():
        start = time.perf_counter()
        pool = train if group=="fit" else val
        count = words = 0
        for j,key in enumerate(ids):
            row,text,y = zip(*pool[key])
            assert fingerprint(pool[key]) == plan["text_fingerprints"][key]
            dest = stage/(key+".npz")
            past = REVIEW/"cases"/(key+".npz")
            if dest.exists():
                with np.load(dest,allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["rows"],row)
                    np.testing.assert_array_equal(saved["gold"],y)
                    assert saved["fingerprint"].item()==plan["text_fingerprints"][key]
                    n = len(saved["tokens"])
            else:
                inputs = token(list(text),add_special_tokens=False,truncation=False,
                               return_attention_mask=False,return_token_type_ids=False,verbose=False)["input_ids"]
                arrays = [x or [token.unk_token_id] for x in inputs]
                tokens = np.concatenate(arrays).astype(np.int32)
                ends = np.r_[0,np.cumsum([len(x) for x in arrays])].astype(np.int32)
                n = len(tokens)
                old = HERE.parent/"hierarchy/scale/words"/(key+".npy")
                if group=="fit" and old.exists():
                    buffer = np.load(old,allow_pickle=False)
                    assert buffer.shape == (len(row),200)
                    np.testing.assert_allclose(hierarchy.words([text[0],text[-1]]).numpy(),buffer[[0,-1]],atol=4e-5,rtol=0.)
                    h = hierarchy.contextualize(torch.from_numpy(buffer))
                else:
                    h = hierarchy.contextualize(hierarchy.words(text))
                assert h.shape == (len(row),200)
                if group=="confirm":
                    route = review.route_capture(text,token,parent)
                else:
                    with np.load(past,allow_pickle=False) as old_record:
                        np.testing.assert_array_equal(old_record["original_row"],row)
                        np.testing.assert_array_equal(old_record["gold"],y)
                        assert old_record["case_sha256"].item()==plan["text_fingerprints"][key]
                    route = {}
                save_arrays(dest,rows=np.asarray(row,np.int32),gold=np.asarray(y,np.int8),
                            tokens=tokens,ends=ends,context=h,special=np.asarray([token.cls_token_id,token.sep_token_id,token.pad_token_id]),
                            fingerprint=plan["text_fingerprints"][key], **route)
            count += len(row)
            words += n
            checks[key] = dict(gate_case_sha256=sha(dest),cached_review_case_sha256=sha(past) if group!="confirm" else None)
            if j%16 == 0:
                print("gate inputs",group,j+1,len(ids),count,flush=True)
        assert count==plan["rows"][group]
        features[group] = dict(judgments=len(ids),rows=count,wordpieces=words,local_preparation_seconds=time.perf_counter()-start)
        print("ready",group,len(ids),count,words,round(time.perf_counter()-start,1),flush=True)
    write(GATE/"gate-input-receipts.json",dict(cases=checks,groups=features,
         manifest_sha256=sha(GATE/"gate-protocol.json"), hierarchy_head_sha256=sha(GATE/"hierarchy-author-head.npz")))


def package():
    out = HERE/"gate-input.zip"
    temp = out.with_suffix(".part")
    with ZipFile(temp,"w",compression=ZIP_DEFLATED,compresslevel=2) as z:
        for file in sorted(GATE.glob("*")):
            if file.is_file():
                z.write(file,file.name)
        for f in sorted((GATE/"cases").glob("*.npz")):
            z.write(f,"cases/"+f.name)
    temp.replace(out)
    print(out,out.stat().st_size,sha(out),flush=True)


if __name__=="__main__":
    data(*freeze())
    package()
