"""Collect ordered target states and replay the exact incumbent with cached reviewer pools."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import time

import numpy as np
import torch
from scipy.special import softmax

from encode import TopTwo, case_windows
from native import donor_at, read, require_t4, save_arrays, sha, write


def load_reviewer(source):
    spec = importlib.util.spec_from_file_location("frozen_value_reviewer_for_gate",source/"fit.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def vote(source, vectors, reviewer, tool):
    positions = source["positions"]
    bank = SimpleNamespace(parent=source["base"],review=positions,prior=source["prior"],original=source["original"])
    pool = torch.as_tensor(vectors,dtype=torch.float32)
    aux = torch.as_tensor(source["aux"],dtype=torch.float32)
    with torch.inference_mode():
        post = tool.prob(reviewer.raw(aux,pool))
        actual = reviewer.review(bank,post)
    gain,_,_ = reviewer.rule
    anchor = .5*bank.prior + .5*np.eye(7)[bank.parent[positions]]
    signal = (1.-gain)*anchor + gain*post.numpy()
    desired = signal[np.arange(len(positions)),actual[positions]].copy()
    signal[np.arange(len(positions)),actual[positions]] = -np.inf
    margin = desired - signal.max(1)
    return actual.astype(np.int8),margin.astype(np.float32)


def verify_reference(source,features,reviewer,tool):
    cfg = read(source/"protocol.json")
    done = total = 0
    for key in cfg["groups"]["shared"]:
        original = source/"cases"/(key+".npz")
        feat = features/(key+".npz")
        if not feat.exists():
            continue
        meta = read(features/(key+".json"))
        assert meta["case_sha256"] == sha(original) and meta["sha256"] == sha(feat)
        with np.load(original,allow_pickle=False) as x,np.load(feat,allow_pickle=False) as y:
            result, _ = vote(x,y["pooled"],reviewer,tool)
            np.testing.assert_array_equal(result,x["reference_incumbent"])
            done += 1
            total += len(result)
    assert done >= 16
    return dict(judgments=done,rows=total,disagreements=0)


@torch.inference_mode()
def review_pool(net,ids,mask,batch):
    windows = []
    for a in range(0,len(ids),batch):
        b = a+batch
        result = net(torch.as_tensor(ids[a:b],device="cuda",dtype=torch.long),
                     attention_mask=torch.as_tensor(mask[a:b],device="cuda"))
        windows.append(result.pooler_output.cpu().numpy())
    return np.concatenate(windows)


def collect(group,paths,batch=24):
    input_,work,packed_data,review_state,checkpoint = paths
    plan = read(input_/"gate-protocol.json")
    receipt = read(input_/"gate-input-receipts.json")
    assert receipt["manifest_sha256"] == sha(input_/"gate-protocol.json")
    assert sha(input_/"hierarchy-author-head.npz") == receipt["hierarchy_head_sha256"]
    with np.load(input_/"hierarchy-author-head.npz",allow_pickle=False) as orig:
        head = orig["weight"][3:].copy(),orig["bias"][3:].copy()
    assert sha(packed_data/"protocol.json") == plan["packed_manifest_sha256"]
    reviewer_dir = review_state/"source-full"
    refit_cache = review_state/"features"
    assert sha(reviewer_dir/"protocol.json")==plan["incumbent_route_manifest_sha256"]
    legacy = load_reviewer(reviewer_dir)
    reviewer = legacy.Readout(reviewer_dir,"cpu")
    out = work/"features"
    out.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(2)
    validation_source = verify_reference(reviewer_dir,refit_cache,reviewer,legacy)
    hardware = require_t4()
    write(work/"incumbent-parity.json",validation_source)
    encoder = TopTwo(donor_at(packed_data,checkpoint,"cuda"),None)
    n = 0
    started = time.perf_counter()
    for key in plan["groups"][group]:
        data = input_/"cases"/(key+".npz")
        expected = receipt["cases"][key]
        assert sha(data)==expected["gate_case_sha256"]
        path = out/(key+".npz")
        report = out/(key+".json")
        if path.exists() and report.exists():
            item = read(report)
            if item["case_sha256"] == sha(data) and item["features_sha256"] == sha(path) and item["backend"] == hardware:
                n += item["rows"]
                continue
        with np.load(data,allow_pickle=False) as d:
            ids,ends,special = d["tokens"],d["ends"],d["special"]
            h = d["context"]
            w = case_windows(ids,ends,special,key,"ordered")
            hidden, retained, _, billed = encoder.run(w,batch,"cuda",special[2],len(h))
            np.testing.assert_array_equal(retained,np.diff(ends))
            if group=="confirm":
                pool = review_pool(encoder.model,d["ids"],d["mask"],batch)
                current,ref_margin = vote(d,pool,reviewer,legacy)
                reviewed = d["positions"].copy()
            else:
                src = reviewer_dir/"cases"/(key+".npz")
                pool_path = refit_cache/(key+".npz")
                auth = read(refit_cache/(key+".json"))
                assert sha(src)==auth["case_sha256"]==expected["cached_review_case_sha256"]
                assert sha(pool_path)==auth["sha256"]
                with np.load(src,allow_pickle=False) as r,np.load(pool_path,allow_pickle=False) as f:
                    np.testing.assert_array_equal(r["gold"],d["gold"])
                    np.testing.assert_array_equal(r["original_row"],d["rows"])
                    current,ref_margin = vote(r,f["pooled"],reviewer,legacy)
                    reviewed = r["positions"].copy()
            assert len(current)==len(h)
            probability = softmax(h@head[0].T+head[1],axis=1)
            support = probability[np.arange(len(h)),current].copy()
            probability[np.arange(len(h)),current] = -np.inf
            margin = (support-probability.max(1)).astype(np.float32)
            margin[reviewed] = ref_margin
        seen = np.zeros(len(h),bool)
        seen[reviewed] = True
        save_arrays(path,ordered=hidden,current=current,current_margin=margin,reviewed=seen)
        write(report,dict(backend=hardware,case_sha256=sha(data),features_sha256=sha(path),rows=len(h),
                          reviewed=int(seen.sum()),ordered_windows=len(w),billed_positions=billed,
                          wordpieces=len(ids),source="Native current 20% reviewer; FP32 frozen InLegalBERT; original case order."))
        n += len(h)
        print("gate ordered",group,key,len(h),n,flush=True)
    assert n == plan["rows"][group]
    encoder.hook.remove()
    print("gate ordered complete",group,n,round(time.perf_counter()-started,1),flush=True)
