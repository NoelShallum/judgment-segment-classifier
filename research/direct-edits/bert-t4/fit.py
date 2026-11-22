"""Solve an actual low-rank value-weight patch and score held-out judgments."""
import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file
import torch

from native import HEADS, WIDTH, HEAD_DIM, VALUE, NativeTail, batches, donor_at, move, read, require_t4, save_arrays, sha, write

ROLES = 7
RIDGES = (.03, .3, 3.)
GAINS = (.25, .60, 1.)
RANK = 6
KEY_DIMS = 96


@dataclass
class Bank:
    cases: list
    starts: np.ndarray
    review: np.ndarray
    gold: np.ndarray
    parent: np.ndarray
    original: np.ndarray
    prior: np.ndarray
    aux: torch.Tensor
    keys: torch.Tensor
    cls: torch.Tensor
    pooled: torch.Tensor
    existing: np.ndarray | None


def joined(directory, features, group, device, take=None):
    cfg = read(directory / "protocol.json")
    keys = cfg["groups"][group]
    keys = keys[:take] if take is not None else keys
    source, states, lens = [], [], []
    for key in keys:
        origin = directory / "cases" / (key + ".npz")
        path = features / (key + ".npz")
        receipt = read(features / (key + ".json"))
        if sha(origin) != receipt["case_sha256"] or sha(path) != receipt["sha256"]:
            raise ValueError(f"Invalid features for {key}.")
        if device == "cuda" and (not receipt["backend"]["cuda"] or "T4" not in receipt["backend"]["devices"][0]):
            raise ValueError("Expected the T4 FP32 feature shard.")
        with np.load(origin, allow_pickle=False) as f:
            source.append({name: f[name] for name in ("positions", "gold", "base", "original", "aux", "prior")})
            lens.append(len(f["gold"]))
            if "reference_incumbent" in f:
                source[-1]["reference_incumbent"] = f["reference_incumbent"]
        with np.load(path, allow_pickle=False) as f:
            states.append({name: f[name] for name in ("keys", "cls", "pooled")})
    starts = np.cumsum([0, *lens], dtype=np.int64)

    def concat(parts, key):
        return np.concatenate([p[key] for p in parts], axis=0)

    return Bank(
        keys, starts, np.concatenate([a+v["positions"] for a,v in zip(starts,source)]),
        concat(source, "gold"), concat(source, "base"), concat(source, "original"), concat(source, "prior"),
        move(concat(source, "aux"), device), move(concat(states, "keys"), device), move(concat(states, "cls"), device),
        move(concat(states, "pooled"), device),
        concat(source, "reference_incumbent") if group == "shared" else None,
    )


def prob(raw):
    positive = raw.clip(min=0)
    return positive / positive.sum(-1, keepdim=True).clip(min=1e-8)


class Readout:
    def __init__(self, directory, device):
        path = directory / "head.safetensors"
        plan = read(directory / "review.json")
        assert sha(path) == plan["existing_head_sha256"]
        self.params = {k: move(v, device) for k, v in load_file(path).items()}
        with np.load(directory / "donor-classifier.npz", allow_pickle=False) as library:
            self.weight = move(library["weight"], device)
            self.bias = move(library["bias"], device)
        self.rule = plan["rule"]

    def design(self, aux, pool):
        author = (pool @ self.weight.T + self.bias).softmax(-1)
        return torch.cat((aux, author, author * aux[:, 40:47], pool), dim=-1)

    def raw(self, aux, pool):
        w = self.params
        return ((self.design(aux, pool) - w["center"]) * w["scale"]) @ w["matrix"] + w["bias"]

    def review(self, bank, response):
        gain, gap, veto = self.rule
        response = response.detach().cpu().numpy()
        old = bank.parent[bank.review]
        anchor = .5 * bank.prior + .5 * np.eye(ROLES)[old]
        vote = (1.-gain) * anchor + gain * response
        new = vote.argmax(-1)
        allowed = vote[np.arange(len(old)), new] - vote[np.arange(len(old)), old] >= gap
        if veto:
            allowed &= new != bank.original
        guess = bank.parent.copy()
        guess[bank.review[allowed]] = new[allowed]
        return guess


def confusions(bank, guess):
    flat = bank.gold.astype(np.int64) * ROLES + guess
    return np.stack([np.bincount(flat[a:b], minlength=ROLES**2).reshape(ROLES, ROLES) for a,b in zip(bank.starts,bank.starts[1:])])


def measures(matrix):
    hit = np.diagonal(matrix, axis1=-2, axis2=-1)
    den = matrix.sum(-1) + matrix.sum(-2)
    by_role = 2 * hit / den.clip(min=1)
    return by_role.mean(-1), hit.sum(-1) / matrix.sum(axis=(-2,-1)), by_role


def summarize(bank, guess):
    summary = confusions(bank,guess).sum(0)
    f1, acc, roles = measures(summary)
    return {"macro_f1":float(f1), "accuracy":float(acc), "per_role_f1":roles.tolist(),
            "review_accuracy":float(np.mean(guess[bank.review] == bank.gold[bank.review])),
            "confusion": summary.tolist()}


def audit_baseline(bank, head):
    ref = head.review(bank, prob(head.raw(bank.aux, bank.pooled)))
    if bank.existing is not None:
        changes = np.count_nonzero(ref != bank.existing)
        if changes:
            raise ValueError(f"Native baseline disagrees with {changes} archived rows; inspect the route and precision.")
    return ref


def derivatives(tail, head, train):
    values = tail.native_values(train.keys)
    allparts = []
    for s in batches(len(values), 256):
        held = values[s].detach().requires_grad_(True)
        raw = head.raw(train.aux[s], tail.pooled(train.cls[s], held))
        grad = torch.stack([torch.autograd.grad(raw[:, c].sum(), held, retain_graph=(c<ROLES-1))[0]
                            for c in range(ROLES)], dim=1)
        allparts.append(grad - grad.mean(1, keepdim=True))
    return torch.cat(allparts)


def compressed_keys(train):
    # The same right factor is shared by all heads, giving a genuine rank-six native edit.
    flat = train.keys.flatten(0,1)
    center = flat.mean(0)
    gram = torch.zeros((WIDTH, WIDTH), device=flat.device, dtype=torch.float64)
    for batch in batches(len(flat), 8192):
        z = (flat[batch] - center).double()
        gram += z.T @ z
    gram /= len(flat)
    eigval, eigvec = torch.linalg.eigh(gram)
    s = eigval[-KEY_DIMS:].clamp(min=.025)
    basis = (eigvec[:, -KEY_DIMS:] / s.sqrt()).float()
    return center, basis


def class_weights(roles, rarity):
    balance = roles.bincount(minlength=ROLES).float().clamp(min=1).pow(-rarity)
    weights = balance[roles]
    return weights / weights.mean()


def equations(train, jacobian, residual, center, basis, route_weights):
    _, _, vectors = torch.linalg.svd(jacobian.mean(0), full_matrices=False)
    codes = vectors[:RANK].T.contiguous()
    per_head = codes.reshape(HEADS, HEAD_DIM, RANK)
    size = RANK * (KEY_DIMS + 1)
    gram = torch.zeros((size, size), device=jacobian.device, dtype=torch.float64)
    rhs = torch.zeros((size, 2), device=jacobian.device, dtype=torch.float64)
    # Randomize target residual across reviewed training rows while preserving its distribution.
    permutation = move(np.random.default_rng(302609).permutation(len(residual)), jacobian.device)
    both = torch.stack((residual, residual[permutation]), -1)
    for s in batches(len(residual), 256):
        key = (train.keys[s] - center) @ basis
        key = torch.cat((key, torch.ones_like(key[..., :1])), -1)
        g = jacobian[s].reshape(-1, ROLES, HEADS, HEAD_DIM)
        projected = torch.einsum("bchv,hvr->bchr", g, per_head)
        design = torch.einsum("bchr,bhf->bcrf", projected, key).reshape(-1, size).double()
        weights = route_weights[s].repeat_interleave(ROLES).double()
        gram += design.T @ (weights[:, None] * design)
        rhs += design.T @ (weights[:, None] * both[s].reshape(-1, 2).double())
    return codes, gram, rhs


def limited_delta(solution, codes, center, transform, value_layer):
    rows = solution.reshape(RANK, KEY_DIMS+1).float()
    delta_matrix = codes @ rows[:, :-1] @ transform.T
    delta_bias = codes @ rows[:, -1] - delta_matrix @ center
    relative = torch.linalg.norm(delta_matrix) / torch.linalg.norm(value_layer.weight)
    gain = min(1., .08 / max(relative.item(), 1e-12))
    return delta_matrix * gain, delta_bias * gain


def patch_stats(old, new):
    err = (new - old).double().cpu().numpy()
    return {"shape":list(new.shape), "changed_entries":int(np.count_nonzero(err)),
            "total_entries":new.numel(), "frobenius_delta":float(np.linalg.norm(err)),
            "relative_frobenius_delta":float(np.linalg.norm(err)/max(np.linalg.norm(old.cpu().numpy()),1e-12)),
            "source_tensor_sha256":hashlib.sha256(old.cpu().numpy().tobytes()).hexdigest(),
            "edited_tensor_sha256":hashlib.sha256(new.cpu().numpy().tobytes()).hexdigest()}


def replay(tail, head, bank, weight, bias):
    pool = tail.pooled(bank.cls, tail.native_values(bank.keys, weight, bias))
    return head.review(bank, prob(head.raw(bank.aux, pool)))


def head_design(head, bank, transform, center, kind):
    raw = head.design(bank.aux,bank.pooled)
    if kind == "pooled":
        return raw
    keys = ((bank.keys-center) @ transform[:, -12:]).flatten(1)
    return torch.cat((raw, keys), dim=-1)


def fit_heads(head, train, dev, transform, center):
    labels = move(train.gold[train.review], train.keys.device, torch.long)
    targets = torch.nn.functional.one_hot(labels, ROLES).double()
    finalized = {}
    for kind in ("pooled", "attention-aware"):
        x = head_design(head, train, transform, center, kind).double()
        observed = head_design(head, dev, transform, center, kind).double()
        mean, inv = x.mean(0), x.std(0, unbiased=False).clamp(min=.025).reciprocal()
        x = torch.cat(((x-mean)*inv, torch.ones_like(x[:, :1])), 1)
        observed = torch.cat(((observed-mean)*inv, torch.ones_like(observed[:, :1])), 1)
        leaderboard = []
        best = (-1., -1.)
        for rarity in (.0, .35, .65):
            wx = x * class_weights(labels, rarity)[:, None]
            g = x.T @ wx
            rhs = wx.T @ targets
            diag = torch.full_like(g.diag(), g.diag().mean().item())
            diag[-1] *= .02
            for ridge in (.03, .16, .70):
                fit = torch.linalg.solve(g + (ridge*diag).diag(), rhs)
                pred = head.review(dev, prob(observed @ fit))
                score = summarize(dev,pred)
                key = (score["macro_f1"],score["accuracy"])
                leaderboard.append((rarity,ridge,*key))
                if key > best:
                    best = key
                    finalized[kind] = (mean,inv,fit,rarity,ridge)
        finalized[kind] += (leaderboard,)
    return finalized


def head_predict(head, bank, kind, params):
    x = head_design(head, bank, params["basis"], params["key_center"], kind)
    init = (x - params[kind+".center"])*params[kind+".scale"]
    z = torch.cat((init, torch.ones_like(init[:, :1])), 1)
    return head.review(bank, prob(z @ params[kind+".weights"]))


def as_numpy(t):
    return t.detach().cpu().contiguous().numpy()


def fit(directory, checkpoint, features, output, device, nfit):
    train = joined(directory, features, "fit", device, take=nfit)
    dev = joined(directory, features, "select", device)
    model = donor_at(directory, checkpoint, device)
    tail = NativeTail(model)
    head = Readout(directory, device)
    assert set(train.cases).isdisjoint(dev.cases)
    labels = move(train.gold[train.review], device, torch.long)
    target = torch.nn.functional.one_hot(labels, ROLES).double()
    old = audit_baseline(train, head)
    saved_dev = audit_baseline(dev, head)
    dev_baseline = summarize(dev, saved_dev)
    residual = target - prob(head.raw(train.aux, train.pooled))
    importance = class_weights(labels,.42)
    protect = move(np.where(old[train.review] == train.gold[train.review], 1.45, 1.0), device)
    attn, center, right = derivatives(tail,head,train), *compressed_keys(train)
    left, gram, rhs = equations(train,attn,residual,center,right,importance*protect)
    value = tail.attention.value
    candidates, picked = [], None
    target_stats = (-1., -1.)
    for damp in RIDGES:
        coeff = torch.linalg.solve(gram + (damp*gram.diag().mean())*torch.eye(len(gram),device=device,dtype=gram.dtype), rhs)
        delta = [limited_delta(coeff[:,i],left,center,right,value) for i in range(2)]
        good, bad = delta
        correct_norm = torch.linalg.norm(torch.cat(tuple(a.flatten() for a in good)))
        shuffled_norm = torch.linalg.norm(torch.cat(tuple(a.flatten() for a in bad)))
        control = tuple(a*correct_norm/shuffled_norm.clamp(min=1e-12) for a in bad)
        for gain in GAINS:
            edits = tuple(value.weight + arm[0]*gain for arm in (good,control))
            biases = tuple(value.bias + arm[1]*gain for arm in (good,control))
            pred = replay(tail,head,dev,edits[0],biases[0])
            met = summarize(dev,pred)
            k = (met["macro_f1"], met["accuracy"])
            candidates.append([damp,gain,*k])
            if k > target_stats:
                target_stats,picked = k,(damp,gain,edits,biases,met)
    selected_heads = fit_heads(head,train,dev,right,center)
    output.mkdir(parents=True,exist_ok=True)
    for tag, weight, bias in zip(("targeted", "shuffled"),picked[2],picked[3]):
        save_file({VALUE+".weight":as_numpy(weight),VALUE+".bias":as_numpy(bias)},str(output/(tag+".safetensors")))
    auxiliary = {"basis":right,"key_center":center}
    for name,(mean,inv,coeff,*_) in selected_heads.items():
        auxiliary.update({name+".center":mean,name+".scale":inv,name+".weights":coeff})
    save_file({k:as_numpy(v) for k,v in auxiliary.items()},str(output/"refitted-heads.safetensors"))
    edited = {tag:[patch_stats(value.weight,w),patch_stats(value.bias,b)] for tag,w,b in zip(("targeted","shuffled"),picked[2],picked[3])}
    positive = picked[-1]["macro_f1"] > dev_baseline["macro_f1"] and picked[-1]["accuracy"] >= dev_baseline["accuracy"]-.002
    report = {"protocol_sha256":sha(directory/"protocol.json"), "source_sha256":sha(checkpoint),
              "name":"Final attention value closed-form low-rank edit", "tensor_names":[VALUE+".weight",VALUE+".bias"],
              "gradient_update_steps":0, "fixed_jacobian_used":True, "programmed_rank":RANK, "compressed_key_dimensions":KEY_DIMS,
              "fit_cases":train.cases,"fit_rows":len(train.gold),"fit_referrals":len(train.review),
              "select_cases":dev.cases,"select_rows":len(dev.gold),"select_referrals":len(dev.review),
              "select_incumbent":dev_baseline,"select_prefers_targeted":positive,"positive_runner_up_parameters":picked[:2],
              "select_targeted":picked[-1],"edit_grid":candidates,"edit_tensor_stats":edited,
              "readout_selection":{name: v[3:] for name,v in selected_heads.items()},
              "sha256":{p.name:sha(p) for p in output.glob("*.safetensors")}}
    write(output/"freeze.json",report)
    print("frozen",nfit,"training cases,",len(train.review),"windows; select F1",
          dev_baseline["macro_f1"],"targeted",picked[-1]["macro_f1"],"identity gate",not positive,flush=True)


def interval(donor, candidate):
    rng = np.random.default_rng(938200)
    sample = rng.integers(len(donor),size=(2000,len(donor)))
    before = measures(donor[sample].sum(1))[0]
    after = measures(candidate[sample].sum(1))[0]
    return np.quantile(after-before,[.025,.975]).tolist()


def evaluate(directory, checkpoint, features, output, device, group, take):
    b = joined(directory,features,group,device,take)
    spec = read(output/"freeze.json")
    assert set(b.cases).isdisjoint(set(spec["fit_cases"]) | set(spec["select_cases"]))
    assert sha(directory/"protocol.json") == spec["protocol_sha256"]
    for name,signed in spec["sha256"].items():
        assert sha(output/name) == signed
    model = donor_at(directory,checkpoint,device)
    tail, head = NativeTail(model), Readout(directory,device)
    pred = {"current_actual":audit_baseline(b,head), "parent_without_bert":b.parent}
    physical = {}
    for tag in ("targeted","shuffled"):
        patch = load_file(output/(tag+".safetensors"))
        weight,bias = (move(patch[VALUE+"."+k],device) for k in ("weight","bias"))
        pred[tag] = replay(tail,head,b,weight,bias)
        if tag == "targeted":
            itself = model.encoder.layer[-1].attention.self.value
            with torch.no_grad():
                itself.weight.copy_(weight)
                itself.bias.copy_(bias)
            with np.load(directory/"cases"/(b.cases[0]+".npz"),allow_pickle=False) as f:
                ids, mask = f["ids"][:24],f["mask"][:24]
            with torch.inference_mode():
                count = len(ids)
                native = model(move(ids,device,torch.long),attention_mask=move(mask,device,torch.bool)).pooler_output
                computed=tail.pooled(b.cls[:count],tail.native_values(b.keys[:count],weight,bias))
                err=(native-computed).abs().max().item()
                weighted=prob(head.raw(b.aux[:count],native))
                cached=prob(head.raw(b.aux[:count],computed))
                disagree=int((weighted.argmax(1)!=cached.argmax(1)).sum())
                if err>5e-5 or disagree:
                    raise RuntimeError(f"The in-model edit failed replay parity: {err}, {disagree}.")
                physical={"device":device,"case":b.cases[0],"rows":count,"max_pooled_delta":err,"review_argmax_disagreements":disagree}
    aux = {k:move(v,device) for k,v in load_file(output/"refitted-heads.safetensors").items()}
    for name in ("pooled", "attention-aware"):
        pred[name+" refit"] = head_predict(head,b,name,aux)
    curves = {k:confusions(b,p) for k,p in pred.items()}
    baseline = pred["current_actual"]
    report = {"group":group,"judgments":b.cases,"rows":len(b.gold),"referrals":len(b.review),
              "fit_freeze_sha256":sha(output/"freeze.json"),"native_replay":physical,
              "score":{k:summarize(b,v) for k,v in pred.items()},
              "cohort_overlap_with_fit_or_selection":0,
              "paired_targeted_vs":{k:interval(v,curves["targeted"]) for k,v in curves.items() if k!="targeted"},
              "changed_by_target":int(np.count_nonzero(pred["targeted"]!=baseline)),
              "corrected_by_target":int(np.count_nonzero((pred["targeted"]==b.gold)&(baseline!=b.gold))),
              "harmed_by_target":int(np.count_nonzero((pred["targeted"]!=b.gold)&(baseline==b.gold)))}
    save_arrays(output/(group+"-predictions.npz"),case_ids=b.cases,offsets=b.starts,gold=b.gold,**pred)
    write(output/(group+"-report.json"),report)
    print(group,len(b.cases), "all rows",len(b.gold), {k:round(v["macro_f1"],5) for k,v in report["score"].items()},flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("fit","confirm","shared"))
    p.add_argument("directory",type=Path)
    p.add_argument("checkpoint",type=Path)
    p.add_argument("features",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--take",type=int)
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    a=p.parse_args()
    torch.set_num_threads(1)
    if a.device == "cuda":
        require_t4()
    if a.mode=="fit":
        fit(a.directory,a.checkpoint,a.features,a.output,a.device,a.take)
    else:
        evaluate(a.directory,a.checkpoint,a.features,a.output,a.device,a.mode,a.take)
