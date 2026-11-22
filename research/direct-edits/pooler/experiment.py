"""Direct rank-six edits to the published InLegalBERT tanh pooler, with frozen readouts."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file
from scipy.special import softmax

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from weight_edit import ORIGINAL, LABELS, file_hash, patched_checkpoint, tensor

WEIGHT = "bert.pooler.dense.weight"
BIAS = "bert.pooler.dense.bias"
ARM = "BERT previous two and target"
FOLDER = ROOT / "cache/sentence-bert-referral"
SPEC = json.loads((ROOT / "results/sentence-bert-referral-frozen.json").read_text())


def write(name, value):
    target = HERE / name
    temp = target.with_suffix(".partial")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(target)


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def view(split):
    rows = json.loads((FOLDER / (split + "-views-v1.json")).read_text())
    keys, gold, parent, pos, aux, prior, texts = zip(*rows)
    pos = [np.asarray(v, dtype=int) for v in pos]
    target = np.concatenate([np.asarray(y)[q] for y, q in zip(gold, pos)])
    lengths = np.asarray([len(q) for q in pos])
    off = np.cumsum([0] + [len(y) for y in gold])
    loc = np.concatenate([q + b for q, b in zip(pos,off)])
    receipt = json.loads((FOLDER / (split + "-1.json")).read_text())
    path = FOLDER / receipt["file"]
    assert file_hash(path) == receipt["features_sha256"] and receipt["weights_sha256"] == SPEC["backbone_weights_sha256"]
    vec = np.asarray(np.load(path), dtype=np.float64)
    assert vec.shape == (len(loc), 768)
    return dict(keys=keys, target=target, aux=np.concatenate([np.asarray(x).reshape(-1,68) for x in aux]), vec=vec, base=np.concatenate(parent),
                gold=np.concatenate(gold), pos=loc, prior=np.concatenate([np.asarray(x).reshape(-1,7) for x in prior]), off=off, lengths=lengths,
                view_sha=file_hash(FOLDER/(split+"-views-v1.json")), feature_sha=receipt["features_sha256"])


def params():
    from sentence_document_graph_write import OUT
    p = OUT / "sentence-bert-referral-2.safetensors"
    assert file_hash(p) == SPEC["checkpoints"][ARM] and SPEC["frozen_modes"][ARM][0] == "hidden"
    return load_file(str(p))


CW = tensor(ORIGINAL, "classifier.weight").astype(float)
CB = tensor(ORIGINAL, "classifier.bias").astype(float)


def readout(data, vec, arm, head):
    logit = vec @ CW.T + CB
    if arm == "public":
        return logit
    p = softmax(logit, axis=1)
    x = np.column_stack([data["aux"], p, p * data["aux"][:,40:47], vec])
    return ((x - head["center"]) * head["scale"]) @ head["matrix"] + head["bias"]


def choose_labels(data, scores, arm):
    if arm == "public":
        return scores.argmax(1)
    scores = np.maximum(scores, 0)
    scores /= np.maximum(scores.sum(1,keepdims=True), 1.e-8)
    gain, margin = SPEC["frozen_modes"][ARM][-2:]
    base = data["base"][data["pos"]]
    prob = (1-gain) * (.5*data["prior"] + .5*np.eye(7)[base]) + gain*scores
    proposed = prob.argmax(1)
    ix = np.arange(len(base))
    return np.where(prob[ix,proposed] - prob[ix,base] >= margin, proposed, base)


def preact(vec):
    return np.arctanh(np.clip(np.asarray(vec,dtype=np.float64), -1+1.e-8, 1-1.e-8))


def solve(data, arm, head, seed=None):
    vec = data["vec"]
    # Learn keys only on well-conditioned pooler channels. They recover the
    # preactivation from cached FP32 tanh activations without inverting the network.
    keys = np.argsort(np.max(np.abs(vec), axis=0), kind="stable")[:256]
    z = preact(vec)[:,keys]
    center, scale = z.mean(0), np.maximum(z.std(0), .08)
    x = (z - center) / scale
    y = data["target"].copy()
    if seed is not None:
        np.random.default_rng(seed).shuffle(y)
    score = readout(data, vec, arm, head)
    if arm == "public":
        direction = CW.T
        target = 3 * np.eye(7)[y]
    else:
        direction = head["scale"][-768:,None] * head["matrix"][-768:]
        target = np.eye(7)[y]
    residual = target - score
    residual -= residual.mean(1,keepdims=True)
    # Preserve responses on the donor's correct training examples.
    residual[score.argmax(1) == y] = 0
    importance = np.maximum(np.bincount(y, minlength=7)/len(y), .025)[y]**(-.35)
    x1 = np.column_stack([x, np.ones(len(x))])
    gram = x1.T @ (x1 * importance[:,None])
    reg = np.full(len(gram), .25 * np.mean(gram.diagonal()[:-1]))
    reg[-1] *= .02
    beta = np.linalg.solve(gram + np.diag(reg), x1.T @ (residual * importance[:,None]))
    centered = np.eye(7) - np.ones((7,7))/7
    # Algebraic pullback through the existing output map and average tanh slope.
    # The nonlinear tanh and all public auxiliary score features run at evaluation.
    response = ((1-vec**2).mean(0)[:,None] * direction) @ centered
    right = np.linalg.pinv(response, rcond=1.e-5)
    left = beta[:-1] / scale[:,None]
    intercept = beta[-1] - center @ left
    return {"keys":keys, "left":left, "intercept":intercept, "right":right}


def activation(vec, patch, gain):
    z = preact(vec)
    d = (z[:,patch["keys"]] @ patch["left"] + patch["intercept"]) @ patch["right"]
    value=np.tanh(z + gain*d)
    if not np.isfinite(value).all():
        raise ArithmeticError("Pooler edit produced a nonfinite activation.")
    return value


def bake(patch, gain):
    w, b = tensor(ORIGINAL,WEIGHT), tensor(ORIGINAL,BIAS)
    keys, left, right = patch["keys"], patch["left"], patch["right"]
    dw = right.T @ (left.T @ w[keys])
    db = (b[keys] @ left + patch["intercept"]) @ right
    return {WEIGHT:np.asarray(w + gain*dw,dtype=np.float32), BIAS:np.asarray(b + gain*db,dtype=np.float32)}


def summary(cm):
    scores = 2*cm.diagonal()/np.maximum(cm.sum(0)+cm.sum(1),1)
    return {"macro_f1":float(scores.mean()), "accuracy":float(cm.trace()/cm.sum()), "per_role_f1":dict(zip(LABELS,scores))}


def cms(gold, pred, lengths):
    labs = np.repeat(np.arange(len(lengths)), lengths)
    return np.bincount(49*labs + 7*gold + pred, minlength=49*len(lengths)).reshape(-1,7,7)


def measure(data, scores, arm):
    pred = choose_labels(data, scores, arm)
    if arm == "public":
        counts = cms(data["target"],pred,data["lengths"])
        return pred, counts
    full = data["base"].copy()
    full[data["pos"]] = pred
    return full, cms(data["gold"], full, np.diff(data["off"]))


def interval(old, new):
    rng = np.random.default_rng(2240721)
    values=[]
    for batch in np.array_split(rng.integers(len(old),size=(1500,len(old))), 30):
        def metric(c):
            m = c[batch].sum(1)
            return (2*np.diagonal(m, axis1=1, axis2=2)/np.maximum(m.sum(1)+m.sum(2),1)).mean(1)
        values.extend(metric(new)-metric(old))
    return np.quantile(values,[.025,.975]).tolist()


def fit():
    assert file_hash(ORIGINAL) == SPEC["backbone_weights_sha256"]
    train, select, head = view("fit"), view("select"), params()
    assert set(train["keys"]).isdisjoint(select["keys"])
    freeze = {"donor":file_hash(ORIGINAL), "readout":SPEC["checkpoints"][ARM], "fit_ids":train["keys"],
              "select_ids":select["keys"], "gradient_steps":0, "selection":"Choose a nonzero gain by highest select macro-F1, then accuracy, then smaller gain. Identity is reported separately.",
              "feature_inputs":{"fit":train["feature_sha"],"select":select["feature_sha"]}, "arms":{}}
    for arm in ("public","document"):
        baseline = summary(measure(select, readout(select,select["vec"],arm,head),arm)[1].sum(0))
        selection=[]
        patch=solve(train,arm,head)
        for gain in (.125,.25,.5,1.,2.):
            vec=activation(select["vec"],patch,gain)
            selected=summary(measure(select,readout(select,vec,arm,head),arm)[1].sum(0))
            selection.append([gain,selected["macro_f1"],selected["accuracy"]])
        gain,_,_=max(selection,key=lambda r:(r[1],r[2],-r[0]))
        record={"gain":gain,"select_baseline":baseline,"selection_grid":selection,"matrices":{}}
        for kind,solution in (("targeted",patch),("shuffled",solve(train,arm,head,72844))):
            # The shuffle control uses identical examples, training weights policy
            # and gain. Record its independent norm (not a norm-matched control).
            filename=arm+"-"+kind+".safetensors"
            save_file({k:np.ascontiguousarray(v) for k,v in solution.items()},str(HERE/filename))
            changes=bake(solution,gain)
            rec={}
            for key,value in changes.items():
                source=tensor(ORIGINAL,key)
                diff=value.astype(float)-source
                rec[key]={"shape":list(value.shape),"changed":int(np.count_nonzero(diff)),
                          "old_sha256":array_hash(source),"new_sha256":array_hash(value),
                          "l2_delta":float(np.linalg.norm(diff)),"l2_original":float(np.linalg.norm(source)),
                          "max_abs_delta":float(abs(diff).max())}
            record["matrices"][kind]={"file":filename,"sha256":file_hash(HERE/filename),"changes":rec}
        freeze["arms"][arm]=record
    write("freeze.json",freeze)
    print("Frozen", {k:v["selection_grid"] for k,v in freeze["arms"].items()},flush=True)


def evaluate(split):
    frozen=json.loads((HERE/"freeze.json").read_text())
    data,head=view(split),params()
    assert set(frozen["fit_ids"]).isdisjoint(data["keys"])
    out={"split":split, "judgments":len(data["keys"]), "total_sentences":len(data["gold"]), "reviewed_sentences":len(data["target"]),
         "view_sha":data["view_sha"],"feature_sha":data["feature_sha"],"freeze_sha":file_hash(HERE/"freeze.json"),"arms":{}}
    for arm,record in frozen["arms"].items():
        pred0, cm0=measure(data,readout(data,data["vec"],arm,head),arm)
        compare={"unchanged":summary(cm0.sum(0))}
        gold=data["target"] if arm=="public" else data["gold"]
        for kind, file in record["matrices"].items():
            assert file_hash(HERE/file["file"])==file["sha256"]
            solution=load_file(str(HERE/file["file"]))
            vec=activation(data["vec"],solution,record["gain"])
            pred,counts=measure(data,readout(data,vec,arm,head),arm)
            changed=pred!=pred0
            compare[kind]={**summary(counts.sum(0)), "delta_vs_unchanged_ci95":interval(cm0,counts),
                           "changed":int(changed.sum()), "corrected":int(np.sum(changed&(pred==gold))),
                           "broken":int(np.sum(changed&(pred0==gold)))}
        out["arms"][arm]=compare
    write(split+".json",out)
    print(split, {k:{a:v["macro_f1"] for a,v in t.items()} for k,t in out["arms"].items()},flush=True)


def parity():
    import torch
    from encoder import components
    from sentence_bert_referral_write import load_views, pack_ids, MODES
    model,tok,_=components(device="cpu")
    rows,_=load_views("select")
    ids=pack_ids(tok,rows[:4],MODES[1])[:12]
    batch=tok.pad([{"input_ids":s} for s in ids],padding=True,return_tensors="pt")
    with torch.inference_mode():
        before=model.bert(**batch).pooler_output.numpy()
    frozen=json.loads((HERE/"freeze.json").read_text())
    result={}
    for arm,record in frozen["arms"].items():
        file=record["matrices"]["targeted"]["file"]
        solution=load_file(str(HERE/file))
        changes=bake(solution,record["gain"])
        checkpoint=HERE/(arm+"-edited.model.safetensors")
        changed=patched_checkpoint(ORIGINAL,checkpoint,changes)
        for key,stats in record["matrices"]["targeted"]["changes"].items():
            assert changed[key]["changed"]==stats["changed"] and array_hash(tensor(checkpoint,key))==stats["new_sha256"]
        with torch.no_grad():
            model.bert.pooler.dense.weight.copy_(torch.from_numpy(tensor(checkpoint,WEIGHT)))
            model.bert.pooler.dense.bias.copy_(torch.from_numpy(tensor(checkpoint,BIAS)))
            after=model.bert(**batch).pooler_output.numpy()
        simulated=activation(before.astype(float),solution,record["gain"])
        err=float(np.max(abs(after-simulated)))
        assert err<2.e-4,err
        result[arm]={"checkpoint_sha256":file_hash(checkpoint),"torch_probe_count":len(ids),"max_abs_feature_error":err,
                     "native_public_label_disagreements":int(np.sum((after@CW.T+CB).argmax(1)!=(simulated@CW.T+CB).argmax(1))),
                     "probe_identity":"first twelve previous-two-and-target windows from select"}
    write("native-parity.json",result)
    print("Physical checkpoint parity",result,flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("action",choices=["fit","select","confirm","holdout","parity","test"])
    action=cli.parse_args().action
    if action=="fit": fit()
    elif action=="parity": parity()
    else: evaluate(action)
