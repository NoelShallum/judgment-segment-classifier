"""Write low-rank role-contrast directions into both document LSTM input gates."""
import hashlib
from itertools import product

import numpy as np
from safetensors.numpy import load_file, save_file
from scipy.linalg import solve
import torch

from discourse_span_edit import text_features
from hier_model import CRF, LABELS, HierModel, path_labels
from hier_trial import DONOR, OUT
from preserved_emitter_edit import edit_documents
from program_trial import save_json
from raw_emitter_write import paired
from seven_paths import ROOT
from span_eval import counts, sets, summarize
from transition_span_edit import sample_whole
from weight_edit import file_hash

GATES = ("weight_ih_l0", "weight_ih_l0_reverse", "bias_ih_l0", "bias_ih_l0_reverse")


def pool(model, cases, gold=False):
    dest = ROOT / "cache/lstm-role-subspace-input"
    dest.mkdir(exist_ok=True, parents=True)
    for i, d in enumerate(cases):
        stem = str(d["id"]) + ("-gold" if gold else "-raw")
        offsets = d["reference_units"] if gold else d["units"]
        sig = hashlib.sha256(d["text"].encode()).hexdigest()[:12]
        path = dest / f"{stem}-{sig}.npy"
        if not path.exists():
            values = model.words([d["text"][a:b] for a,b in offsets]).numpy()
            with path.with_suffix(".part").open("wb") as stream:
                np.save(stream, values, allow_pickle=False)
            path.with_suffix(".part").replace(path)
        values = np.load(path, allow_pickle=False)
        assert len(values) == len(offsets)
        yield d, values


def directions(data):
    x = np.concatenate([a for _, a in data]).astype(np.float64)
    role = np.concatenate([d["reference_roles"] for d, _ in data])
    mean = x.mean(0)
    means = np.stack([x[role == j].mean(0) for j in range(7)])
    residual = x - means[role]
    cov = residual.T @ residual / len(x)
    reg = .5 * cov.trace() / len(cov)
    matrix = cov + reg * np.eye(len(cov))
    vectors = {
        "speaker": np.stack((means[3]-means[4], .5*(means[3]+means[4])-means[5])),
        "all_roles": means[1:] - means[0],
    }
    z = {k:np.linalg.qr(solve(matrix, v.T, assume_a="pos"))[0] for k,v in vectors.items()}
    return z, mean, len(x)


def edit(original, subspace, center, gain):
    new = {}
    for weight, bias in zip(GATES[:2], GATES[2:]):
        w = original[weight]
        delta = gain * ((w @ subspace) @ subspace.T)
        new[weight] = (w + delta).astype("f4")
        new[bias] = (original[bias] - delta @ center).astype("f4")
    return new


def apply(model, arrays):
    with torch.no_grad():
        for name in GATES:
            getattr(model.doc, name).copy_(torch.from_numpy(arrays[name]))


def measure(model, bank, parent):
    scores = []
    for d, x, authored in bank:
        emission = model.logits(model.contextualize(torch.from_numpy(x))) + authored
        scores.append(counts(d, d["units"], path_labels(emission, parent[CRF])))
    return np.stack(scores)


def add_cues(model, cases, parent):
    bank = []
    for d, x in pool(model, cases):
        cue, position = text_features(d)
        bank.append((d, x, cue @ parent["cue_head"] + position @ parent["position_head"]))
    return bank


def main():
    torch.set_num_threads(2)
    model = HierModel(DONOR)
    parent = load_file(str(OUT / "whole-discourse-exact.safetensors"))
    original = {name:getattr(model.doc, name).detach().numpy().copy() for name in GATES}
    # Fixed evenly spaced source cases keep this CPU screen manageable.
    source = edit_documents()[::3]
    vec, center, rows = directions(list(pool(model, source, gold=True)))
    print("Wrote candidate directions from", len(source), "cases and", rows, "rows", flush=True)

    validation = sample_whole()
    bank = {k:add_cues(model, v, parent) for k,v in validation.items()}
    baseline = {k:measure(model, b, parent) for k,b in bank.items()}

    def rank(aggregate):
        v = summarize(aggregate.sum(0))
        return v["exact_labeled_spans"]["f1"], v["token_accuracy"]

    selected = ["parent", 0.]
    best = rank(baseline["select"])
    chosen = original
    grid = []
    for family, gain in product(vec, (-.30,-.10,.10,.30)):
        patch = edit(original, vec[family], center, gain)
        apply(model, patch)
        score = rank(measure(model, bank["select"], parent))
        grid.append([family,gain,*score])
        print(family,gain,score,flush=True)
        if score > best:
            best, chosen, selected = score, patch, [family,gain]
    dest = OUT / "whole-lstm-input-subspace.safetensors"
    save_file({**parent, **{"emitter.lstm."+k:v for k,v in chosen.items()}}, str(dest))
    sha = file_hash(dest)
    receipt = ROOT / "results/lstm-input-subspace-frozen.json"
    save_json(receipt, {"selected":selected, "selection_grid":grid, "checkpoint_sha256":sha})
    print("Froze",selected,"before test",flush=True)
    test = add_cues(model, sets()["test"], parent)
    apply(model, original)
    baseline["test"] = measure(model, test, parent)
    bank["test"] = test
    physical = load_file(str(dest))
    assert file_hash(dest) == sha
    apply(model, {k:physical["emitter.lstm."+k] for k in GATES})
    records = {}
    for name,b in bank.items():
        q = measure(model,b,parent)
        z = baseline[name]
        records[name] = {"cases":len(q), "candidate":summarize(q.sum(0)),
                         "parent":summarize(z.sum(0)), "gap_ci95":paired(q,z)}
        print(name,rank(q),rank(z),flush=True)
    save_json(ROOT/"results/lstm-input-subspace-write.json",
              {"selected":selected,"train_cases":len(source),"train_rows":rows,"scores":records,
               "checkpoint_sha256":sha,"selection_receipt_sha256":file_hash(receipt),
               "method":"Closed-form within-role covariance whitening of legal role contrasts. Physically adjust both directions of the existing document LSTM input matrix and bias, with mean activation preserved. Reuse unchanged emissions and discourse weights. No gradient steps."})


if __name__ == "__main__":
    main()
