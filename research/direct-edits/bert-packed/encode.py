"""Pool exactly the same target wordpieces in ordered, shuffled and isolated BERT contexts."""

import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
import time

import numpy as np
from safetensors import safe_open
import torch
from torch.nn.functional import linear

HERE = Path(__file__).resolve().parent
if (HERE.parent / "bert-t4/native.py").exists():
    sys.path.append(str(HERE.parent / "bert-t4"))
from native import donor_at, read, require_t4, save_arrays, sha, write

MAXIMUM = 512
PAYLOAD = 510
STRIDE = 382
VIEWS = ("ordered", "shuffled", "isolated")


@dataclass
class Window:
    tokens: np.ndarray
    positions: np.ndarray
    owners: np.ndarray
    source_row: int


def packed(tokens, ends, sep, cls, order):
    slices = [tokens[ends[j]:ends[j+1]] for j in order]
    length = len(tokens) + len(order)
    stream = np.empty(length, np.int32)
    owners = np.full(length, -1, np.int32)
    offset = 0
    for j, ids in zip(order, slices):
        stream[offset:offset+len(ids)] = ids
        owners[offset:offset+len(ids)] = j
        stream[offset+len(ids)] = sep
        offset += len(ids) + 1
    assert offset == length
    margin = np.full(length, -1, np.int16)
    take = np.full(length, -1, np.int32)
    bounds = []
    for a in range(0, length, STRIDE):
        b = min(a + PAYLOAD, length)
        bounds.append((a, b))
        d = np.minimum(np.arange(b-a), np.arange(b-a)[::-1])
        improve = d > margin[a:b]
        margin[a:b][improve] = d[improve]
        take[a:b][improve] = len(bounds) - 1
        if b == length:
            break
    assert np.all(take >= 0)
    for k, (a, b) in enumerate(bounds):
        places = np.flatnonzero((take[a:b] == k) & (owners[a:b] >= 0))
        yield Window(np.r_[cls, stream[a:b], sep], places+1, owners[a + places], -1)


def isolated(tokens, ends, sep, cls):
    for row, (a, b) in enumerate(zip(ends, ends[1:])):
        for x in range(a, b, PAYLOAD):
            y = min(x + PAYLOAD, b)
            visible = tokens[x:y]
            yield Window(np.r_[cls, visible, sep], np.arange(1, len(visible)+1),
                         np.full(len(visible), row, dtype=np.int32), row if x == a else -1)


def case_windows(tokens, ends, specials, case_id, view):
    cls, sep, _ = specials
    if view == "isolated":
        return list(isolated(tokens, ends, sep, cls))
    indices = np.arange(len(ends)-1)
    if view == "shuffled":
        seed = int.from_bytes(hashlib.sha256(("wordpiece-order-v1|" + case_id).encode()).digest()[:8], "little")
        indices = np.random.default_rng(seed).permutation(indices)
    return list(packed(tokens, ends, sep, cls, indices))


class TopTwo:
    def __init__(self, model, head):
        self.model = model
        self.head = head
        self.intermediate = None
        self.hook = model.encoder.layer[10].register_forward_hook(self.capture)

    def capture(self, module, args, result):
        self.intermediate = result[0] if isinstance(result, (tuple, list)) else result

    @torch.inference_mode()
    def run(self, windows, batch, device, pad, count):
        # Preserve deterministic window order within ties while grouping lengths for padded-token efficiency.
        jobs = sorted(windows, key=lambda w:len(w.tokens))
        counts = np.zeros(count, np.int32)
        out = np.zeros((count,2,768), np.float32)
        has_source = any(w.source_row >= 0 for w in jobs)
        source = np.full((count,7), np.nan, np.float32) if has_source else None
        charged = 0
        for offset in range(0, len(jobs), batch):
            active = jobs[offset:offset+batch]
            longest = max(len(w.tokens) for w in active)
            tokens = np.full((len(active),longest), pad, dtype=np.int64)
            mask = np.zeros_like(tokens)
            for i, window in enumerate(active):
                begin = len(window.tokens)
                tokens[i,:begin] = window.tokens
                mask[i,:begin] = 1
            result = self.model(torch.from_numpy(tokens).to(device), attention_mask=torch.from_numpy(mask).to(device))
            assert self.intermediate.shape == result.last_hidden_state.shape
            high = result.last_hidden_state
            middle = self.intermediate
            self.intermediate = None
            charged += tokens.size
            if has_source:
                raw_scores = linear(result.pooler_output, *self.head).cpu().numpy()
            for i, window in enumerate(active):
                if window.source_row >= 0:
                    source[window.source_row] = raw_scores[i]
                place = window.positions
                owners = window.owners
                if not len(place):
                    continue
                boundary = np.flatnonzero(np.r_[True, owners[1:] != owners[:-1]])
                j = owners[boundary]
                ni = np.diff(np.r_[boundary, len(owners)])
                for depth, hidden in enumerate((middle, high)):
                    value = hidden[i,place,:].cpu().numpy()
                    out[j,depth] += np.add.reduceat(value, boundary, axis=0)
                counts[j] += ni
        assert np.min(counts) >= 1 and np.isfinite(out).all()
        out /= counts[:,None,None]
        if source is not None:
            assert np.isfinite(source).all()
        return out, counts, source, charged


def collect(directory, checkpoint, out, groups, device, batch, limit=None, threads=2):
    torch.set_num_threads(threads)
    backend = require_t4() if device == "cuda" else {"devices":["local CPU"], "torch":str(torch.__version__), "cuda":None}
    plan = read(directory / "protocol.json")
    verified = read(directory / "review.json")
    assert verified["protocol_sha256"] == sha(directory / "protocol.json")
    net = donor_at(directory, checkpoint, device)
    with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
        head = tuple(source.get_tensor("classifier."+key).to(device) for key in ("weight", "bias"))
    model = TopTwo(net, head)
    out.mkdir(parents=True, exist_ok=True)
    code = sha(Path(__file__))
    for group in groups:
        begin = time.perf_counter()
        cases = plan["groups"][group][:limit] if limit is not None else plan["groups"][group]
        seen = 0
        for key in cases:
            data = directory / "cases" / (key + ".npz")
            path = out / (key + ".npz")
            receipt = out / (key + ".json")
            signature = dict(backend=backend, implementation=code, case_sha256=sha(data),
                             model=verified["model"]["sha256"])
            if path.exists() and receipt.exists():
                old = read(receipt)
                if all(old[k] == v for k,v in signature.items()) and old["features_sha256"] == sha(path):
                    seen += old["rows"]
                    continue
            with np.load(data, allow_pickle=False) as a:
                tokens, ends, special = a["tokens"], a["ends"], a["special"]
                assert a["fingerprint"].item() == plan["text_sha256"][key]
            count = len(ends) - 1
            feature = {}
            cost = {}
            expected = np.diff(ends)
            for view in VIEWS:
                windows = case_windows(tokens, ends, special, key, view)
                state, retained, scores, padded = model.run(windows, batch, device, special[2], count)
                np.testing.assert_array_equal(retained, expected)
                feature[view] = state
                if scores is not None:
                    feature["source_scores"] = scores
                cost[view] = dict(windows=len(windows), padded_positions=padded,
                                  visible_positions=sum(len(z.tokens) for z in windows), retained_wordpieces=len(tokens))
            assert cost["ordered"] == cost["shuffled"]
            save_arrays(path, **feature)
            write(receipt, dict(**signature, judgment=key, rows=count, cost=cost, features_sha256=sha(path)))
            seen += count
            print(group, key, count, len(tokens), "cumulative sentences", seen, flush=True)
        print(group, "encoded", len(cases), "judgments", seen, "sentences", round(time.perf_counter()-begin,2), flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directory", type=Path)
    cli.add_argument("checkpoint", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("groups", nargs="+", choices=("fit","select","shared"))
    cli.add_argument("--device", choices=("cpu","cuda"), default="cpu")
    cli.add_argument("--batch", type=int, default=24)
    cli.add_argument("--limit", type=int)
    cli.add_argument("--threads", type=int, default=2)
    a = cli.parse_args()
    collect(a.directory, a.checkpoint, a.output, a.groups, a.device, a.batch, a.limit, a.threads)


if __name__ == "__main__":
    main()
