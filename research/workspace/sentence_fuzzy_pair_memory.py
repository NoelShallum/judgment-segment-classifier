"""Write a sparse near-match memory for the none/reasoning contrast."""
import argparse
from hashlib import blake2b
import heapq
from itertools import product
import json

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer

import sentence_pairwise_consensus as consensus
import sentence_pairwise_repair as repair
import sentence_phase_readout_write as phase
import sentence_referral_actual_replay as replay
import sentence_template_memory_write as template
from sentence_document_graph_write import cm, file_hash, paired, row_summary, save_json


P = repair.P
TAG = "sentence-fuzzy-pair-memory"
BASE = P.ROOT / "cache" / TAG
PAIR = (0, 5)
PER_ROLE = 16384
NEIGHBORS = 8
VECTOR = HashingVectorizer(analyzer="char", ngram_range=(3, 5), n_features=2**18,
                           alternate_sign=False, norm="l2", dtype=np.float32)


def canonical(text):
    return template.DIGITS.sub("#", template.canonical(text))


def digest(text, person):
    return int.from_bytes(blake2b(text.encode(), digest_size=8, person=person).digest(), "big")


def write(source):
    plan = json.loads(source.read_text())
    path = template.DATA
    assert plan["csv_sha256"]["train"] == file_hash(path)
    allowed = set(plan["train_cases"])
    heaps = {y: [] for y in PAIR}
    for index, (case, rows) in enumerate(template.dataset("train")):
        if case not in allowed:
            continue
        alternatives = {}
        for text, y in rows:
            text = canonical(text)
            if not (y in PAIR and 30 <= len(text) <= 768):
                continue
            alternatives[text] = alternatives.get(text, 0) | (1 << y)
        for text, mask in alternatives.items():
            if mask & (mask-1):
                continue
            y = mask.bit_length()-1
            rank = digest(case+"\t"+text, b"AttoFlow-sample")
            entry = (-rank, index, text)
            heap = heaps[y]
            if len(heap) < PER_ROLE:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)
    keys, cases, labels, texts = [], [], [], []
    for y, heap in heaps.items():
        for _, case, text in sorted(heap, reverse=True):
            keys.append(digest(text, b"AttoFlow-fuzzy"))
            cases.append(case)
            labels.append(y)
            texts.append(text)
    assert len(texts) == 2 * PER_ROLE
    matrix = VECTOR.transform(texts).tocsr()
    BASE.mkdir(parents=True, exist_ok=True)
    bank = BASE / "rows.npz"
    tensor = BASE / "coefficients.npz"
    np.savez_compressed(bank, keys=np.asarray(keys,np.uint64), cases=cases, labels=np.asarray(labels,np.int8))
    sparse.save_npz(tensor,matrix)
    receipt = {"dataset_sha256":file_hash(source), "row_file_sha256":file_hash(bank),
               "coefficient_file_sha256":file_hash(tensor), "shape":list(matrix.shape),
               "nonzero_coefficients":int(matrix.nnz), "gradient_steps":0}
    save_json(BASE/"source.json",receipt)
    print("Memory",receipt,flush=True)
    return receipt


def assets(source):
    proof_file = BASE/"source.json"
    receipt = json.loads(proof_file.read_text()) if proof_file.is_file() else write(source)
    bank, tensor = BASE/"rows.npz", BASE/"coefficients.npz"
    assert receipt["dataset_sha256"] == file_hash(source)
    assert receipt["row_file_sha256"] == file_hash(bank)
    assert receipt["coefficient_file_sha256"] == file_hash(tensor)
    index = sparse.load_npz(tensor).T.tocsr()
    with np.load(bank,allow_pickle=False) as z:
        values = tuple(z[k] for k in ("keys","cases","labels"))
    assert index.shape == tuple(receipt["shape"][::-1])
    return index, values, receipt


def query(texts, index, values):
    keys, cases, labels = values
    sims = np.zeros((len(texts), NEIGHBORS), np.float32)
    roles = np.zeros_like(sims,dtype=np.int8)
    for start in range(0,len(texts),24):
        text = texts[start:start+24]
        fingerprints = [digest(t,b"AttoFlow-fuzzy") for t in text]
        current = (VECTOR.transform(text) @ index).tocsr()
        for j,key in enumerate(fingerprints):
            a,b = current.indptr[j:j+2]
            w,idx = current.data[a:b],current.indices[a:b]
            use = (w >= .50) & (keys[idx] != key)
            w,idx = w[use],idx[use]
            order = np.lexsort((idx,-w))
            documents, used_texts = set(), set()
            k = 0
            for rank in order:
                place = idx[rank]
                doc, marker = cases[place], keys[place]
                if doc in documents or marker in used_texts:
                    continue
                documents.add(doc)
                used_texts.add(marker)
                sims[start+j,k], roles[start+j,k] = w[rank], labels[place]
                k += 1
                if k == NEIGHBORS:
                    break
    return sims, roles


def features(split,parent,weights,recipe,index,values):
    bank,_,base,unary = phase.view(split,parent,weights,recipe)
    docs = P.docs(split)
    np.testing.assert_array_equal(bank[0],[d["id"] for d in docs])
    texts = [d["text"][a:b] for d in docs for a,b in d["reference_units"]]
    assert len(texts) == len(base)
    row = np.arange(len(base))
    other = np.where(base==PAIR[0],PAIR[1],PAIR[0])
    margin = unary[row,base]-unary[row,other]
    plausible = np.flatnonzero(np.isin(base,PAIR)&(margin<=1.2))
    strings = [canonical(texts[j]) for j in plausible]
    keep = np.array([30<=len(t)<=768 for t in strings])
    select = plausible[keep]
    sims,roles = query([t for t,k in zip(strings,keep) if k],index,values)
    print(split,len(base),"sentences and",len(select),"queried",flush=True)
    return bank,base,select,other[select],margin[select],sims,roles


def apply(view,option):
    _,base,select,other,margin,sims,roles = view
    minimum,quota,agreement,limit = option
    support = sims>=minimum
    weights = np.where(support,sims,0.)
    for_rival = (roles==other[:,None]) & support
    accepted = ((for_rival.sum(1)>=quota)&((weights*for_rival).sum(1)>=agreement*weights.sum(1))&(margin<=limit))
    candidate = base.copy()
    candidate[select[accepted]] = other[accepted]
    return candidate


def run(splits):
    parent,w,parent_sha = consensus.inputs()
    recipe = repair.load_recipe()
    source = P.RESULT/(template.TAG+"-source.json")
    train = set(json.loads(source.read_text())["train_cases"])
    index, values, memory = assets(source)
    dev = features("select",parent,w,recipe,index,values)
    assert not train & set(dev[0][0])
    old = row_summary(cm(dev[0][2],dev[1]))
    best = (old["macro_f1"],old["accuracy"],0)
    pick = None
    grid = []
    for option in product((.55,.68,.78,.86),(2,3,4),(.67,.82,1.),(.20,.55,1.2)):
        candidate = apply(dev,option)
        s = row_summary(cm(dev[0][2],candidate))
        grade = (s["macro_f1"],s["accuracy"],-int(np.count_nonzero(candidate!=dev[1])))
        grid.append([option,grade])
        if grade > best and grade[1] >= old["accuracy"]-.001:
            best,pick = grade,option
    frozen = repair.receipt(TAG)
    save_json(frozen,{"parent_sha256":parent_sha,"memory":memory,"gradient_steps":0,
                      "selection_cases":dev[0][0].tolist(),"choice":pick,"selection":best,"grid":grid})
    print("Frozen",pick,best,flush=True)
    answer = {"freeze_sha256":file_hash(frozen),"splits":{}}
    for split in splits:
        data = dev if split=="select" else features(split,parent,w,recipe,index,values)
        bank,base,positions,*_ = data
        assert not train & set(bank[0])
        if split!="select":
            assert not set(bank[0]) & set(dev[0][0])
        candidate = apply(data,pick) if pick else base
        _,now,trial = replay.scores(bank,base,candidate,np.flatnonzero(candidate!=base))
        before = P.dense.paired_confusion(bank,base)
        record = {"cases":len(bank[0]),"queried":len(positions),"baseline":row_summary(before.sum(0)),
                  "edited":trial,"paired":paired(now,before)}
        print(split,record["baseline"]["macro_f1"],trial["labels"]["macro_f1"],trial["changed"],flush=True)
        answer["splits"][split] = record
        save_json(P.RESULT/(TAG+".json"),answer)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("splits",nargs="*")
    args=cli.parse_args()
    run(args.splits or ("select","confirm"))
