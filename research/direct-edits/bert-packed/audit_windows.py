"""Verify token identity and geometry across every prepared case before fitting."""

import numpy as np

from encode import HERE, PAYLOAD, VIEWS, case_windows
from native import read, sha, write


def main():
    plan = read(HERE/"protocol.json")
    report = {}
    for name, keys in plan["groups"].items():
        tallied = {k:dict(windows=0,visible=0,wordpieces_retained=0) for k in VIEWS}
        sentences = 0
        long = 0
        max_row = 0
        for key in keys:
            path = HERE/"cases"/(key+".npz")
            with np.load(path,allow_pickle=False) as f:
                tokens, ends, special = (f[p] for p in ("tokens","ends","special"))
                gold, idx = f["gold"],f["rows"]
                assert np.all(np.diff(idx)>0) and len(gold)==len(ends)-1
            sentences += len(gold)
            long += int(np.sum(np.diff(ends)>PAYLOAD))
            max_row = max(max_row,int(max(np.diff(ends))))
            for view in VIEWS:
                w = case_windows(tokens,ends,special,key,view)
                active = [[] for _ in gold]
                for block in w:
                    owners = block.owners
                    assert 2 < len(block.tokens) <= 512 and len(block.positions)==len(owners)
                    wordpieces = block.tokens[block.positions]
                    for row in np.unique(owners):
                        active[row].append(wordpieces[owners==row])
                for pieces,a,b in zip(active,ends,ends[1:]):
                    np.testing.assert_array_equal(np.concatenate(pieces),tokens[a:b])
                tallied[view]["windows"] += len(w)
                tallied[view]["visible"] += sum(len(z.tokens) for z in w)
                tallied[view]["wordpieces_retained"] += len(tokens)
        assert sentences == plan["rows"][name]
        assert tallied["ordered"] == tallied["shuffled"]
        report[name] = dict(rows=sentences,cases=len(keys),rows_longer_than_payload=long,
                            max_wordpieces_in_one_sentence=max_row,views=tallied)
    write(HERE/"window-audit.json",dict(
        protocol_sha256=sha(HERE/"protocol.json"),checked=report,
        check="Recovered each row's complete ordered wordpiece vector, exactly once, in every context. Verified by direct array equality."
    ))
    print(report,flush=True)


if __name__=="__main__":
    main()
