"""CPU prediction with the frozen sequence model and either pairwise readout."""
import argparse
import json
from pathlib import Path

import numpy as np

import sentence_pairwise_consensus as consensus
import sentence_pairwise_repair as repair
import sentence_phrase_sequence as seq


def predictor(method):
    direct, weights, direct_hash = consensus.inputs()
    if method == "direct":
        option = direct["arms"]["sequence only"]["choice"]
    else:
        path = repair.receipt(consensus.TAG)
        replay = json.loads(path.read_text())
        assert replay["repair_freeze_sha256"] == direct_hash
        option = replay["arms"]["sequence only"]["choice"]
    if option is None:
        return None

    def revise(bank, lex, unary, base):
        feature = repair.dense_basis(bank, lex)
        margins = repair.score_pairs(feature, weights, option[0])
        unprotected = np.zeros(len(base), dtype=bool)
        if method == "direct":
            _, scale, top, threshold, guard = option
            choices = repair.proposals(margins, base, unary, scale, top)
            return repair.decide(choices, base, unprotected, threshold, guard)
        _, rule, strength, threshold, guard = option
        alt = consensus.reconcile(margins, unary, rule)
        return consensus.decide(alt, base, unary, unprotected, strength, threshold, guard)

    return revise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("direct", "consensus"), default="direct")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--raw", action="store_true")
    args = parser.parse_args()
    data = args.input.read_bytes().decode("utf-8")
    results = seq.run_input(data if args.raw else json.loads(data), args.raw, "spans",
                            predictor(args.method))
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
