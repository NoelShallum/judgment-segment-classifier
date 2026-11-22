"""Measure the exact best attainable seven-role score for fixed referral positions."""
import numpy as np

import sentence_pairwise_consensus as consensus
import sentence_pairwise_repair as repair
from sentence_document_graph_write import cm, file_hash, row_summary, save_json

P = repair.P
BERT = repair.fusion.bert
TAG = "pairwise-referral-coverage"
METHODS = ("orig", "confidence", "margin", "pair dissent", "rare role dissent")
FRACTIONS = (.05, .10, .20, .33)


def routing(bank, lex, recipe, weights, option):
    base, unary, _ = repair.incumbent(bank, lex, recipe)["sequence only"]
    mode, scale, top, threshold, guard = option
    pair = repair.score_pairs(repair.dense_basis(bank, lex), weights, mode)
    contest = repair.proposals(pair, base, unary, scale, top)
    pred = repair.decide(contest, base, np.zeros(len(base),dtype=bool), threshold, guard)
    prior = np.exp(unary - unary.max(axis=1, keepdims=True))
    prior /= prior.sum(axis=1, keepdims=True)
    index = np.arange(len(base))
    keep = prior[index,pred].copy()
    rival = prior.copy()
    rival[index,pred] = 0.
    competitor = rival.argmax(1)
    gap = keep - rival[index,competitor]
    proposal = repair.proposals(pair, pred, unary, .27, 4)
    disagreement = np.clip(proposal.max(1),-.75,.75)
    rare = np.array([0.,.05,.25,.08,.18,0.,.12])
    scores = {"confidence":1. - keep,"margin":-gap,
              "pair dissent":-gap+.5*disagreement,
              "rare role dissent":-gap+.5*disagreement+np.maximum(rare[pred],rare[competitor])}
    return pred, scores


def gather(split, recipe, weights, option):
    bank = P.dense.prepare(split)
    lex = P.cached_scores(split)
    pred, scores = routing(bank, lex, recipe, weights, option)
    origin = BERT.load_views(split)[0]
    np.testing.assert_array_equal(bank[0], [r[0] for r in origin])
    np.testing.assert_array_equal(bank[2], np.concatenate([r[1] for r in origin]))
    assert row_summary(cm(bank[2],pred)) == P.dense.read(repair.TAG)["splits"][split]["arms"]["sequence only"]["labels"]
    ordered = {}
    for key,values in scores.items():
        ordered[key] = [np.argsort(-values[a:b],kind="stable")+a for a,b in zip(bank[1],bank[1][1:])]
    origin = [row[3]+a for a,row in zip(bank[1],origin)]
    return bank, pred, ordered, origin


def choose(method, ratio, bank, order, original):
    if ratio == "matched":
        sizes = [len(x) for x in original]
    else:
        sizes = np.ceil(np.diff(bank[1])*float(ratio)).astype(int)
    if method == "orig":
        assert ratio == "matched"
        return np.concatenate(original)
    return np.concatenate([rows[:k] for rows,k in zip(order[method],sizes)])


def ceiling(bank, pred, positions):
    gold = bank[2]
    guess = pred.copy()
    guess[positions] = gold[positions]
    errors = gold != pred
    repaired = positions[errors[positions]]
    return {"ceiling":row_summary(cm(gold,guess)),"referred":len(positions),
            "errors_reachable":len(repaired),"total_errors":int(errors.sum()),
            "gold_errors_reachable":np.bincount(gold[repaired],minlength=7).tolist(),
            "gold_errors_total":np.bincount(gold[errors],minlength=7).tolist()}


def main():
    parent,w,source = consensus.inputs()
    option = parent["arms"]["sequence only"]["choice"]
    recipe = repair.load_recipe()
    options = {
        "orig": ("orig", "matched"),
        **{k:(k,"matched") for k in METHODS if k!="orig"}
    }
    bank, pred, order, original = gather("select",recipe,w,option)
    options_score = {name:ceiling(bank,pred,choose(*policy,bank,order,original))
                     for name,policy in options.items()}
    selected = max(METHODS[1:],key=lambda k:options_score[k]["ceiling"]["macro_f1"])
    receipt = P.RESULT/(TAG+"-frozen.json")
    save_json(receipt,{"repair_freeze_sha256":source,"chosen":selected,"selection_cases":bank[0].tolist(),
                       "matched":"Same sentence count as the existing BERT router within each judgment",
                       "fractions":FRACTIONS,"development":options_score})
    results = {"freeze_sha256":file_hash(receipt),"splits":{}}
    for split in ("confirm","holdout","test"):
        bank,pred,order,original = gather(split,recipe,w,option)
        old = row_summary(cm(bank[2],pred))
        budget = {"existing BERT positions":("orig","matched"),
                  "reselected, same budget":(selected,"matched"),
                  **{f"reselected, {ratio:.0%}":(selected,ratio) for ratio in FRACTIONS}}
        test = {name:ceiling(bank,pred,choose(*args,bank,order,original)) for name,args in budget.items()}
        oracle = test["existing BERT positions"]
        assert oracle["referred"] == test["reselected, same budget"]["referred"]
        results["splits"][split]={"cases":len(bank[0]),"baseline":old,"coverage":test}
        save_json(P.RESULT/(TAG+".json"),results)
        print(split, selected, {k:(q["referred"],q["errors_reachable"],q["ceiling"]["macro_f1"])
                                for k,q in test.items()},flush=True)


if __name__ == "__main__":
    main()
