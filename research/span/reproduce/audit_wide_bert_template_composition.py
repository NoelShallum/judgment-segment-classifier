"""Measure the existing exact-template edit with the wider confidence reviewer."""
import argparse
import json

import numpy as np

import sentence_bert_reviewer_gain as gain
import sentence_pairwise_repair as repair
import sentence_phrase_sequence as sequence
import sentence_referral_actual_replay as replay
import sentence_template_memory_write as template
import sentence_wide_bert_multiclass_write as wide
from sentence_document_graph_write import file_hash, paired, row_summary, save_json


TAG = "sentence-wide-bert-template"
P = repair.P
RATES = (.15,.20)


def positions(offsets,referred,rate):
    pieces = []
    for a,b in zip(offsets,offsets[1:]):
        local = referred[(referred>=a)&(referred<b)]
        count = int(np.ceil(rate*(b-a)))
        assert count<=len(local)
        pieces.append(local[:count])
    return np.concatenate(pieces)


def evaluate(splits):
    context = gain.plans()
    upstream = wide.read_wide()
    corpus,weights,source_hash = template.read_weights()
    freeze = repair.receipt(template.TAG)
    policy = json.loads(freeze.read_text())
    assert policy["source_sha256"]==source_hash
    assert policy["pairwise_freeze_sha256"]==context[5][repair.TAG]
    config = {"wide_method":upstream["method"],"template_freeze_sha256":file_hash(freeze),
              "template_source_sha256":source_hash,"tested_rates":list(RATES)}
    path = P.RESULT/(TAG+".json")
    result = {"method":config,"splits":{}}
    if path.is_file():
        saved = json.loads(path.read_text())
        assert saved["method"]==config
        result = saved
    budget = json.loads((P.RESULT/"sentence-wide-bert-budget.json").read_text())
    assert budget["source_method"]==upstream["method"]
    for split in splits:
        data = wide.inputs(split,context,upstream)
        bank = data.bank
        assert not set(bank[0]) & set(corpus["train_cases"])
        if split!="select":
            assert not set(bank[0]) & set(policy["selection_cases"])
        docs = P.docs(split)
        np.testing.assert_array_equal(bank[0],[d["id"] for d in docs])
        strings = [d["text"][a:b] for d in docs for a,b in d["reference_units"]]
        assert len(strings)==len(data.base)
        memory = template.lookup(strings,weights,bank[1])
        _,unary = sequence.source_unary(bank,P.cached_scores(split))
        assert policy["choice"] is not None
        material = template.propose(data.base,unary,memory,*policy["choice"])
        q = {"cases":len(docs),"embedding_sha256":data.signature,"rates":{}}
        previous_counts = None
        if split=="test":
            import compare_broad_edits_to_bert as broad

            information = context[3]
            prior_rule = context[4]["arms"]["BERT only"]["choice"]
            narrow = broad.bert_reference(split,bank,data.base,information,prior_rule)
            previous = np.where(narrow!=data.base,narrow,material)
            previous_counts = P.dense.paired_confusion(bank,previous)
            q["previous_end_to_end"] = row_summary(previous_counts.sum(0))
            published = P.dense.read(template.TAG+"-bert")["splits"][split]["candidate"]
            assert q["previous_end_to_end"]==published
        for rate in RATES:
            selected = positions(bank[1],data.positions,rate)
            keep = data.base.copy()
            keep[selected] = data.incumbent[selected]
            has_bert_revision = keep!=data.base
            composed = np.where(has_bert_revision,keep,material)
            revised = np.flatnonzero(composed!=keep)
            result_labels,counts,trial = replay.scores(bank,keep,composed,revised)
            np.testing.assert_array_equal(composed,result_labels)
            before = P.dense.paired_confusion(bank,keep)
            observed = row_summary(before.sum(0))
            if split in budget["splits"]:
                assert observed==budget["splits"][split]["routes"]["confidence"][str(rate)]["labels"]
            q["rates"][str(rate)] = {"reviewed":len(selected),"base":observed,"composed":trial,
                                    "paired":paired(counts,before)}
            if previous_counts is not None:
                q["rates"][str(rate)]["paired_to_previous_end_to_end"] = paired(counts,previous_counts)
            print(split,rate,observed["macro_f1"],trial["labels"]["macro_f1"],trial["changed"],flush=True)
        result["splits"][split] = q
        save_json(path,result)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("splits",nargs="*")
    args = parser.parse_args()
    evaluate(args.splits or ("select","confirm"))
