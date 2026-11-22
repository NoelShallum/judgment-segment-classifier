"""Compose frozen edits outside the existing InLegalBERT corrections."""
import argparse
import json

import numpy as np
from safetensors.numpy import load_file

import mixedbread_referral_overlay as overlay
import sentence_full_corpus_lexical as corpus
import sentence_pairwise_consensus as consensus
import sentence_pairwise_repair as repair
import sentence_phase_readout_write as phase
import sentence_phrase_sequence as seq
from sentence_document_graph_write import OUT, cm, file_hash, paired, row_summary, save_json

P=repair.P
B=overlay.B
SPLITS=("select","confirm","holdout","test")


def bert_reference(split, bank, base, config, rule):
    rows, vectors, _=B.encoded(split,(overlay.BERT,))
    np.testing.assert_array_equal(bank[0],[r[0] for r in rows])
    np.testing.assert_array_equal(bank[2],np.concatenate([r[1] for r in rows]))
    np.testing.assert_array_equal(bank[3],np.concatenate([r[2] for r in rows]))
    positions=np.concatenate([q+lo for lo,(_,_,_,q,_,_,_) in zip(bank[1],rows)])
    prior=np.concatenate([r[5] for r in rows])
    matrix=OUT/(B.TAG+"-2.safetensors")
    assert file_hash(matrix)==config["checkpoints"][overlay.BERT]
    w=load_file(str(matrix))
    mode,*_=config["frozen_modes"][overlay.BERT]
    scores=B.response(B.design(rows,vectors[overlay.BERT],mode,w),w)
    gain,gap,veto=rule
    original=np.concatenate([r[2][r[3]] for r in rows])
    result=overlay.edit(base,positions,prior,scores,gain,gap,original if veto else None)
    assert row_summary(cm(bank[2],result))==P.dense.read(overlay.TAG)["splits"][split]["arms"]["BERT only"]["labels"]
    return result


def load_addon(arm):
    module=phase if arm=="phase" else corpus
    freeze=repair.receipt(module.TAG)
    config=json.loads(freeze.read_text())
    if arm=="phase":
        name=max(config["arms"],key=lambda k:config["arms"][k]["selection"])
        index=phase.MODES.index(name)
    else:
        name=corpus.ARMS[1]
        index=1
    option=config["arms"][name]
    weights=None
    if option["settings"]:
        path=OUT/(module.TAG+f"-{index}.safetensors")
        assert file_hash(path)==option["sha256"]
        weights=load_file(str(path))
    return module.TAG,name,option,weights,file_hash(freeze)


def main(arm):
    parent,pair_w,digest=consensus.inputs()
    recipe=repair.load_recipe()
    bert=B.read(B.TAG+"-frozen")
    ov_path=repair.receipt(overlay.TAG)
    ov=json.loads(ov_path.read_text())
    assert ov["sources"][B.TAG]==file_hash(repair.receipt(B.TAG))
    assert ov["sources"][repair.TAG]==digest
    tag,name,pick,w,source=load_addon(arm)
    rule=ov["arms"]["BERT only"]["choice"]
    if arm=="corpus":
        memory,mem_w,m_sha=corpus.memory()
        assert corpus.P.dense.read(tag+"-frozen")["memory_freeze_sha256"]==m_sha
    receipt=repair.receipt(tag+"-bert")
    save_json(receipt,{"source_edit_sha256":source,"source_bert_overlay_sha256":file_hash(ov_path),
                       "pairwise_freeze_sha256":digest,"arm":name,"gradient_steps":0,
                       "policy":"Keep every accepted BERT revision. Apply the frozen all-sentence edit to the remaining rows."})
    evidence={"freeze_sha256":file_hash(receipt),"splits":{}}
    for split in SPLITS:
        bank,lex,base,u=phase.view(split,parent,pair_w,recipe)
        original=bert_reference(split,bank,base,bert,rule)
        alternative=base
        if w is not None:
            if arm=="phase":
                score=phase.score(bank,lex,w,name)
            else:
                addon=corpus.lexical(bank,split,memory,mem_w)
                x=corpus.design(bank,lex,addon,name)
                score=corpus.standardized(x,w)@w["weight"]
            alternative=phase.edit(bank,base,u,score,pick["settings"][2:])
        assert row_summary(cm(bank[2],alternative))==P.dense.read(tag)["splits"][split]["arms"][name]["labels"]
        bert_changed=original!=base
        alt_changed=alternative!=base
        combined=np.where(bert_changed,original,alternative)
        np.testing.assert_array_equal(combined[bert_changed],original[bert_changed])
        before,after=(P.dense.paired_confusion(bank,z) for z in (original,combined))
        sp_before,sp_after=(seq.span_tally(bank,z) for z in (original,combined))
        case={"cases":len(bank[0]),"base":row_summary(before.sum(0)),"candidate":row_summary(after.sum(0)),
              "paired_base":paired(after,before),"base_spans":seq.span_info(sp_before),
              "candidate_spans":seq.span_info(sp_after),"paired_spans":seq.span_intervals(sp_after,sp_before),
              "changed":int(np.count_nonzero(combined!=original)),
              "conflicting_edits":int(np.count_nonzero(bert_changed&alt_changed&(original!=alternative)))}
        evidence["splits"][split]=case
        save_json(P.RESULT/(tag+"-bert.json"),evidence)
        print(split,case["base"]["macro_f1"],case["candidate"]["macro_f1"],case["changed"],
              case["paired_base"]["macro_f1"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("edit",choices=("phase","corpus"))
    main(p.parse_args().edit)
