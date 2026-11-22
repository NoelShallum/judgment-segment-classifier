"""Label a judgment with the frozen sequence model and confidence-routed review."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

import numpy as np

import predict_pairwise_judgment as pairwise
import sentence_bert_referral_write as bert
import sentence_phrase_sequence as sequence
import sentence_pairwise_repair as repair
import sentence_referral_utility_write as utility
import mixedbread_referral_overlay as overlay
from sentence_document_graph_write import OUT, file_hash, load_file


def predictor(device="CPU",rate=.20):
    from openvino_worker import embed_ids,get_engine

    approve = repair.receipt(overlay.TAG)
    config = json.loads(approve.read_text())
    rule = config["arms"]["BERT only"]["choice"]
    frozen = repair.receipt(bert.TAG)
    plan = json.loads(frozen.read_text())
    assert config["sources"][bert.TAG] == file_hash(frozen)
    assert config["sources"][repair.TAG] == file_hash(repair.receipt(repair.TAG))
    assert plan["document_weights_sha256"] == file_hash(bert.PARENT)
    assert plan["backbone_weights_sha256"] == file_hash(bert.ORIGINAL)
    weight = OUT/(bert.TAG+"-2.safetensors")
    assert plan["checkpoints"][overlay.BERT] == file_hash(weight)
    w = load_file(str(weight))
    mode = plan["frozen_modes"][overlay.BERT][0]
    parent = pairwise.predictor("direct")
    engine = tokenizer = None

    def predict(text,raw=False):
        if raw and (not isinstance(text,str) or not text.strip()):
            raise ValueError("Supply nonblank judgment text")
        route = {}

        def revise(bank,lex,unary,initial,item,state,model,cfg,vocab):
            nonlocal engine,tokenizer
            base = parent(bank,lex,unary,initial) if parent is not None else initial
            # Match audit_pairwise_referral_coverage.routing's dtype and tie ordering.
            confidence = np.exp(unary-unary.max(1,keepdims=True))
            confidence /= confidence.sum(1,keepdims=True)
            risk = 1.-confidence[np.arange(len(base)),base]
            positions = utility.route(bank[1],risk,rate)
            row = bert.case_view(item,state,model,cfg,vocab,positions)
            np.testing.assert_array_equal(row[2],bank[3])
            if engine is None:
                with redirect_stdout(sys.stderr):
                    engine,tokenizer = get_engine(device)
            windows = bert.pack_ids(tokenizer,[row],bert.MODES[1])
            embeddings = embed_ids(engine,windows,max_length=bert.LIMIT)
            post = bert.response(bert.design([row],embeddings,mode,w),w)
            g,t,k = rule
            edited = overlay.edit(base,positions,row[5],post,g,t,row[2][positions] if k else None)
            route.update(referrals=positions.tolist(),reviewed=len(positions),changed=int(np.count_nonzero(edited!=base)))
            return edited

        result = sequence.run_input(text,raw,"spans",revise_with_context=revise)
        return {**result,**route}

    return predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--raw",action="store_true")
    parser.add_argument("--device",choices=("CPU","GPU"),default="CPU")
    parser.add_argument("--rate",type=float,choices=(.15,.20),default=.20)
    args = parser.parse_args()
    data = args.input.read_text()
    result = predictor(args.device,args.rate)(data if args.raw else json.loads(data),args.raw)
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":
    main()
