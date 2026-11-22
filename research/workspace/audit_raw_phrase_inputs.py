"""Verify the raw phrase transfer scores isolate gold metadata from emissions."""
import json

import numpy as np
from safetensors.numpy import load_file

import sentence_phrase_sequence as sequence
from hier_full import CACHE
from run_entry import prepare_score
from sentence_document_graph_write import file_hash,save_json

P=sequence.phrase
TAG="sentence-phrase-raw-input-qa"


def main():
    docs=P.docs("test")
    parent=P.dense.PARENT
    cfg=load_file(str(parent))
    total=0
    for doc in docs:
        dummy=sequence.raw_document(doc["text"],doc["id"])
        assert dummy["units"]==doc["units"]
        with np.load(CACHE/(doc["id"]+".npz"),allow_pickle=False) as file:
            raw=file["raw"]
        assert len(raw)==len(doc["units"])
        gold=prepare_score(doc,raw,{"exact":cfg})
        scrubbed=prepare_score(dummy,raw,{"exact":cfg})
        np.testing.assert_array_equal(gold.logits["exact"],scrubbed.logits["exact"])
        total+=len(raw)
    assert len(docs)==712
    source=P.RESULT/"sentence-phrase-sequence-raw.json"
    assert total==json.loads(source.read_text())["predicted_sentence_units"]
    save_json(P.RESULT/(TAG+".json"),{"source_raw_benchmark_sha256":file_hash(source),"decoder_sha256":file_hash(parent),
                  "cases":len(docs),"units":total,"exact_emission_parity_after_replacing_gold_metadata":True})
    print("Verified raw scoring-input parity",len(docs),total,flush=True)


if __name__=="__main__":
    main()
