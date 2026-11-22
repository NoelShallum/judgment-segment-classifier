"""Audit the realized Mixedbread referral window without running the encoder."""
import json

import numpy as np
from transformers import AutoTokenizer

import mixedbread_referral_write as m
from sentence_document_graph_write import file_hash, save_json

TAG = "mixedbread-referral-context"


def main():
    token = AutoTokenizer.from_pretrained(m.BASE,local_files_only=True)
    provenance = json.loads((m.BASE / "provenance.json").read_text())
    assert all(file_hash(m.BASE/k)==provenance["sha256"][k] for k in ("tokenizer.json","vocab.txt","config.json"))
    freeze=m.RESULT/(m.TAG+"-frozen.json")
    score={"mixedbread_freeze_sha256":file_hash(freeze),"window":m.bert.LIMIT,"cohorts":{}}
    for split in ("select","confirm","holdout","test"):
        rows,_=m.bert.load_views(split)
        packs=m.bert.pack_ids(token,rows,m.bert.MODES[1])
        texts=[r[6][i] for r in rows for i in r[3]]
        arr=np.array([len(p) for p in packs])
        counts=np.array([p[1:-1].count(token.sep_token_id) for p in packs])
        targets=np.array([len(x) for x in token(texts,add_special_tokens=False,verbose=False)["input_ids"]])
        assert len(arr)==len(targets) and np.all(arr<=m.bert.LIMIT)
        clipped=targets>m.bert.LIMIT-4
        record={"judgments":len(rows),"units":len(arr),"minimum_packed_tokens":int(arr.min()),
                "median_packed_tokens":float(np.median(arr)),"maximum_packed_tokens":int(arr.max()),
                "full_windows":int(np.count_nonzero(arr==m.bert.LIMIT)),
                "target_truncated":int(clipped.sum()),
                "candidate_positions_without_separator":int(np.count_nonzero(counts==0))}
        score["cohorts"][split]=record
        print(split,record,flush=True)
    save_json(m.RESULT/(TAG+".json"),score)


if __name__=="__main__":
    main()
