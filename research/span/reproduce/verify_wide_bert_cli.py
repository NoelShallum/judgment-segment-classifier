"""Compare CPU prediction with the frozen FP32 confirmation labels."""
from time import monotonic

import numpy as np

import predict_wide_bert_review as runtime
import sentence_bert_reviewer_gain as gain
import sentence_phrase_sequence as sequence
import sentence_wide_bert_multiclass_write as study
from hier_model import LABELS
from sentence_document_graph_write import save_json


def main():
    context = gain.plans()
    result = study.read_wide()
    data = study.inputs("confirm",context,result)
    bank = data.bank
    documents = study.P.docs("confirm")
    np.testing.assert_array_equal(bank[0],[d["id"] for d in documents])
    subset = []
    for i,(lo,hi) in enumerate(zip(bank[1],bank[1][1:])):
        changed = int(np.count_nonzero(data.base[lo:hi]!=data.incumbent[lo:hi]))
        if changed and 10 <= hi-lo <= 220:
            subset.append((changed,i,lo,hi))
    subset.sort(reverse=True)
    assert len(subset)>=3
    predict = runtime.predictor("CPU",.20)
    evidence = {"method":result["method"],"cohort":"confirm","cases":[]}
    for changed,i,lo,hi in subset[:3]:
        doc = documents[i]
        sentences = [doc["text"][a:b] for a,b in doc["reference_units"]]
        start = monotonic()
        answer = predict(sentences)
        elapsed = monotonic()-start
        expected = data.incumbent[lo:hi]
        positions = data.positions[(data.positions>=lo)&(data.positions<hi)]-lo
        assert len(sentences)==hi-lo
        assert answer["labels"]==[LABELS[x] for x in expected]
        assert answer["referrals"]==positions.tolist()
        assert answer["changed"]==changed
        begins,ends,roles = sequence.runs([0,len(expected)],expected)
        assert answer["spans"]==[{"start_sentence":int(a),"end_sentence":int(b),"label":LABELS[y]}
                                 for a,b,y in zip(begins,ends,roles)]
        evidence["cases"].append({"id":doc["id"],"sentences":len(sentences),"reviewed":len(positions),
                                   "changed":changed,"elapsed_seconds":elapsed})
        print("Matched",doc["id"],len(sentences),len(positions),changed,round(elapsed,2),flush=True)
    raw = "Reasons for the judgment\n\nThe petitioner filed an appeal. The notice was served. We allow the appeal."
    ans = predict(raw,raw=True)
    ends = sequence.raw_document(raw)["units"]
    assert ans["reviewed"] == int(np.ceil(.20*len(ends)))
    for s in ans["spans"]:
        assert 0<=s["start"]<s["end"]<=len(raw) and s["label"] in LABELS
    assert all(a["end"]<=b["start"] for a,b in zip(ans["spans"],ans["spans"][1:]))
    evidence["raw_smoke_test"]={"units":len(ends),"spans":len(ans["spans"]),"reviewed":ans["reviewed"]}
    save_json(study.P.RESULT/"sentence-wide-bert-cli-qa.json",evidence)


if __name__=="__main__":
    main()
