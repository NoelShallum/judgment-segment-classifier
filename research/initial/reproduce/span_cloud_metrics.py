"""Pair the local span decoder and Gemini on identical seven-role LegalSeg source passages."""
import argparse,hashlib,json
import numpy as np
from seven_paths import ROOT,LABELS
from span_eval import document,counts,summarize
from program_trial import save_json
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',choices=['local','hier','fusion','fragment'],default='local');args=ap.parse_args()
    s=ROOT/"seven-spans";blocks=json.loads((s/"cloud.json").read_text());local={a["doc"]:a for a in json.loads((s/(args.profile+"-cloud.json")).read_text())}
    provider=json.loads((s/"gemini-cloud.json").read_text());aa=[];bb=[];records=[];agreement=slots=0
    for block in blocks:
        key=block["doc"];doc=document(key,[(r["row"],r["text"],LABELS.index(r["gold"])) for r in block["rows"]])
        g=provider[key];l=local[key];h=hashlib.sha256(doc["text"].encode()).hexdigest();assert h==g["raw_sha256"]
        assert [list(x) for x in doc["units"]]==g["unit_offsets"] and len(doc["units"])==len(l["unit_labels"])
        ac=counts(doc,doc["units"],[LABELS.index(k) for k in l["unit_labels"]]);bc=counts(doc,doc["units"],[LABELS.index(k) for k in g["unit_labels"]])
        aa.append(ac);bb.append(bc);agreement+=sum(a==b for a,b in zip(l["unit_labels"],g["unit_labels"]));slots+=len(doc["units"])
        records.append({"case":key,"raw_sha256":h,"source_rows":[r["row"] for r in block["rows"]],"units":g["unit_offsets"],"gold_spans":doc["gold_spans"],
                        "local":l["unit_labels"],"gemini":g["unit_labels"],"provider_timestamp":g["timestamp"]})
    aa=np.stack(aa);bb=np.stack(bb);rng=np.random.default_rng(50811);deltas=[]
    for _ in range(4000):
        ix=rng.integers(len(aa),size=len(aa));a,b=summarize(aa[ix].sum(0)),summarize(bb[ix].sum(0))
        deltas.append([a["token_accuracy"]-b["token_accuracy"],a["exact_labeled_spans"]["f1"]-b["exact_labeled_spans"]["f1"],a["exact_role_boundaries"]["f1"]-b["exact_role_boundaries"]["f1"]])
    result={"cases":len(blocks),"source_rows":sum(len(x["rows"]) for x in blocks),"proposed_units":slots,"agreement_units":agreement,
            "local":summarize(aa.sum(0)),"gemini":summarize(bb.sum(0)),"ci_order":["token_accuracy","exact_labeled_spans_f1","exact_role_boundaries_f1"],
            "paired_document_gap_95pct":np.quantile(deltas,[.025,.975],axis=0).T.tolist(),"records":records,
            "protocol":"Same unmarked shared input, same gold-blind candidate units, seven role choices. Gemini receives one question per candidate unit with its location and excerpt; the local decoder uses the same units. Merge adjoining equal predictions and score half-open Unicode character ranges. Every reference boundary comes from contiguous official LegalSeg source rows."}
    save_json(ROOT/"results"/("gemini-seven-"+("" if args.profile=="local" else args.profile+"-")+"spans.json"),result)
    print('Paired labeled-span F1 local / Gemini',result["local"]["exact_labeled_spans"]["f1"],result["gemini"]["exact_labeled_spans"]["f1"])
if __name__=="__main__":main()
