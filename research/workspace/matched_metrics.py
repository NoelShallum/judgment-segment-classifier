"""Create paired, document-clustered, candidate-vs-Gemini results without releasing any court text."""
import argparse
from collections import Counter
import hashlib
import json
import numpy as np
from seven_paths import FOLDERS,folder_for

from program_trial import save_json
from weight_edit import LABELS,ROOT

def from_matrices(cm):
    n=cm.shape[-2]
    tp=np.diagonal(cm[...,:n],axis1=-2,axis2=-1)
    div=cm.sum(-1)+cm[...,:n].sum(-2)
    f1=np.divide(2*tp,div,out=np.zeros_like(tp,dtype=float),where=div>0)
    return tp.sum(-1)/cm.sum((-1,-2)),f1.mean(-1),f1

def group_metrics(rows,local,provider):
    labels=list(LABELS)
    n=len(labels)
    names=dict.fromkeys(r["doc"] for r in rows);docs={d:i for i,d in enumerate(names)}
    predset={**{k:[data["predictions"][r["id"]] for r in rows] for k,data in local.items()},
            "gemini":[provider[r["id"]]["choice"] for r in rows]}
    truth=np.array([labels.index(r["gold"]) for r in rows])
    di=np.array([docs[r["doc"]] for r in rows])
    raw={k:np.array(v) for k,v in predset.items()}
    pred={k:np.array([labels.index(u) for u in p]) for k,p in predset.items()}
    by_doc={k:np.bincount(di*n*(n+1)+truth*(n+1)+y,minlength=len(docs)*n*(n+1)).reshape(len(docs),n,n+1)
            for k,y in pred.items()}
    rng=np.random.default_rng(268171)
    sizes=rng.multinomial(len(docs),np.ones(len(docs))/len(docs),size=4000)
    allscores={};boots={};paired={}
    for k,cm in by_doc.items():
        total=cm.sum(0);acc,macro,fs=from_matrices(total)
        allscores[k]={"correct":int(np.sum(pred[k]==truth)),"accuracy":float(acc),"macro_f1":float(macro),
            "per_role_f1":dict(zip(labels,fs.tolist())),"confusion":total.tolist(),"outside_taxonomy":int(total[:,-1].sum())}
        sample=(sizes@cm.reshape(len(docs),-1)).reshape(len(sizes),n,n+1)
        boots[k]=from_matrices(sample)[:2]
    for k in local:
        a=pred[k]==truth;b=pred["gemini"]==truth
        paired[k]={"projected_agreement_count":int(np.count_nonzero(pred[k]==pred["gemini"])),
            "seven_label_agreement_count":int(np.count_nonzero(raw[k]==raw["gemini"])),
            "both_correct":int(np.count_nonzero(a&b)),"only_local_correct":int(np.count_nonzero(a&~b)),
            "only_gemini_correct":int(np.count_nonzero(~a&b)),"both_wrong":int(np.count_nonzero(~a&~b)),
            "accuracy_gap":allscores[k]["accuracy"]-allscores["gemini"]["accuracy"],
            "accuracy_gap_95pct":np.quantile(boots[k][0]-boots["gemini"][0],[.025,.975]).tolist(),
            "macro_f1_gap":allscores[k]["macro_f1"]-allscores["gemini"]["macro_f1"],
            "macro_f1_gap_95pct":np.quantile(boots[k][1]-boots["gemini"][1],[.025,.975]).tolist()}
    return {"rows":len(rows),"cases":len(docs),"labels":labels,"confusion_columns":labels+["outside"],
        "always_none_correct":sum(r["gold"]=="none" for r in rows),"gold_support":dict(Counter(r["gold"] for r in rows)),"metrics":allscores,"paired_with_gemini":paired}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--corpus",choices=tuple(FOLDERS),default="pilot");args=ap.parse_args()
    folder=folder_for(args.corpus)
    rows=json.loads((folder/"approved.json").read_text())
    provider=json.loads((folder/"answers.json").read_text());local=json.loads((folder/"local.json").read_text())
    # An independently frozen alternative can be scored on the same unchanged pilot.
    for extra in ("instruction-projection.json","instruction-semantic.json","graft.json","cues.json","hier.json","fusion.json","memory.json","mlp.json"):
        opt=folder/extra
        if opt.is_file():local.update(json.loads(opt.read_text()))
    schema=json.loads((folder/"request-schema.json").read_text())
    for r in rows:
        assert provider[r["id"]]["input_sha256"]==r["sha256"]==hashlib.sha256(r["text"].encode()).hexdigest()
        assert provider[r["id"]]["question_sha256"]==schema["sha256"]
    output={"sources":{"sentence":"https://huggingface.co/datasets/L-NLProc/LegalSeg_CSV","paragraph":"https://huggingface.co/datasets/L-NLProc/LegalSeg_CSV"},
        "selection":"LegalSeg only. Seven-way source labels; 16 screened sentences per role in the initial balanced pilot. Teachers, local outputs and source labels were withheld from the privacy review of the independent confirmation.",
        "inference":"One seven-way choice on identical UTF-8 text. No surrounding judgment, gold, source, document ID or party metadata supplied. Gemini uses a fixed role rubric. Local BERT takes raw text only.",
        "uncertainty":"4,000 paired resamples of whole judgments within each set. Percentile 95% intervals; exact seven-label agreement.",
        "candidates":{k:v["sha256"] for k,v in local.items()},"gemini_request":schema,
        "source_date_utc":sorted(x["timestamp"] for x in provider.values())[0][:10],
        "results":{unit:group_metrics([r for r in rows if r["source"]==unit],local,provider) for unit in dict.fromkeys(r["source"] for r in rows)},
        "rows":[{"id":r["id"],"doc":r["doc"],"gold":r["gold"],"text_sha256":r["sha256"],"transformation":r["transformation"],
              "gemini":provider[r["id"]]["choice"],**{k:v["predictions"][r["id"]] for k,v in local.items()}} for r in rows]}
    if args.corpus=="confirmation":
        output["selection"]="Sample 196 distinct eligible judgments from the official test split, choosing one eligible sentence per case without a role quota. The 98 pilot test judgments were held out. The random draw used seed 118220971, followed by privacy review without scores or labels. This is an equal-case screened population, not an estimate for every test row."
        output.pop("paragraph_attribution",None)
    if args.corpus=="paragraphs":
        output["selection"]="Twenty-one multi-statement source rows from the original seven-role test split, three per role. Each source row carries one existing human role. Whitespace, monetary amounts and dates are normalized before both predictions. The CSV does not specify original typeset paragraph boundaries."
    if args.corpus=="homogeneous":
        output["selection"]="Adjacent original LegalSeg sentences from the same judgment combined into a paragraph-length input. All constituent seven-role gold labels agree. Their shared role is a derived paragraph reference."
    result="gemini-seven-"+args.corpus+".json"
    save_json(ROOT/"results"/result,output)
    print("Verified matched results",[(u,v["rows"],v["metrics"]["gemini"]["correct"],v["metrics"]["associative_context"]["correct"]) for u,v in output["results"].items()])
if __name__=="__main__":main()
