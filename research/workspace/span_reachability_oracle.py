"""Calculate the best exact labeled-span F1 achievable with a fixed set of candidate cuts."""
import numpy as np

from boundary_span_edit import with_cuts
from fragment_trial import passages
from hier_full import gold_cases
from program_trial import save_json
from seven_paths import LABELS,ROOT
from span_eval import counts,document,summarize


def oracle(doc):
    a=doc["units"]
    cuts={l for l,r in a}|{len(doc["text"])}
    spans=[x for x in doc["gold_spans"] if x["start"] in cuts and x["end"] in cuts]
    # Fill each unresolved gap with any role distinct from both adjacent spans.
    authored=[];start=0
    for s in spans:
        if s["start"]>start:
            forbidden={s["label"]}
            if authored:forbidden.add(authored[-1]["label"])
            authored.append({"start":start,"end":s["start"],"label":next(c for c in LABELS if c not in forbidden)})
        authored.append(s);start=s["end"]
    if start<len(doc["text"]):
        last=authored[-1]["label"] if authored else None
        authored.append({"start":start,"end":len(doc["text"]),"label":next(c for c in LABELS if c!=last)})
    coordinates=[s["start"] for s in authored]
    idx=np.searchsorted(coordinates,[l for l,r in a],side="right")-1
    names=[LABELS.index(s["label"]) for s in authored]
    got=counts(doc,a,np.asarray(names)[idx])
    assert int(got[49])==len(spans) and int(got[50])==len(authored)
    return got

def main():
    cases=[document(k,rows) for k,rows in gold_cases().items()]
    p=passages()
    r={}
    for name,docs in (("whole",cases),("fragment",p["test"]),("matched",p["cloud"])):
        r[name]={}
        for cut in ("current","sentence","all"):
            score=summarize(np.sum([oracle(with_cuts(d,cut)) for d in docs],axis=0))
            r[name][cut]={"cases":len(docs),"exact_span_f1_attainable":score["exact_labeled_spans"]["f1"],"counts":score["exact_labeled_spans"]}
        print(name,{k:v["exact_span_f1_attainable"] for k,v in r[name].items()},flush=True)
    save_json(ROOT/"results"/"span-boundary-oracle.json",{
       "method":"Keep every reference span whose endpoints are candidate cuts. Fill each unresolved interval by one role distinct from adjacent exact spans. This realizes all attainable true positives using a minimum number of extra spans. Adding a reachable span can add at most two predicted spans and therefore cannot lower F1. Gold labels are used only to calculate this diagnostic ceiling.",
       "tasks":r})
if __name__=="__main__":
    main()
