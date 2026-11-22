"""Compare isolated, paragraph-sized, and full text context on the same seven-role targets."""
import itertools
import json
import re

import numpy as np
from safetensors.numpy import load_file,save_file
import torch

from fragment_trial import passages
from hier_model import CRF, HierModel, path_labels
from hier_trial import DONOR,OUT
from program_trial import save_json
from seven_paths import LABELS,ROOT,folder_for
from span_eval import counts,sets,summarize
from span_model import units
from weight_edit import file_hash

CONTEXTS=(1,3,6,12,0)

def context_scores(model, words, offsets, text, width):
    if width==0:
        return model.logits(model.contextualize(words))
    if width==1:
        return model.logits(model.contextualize(words,independent=True))
    ends=set(range(width,len(offsets),width))|{len(offsets)}
    for i,(_,b) in enumerate(offsets,1):
        if re.search(r"\n\s*\n\s*$",text[offsets[i-1][0]:b]):
            ends.add(i)
    starts=[0]+sorted(ends)
    return np.concatenate([model.logits(model.contextualize(words[a:b])) for a,b in zip(starts,starts[1:])])

def local_gold_metrics(rows,pred,saved):
    y=np.array([LABELS.index(r["gold"]) for r in rows]);g=np.array([LABELS.index(saved[r["id"]]["choice"]) for r in rows])
    return {"rows":len(y),"local_correct":int((pred==y).sum()),"gemini_correct":int((g==y).sum()),
            "same_predictions":int((g==pred).sum()),"per_id":dict(zip([r["id"] for r in rows],[LABELS[int(i)] for i in pred]))}

def excerpts(model):
    result={}
    short=HierModel(DONOR,OUT/"excerpt.safetensors")
    for group in ("pilot","confirmation","paragraphs","homogeneous"):
        folder=folder_for(group)
        rows=json.loads((folder/"approved.json").read_text())
        api=json.loads((folder/"answers.json").read_text())
        one=[];parts=[]
        for r in rows:
            text=r["text"];offsets=units(text)
            one.append(LABELS.index(short.classify(text)["label"]))
            if len(offsets)==1:
                parts.append(one[-1]);continue
            words=model.words([text[a:b] for a,b in offsets])
            length=np.array([len(text[a:b].split()) for a,b in offsets])
            h=model.logits(model.contextualize(words))
            parts.append(int(np.argmax(np.average(h,axis=0,weights=length))))
        result[group]={"literal_excerpt":local_gold_metrics(rows,np.asarray(one),api),
                       "contextual_units_vote":local_gold_metrics(rows,np.asarray(parts),api)}
    return result

def main():
    torch.set_num_threads(2)
    m=HierModel(DONOR);result={"single_label_diagnostics":excerpts(m),"spans":{}}
    path=ROOT/"results/context-size-validation.json"
    save_json(path,result)
    for kind,groups in (("whole",sets()),("fragment",passages())):
        state=load_file(str(OUT/(("document" if kind=="whole" else "fragment")+".safetensors")))
        tr=state[CRF] if kind=="whole" else state["transition"]
        data={}
        for split,docs in groups.items():
            data[split]={}
            for d in docs:
                x=d["text"];offs=d["units"]
                words=m.words([x[a:b] for a,b in offs])
                context={w:context_scores(m,words,offs,x,w) for w in CONTEXTS}
                data[split][d["id"]]=context
        def compare(split,width,blend,weights):
            measures=[]
            for d in groups[split]:
                pack=data[split][d["id"]];z=(1-blend)*pack[0]+blend*pack[width]
                labels=path_labels(z,weights)
                measures.append(counts(d,d["units"],labels))
            return np.stack(measures)
        all_scores=[];best=None
        for width,share,edge in itertools.product(CONTEXTS,(.5,1.),(0,.5,1)):
            if width==0 and share<1:continue
            t=tr.copy()
            t[1,3:]*=edge;t[3:,2]*=edge
            q=summarize(compare("select",width,share,t).sum(0));key=q["exact_labeled_spans"]["f1"],q["token_accuracy"]
            all_scores.append([width,share,edge,*key])
            if best is None or key>best[0]:best=(key,width,share,edge,t)
        _,width,share,edge,t=best
        target=OUT/(kind+"-context-size.safetensors")
        save_file({CRF:t,"context_mix":np.array([width,share],np.float32)},str(target))
        stats={}
        for split in groups:
            obs=compare(split,width,share,t);base=compare(split,0,1,tr)
            diffs=[];rng=np.random.default_rng(20261122)
            for ix in rng.integers(len(obs),size=(1200,len(obs))):
                a=summarize(obs[ix].sum(0));b=summarize(base[ix].sum(0))
                diffs.append(a["exact_labeled_spans"]["f1"]-b["exact_labeled_spans"]["f1"])
            stats[split]={"cases":len(obs),"chosen":summarize(obs.sum(0)),"full_context":summarize(base.sum(0)),
                          "context_only":{str(w):summarize(compare(split,w,1,tr).sum(0)) for w in CONTEXTS},
                          "exact_span_delta_ci95":np.quantile(diffs,[.025,.975]).tolist()}
        result["spans"][kind]={"selected_context_units":width,"share":share,"document_edge_multiplier":edge,
             "selection":"Validation exact span F1, then token accuracy. Width zero uses all supplied text. Positive widths use bounded groups of automatically proposed units, with breaks when blank-line paragraph layout exists. Reconstructed test judgments mostly have no original paragraph formatting.",
             "sha256":file_hash(target),"grid":all_scores,"scores":stats}
        print(kind,width,share,stats["test"]["chosen"]["exact_labeled_spans"]["f1"],stats["test"]["full_context"]["exact_labeled_spans"]["f1"],flush=True)
        save_json(path,result)

if __name__=="__main__":
    main()
