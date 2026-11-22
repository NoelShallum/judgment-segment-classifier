"""Score the official per-judgment hierarchy archives with supplied sentence cuts."""
import numpy as np
from safetensors.numpy import load_file
from hier_full import CACHE,gold_cases
from hier_model import CRF,PUNCTUATION,path_labels
from hier_trial import OUT
from program_trial import TO_ID,save_json
from seven_paths import ROOT
from span_eval import counts,document,summarize

def main():
    f=ROOT/"references/baseline_document/predictions.txt"
    predictions={}
    for line in f.read_text().splitlines():
        key,seq=line.split("\t",1)
        predictions[str(int(key.removeprefix("file_"))-1)]=[TO_ID[x] for x in seq.rstrip(",").split(",")]
    modes={"baseline_oracle":load_file(str(OUT/"document.safetensors"))[CRF],
           "role_specific_oracle":load_file(str(OUT/"role-specific-whole.safetensors"))[CRF],
           "balanced_oracle":load_file(str(OUT/"role-specific-balanced.safetensors"))[CRF]}
    raw={"named_hierarchy":[]}
    raw.update({k:[] for k in modes})
    missing=0
    for key,rows in gold_cases().items():
        raw_index=[j for j,(_,t,_) in enumerate(rows) if t.strip()]
        author=[j for j,(_,t,_) in enumerate(rows) if "\t" not in t and t.lower().translate(PUNCTUATION).split()]
        y=np.asarray(predictions[key])
        assert len(author)==len(y)
        missing+=len(raw_index)-len(author)
        projected=y[np.maximum(0,np.searchsorted(author,raw_index,side="right")-1)]
        d=document(key,rows)
        raw["named_hierarchy"].append(counts(d,d["reference_units"],projected))
        with np.load(CACHE/(key+".npz")) as f:
            for k,m in modes.items():
                pred=path_labels(f["context"],m)
                raw[k].append(counts(d,d["reference_units"],pred[raw_index]))
    scores={k:summarize(np.stack(v).sum(0)) for k,v in raw.items()}
    save_json(ROOT/"results"/"named-hierarchy-span-oracle.json",
              {"cases":len(predictions),"missing_nonblank_rows_whose_labels_inherited":missing,"scores":scores,
               "protocol":"All methods get original sentence boundaries. The exact role spans are formed by merging equal adjacent role labels. Author-file alignment is per case. For nonblank fragments dropped by the author's punctuation or tab filter, carry the preceding predicted role and use the first known prediction at document start. No gold roles are consulted in this fill."})
    print([(k,v["exact_labeled_spans"]["f1"],v["token_accuracy"]) for k,v in scores.items()],flush=True)
if __name__=="__main__":
    main()
