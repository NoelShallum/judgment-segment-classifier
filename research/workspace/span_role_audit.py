"""Audit which seven roles gain or lose from the selected full-judgment transition edit."""
from collections import defaultdict
import numpy as np
from safetensors.numpy import load_file
from hier_full import CACHE,cm,gold_cases,row_summary
from hier_model import CRF,path_labels
from hier_trial import OUT
from program_trial import save_json
from seven_paths import LABELS,ROOT
from span_eval import document
from span_model import merge

def main():
    old=load_file(str(OUT/"document.safetensors"))[CRF]
    new=load_file(str(OUT/"role-specific-whole.safetensors"))[CRF]
    stats={}
    for name,matrix in (("incumbent",old),("role_specific",new)):
        supports=defaultdict(lambda:np.zeros(3,dtype=int));row=np.zeros((7,7),int)
        for case,rows in gold_cases().items():
            d=document(case,rows)
            with np.load(CACHE/(case+".npz")) as f:
                q=path_labels(f["raw"],matrix)
                pred=merge(d["units"],q,LABELS)
                labels=path_labels(f["context"],matrix)
            y=np.array([r[2] for r in rows],dtype=int)
            row+=cm(y,labels)
            for role in LABELS:
                def get(which):return {(s["start"],s["end"]) for s in which if s["label"]==role}
                gold=get(d["gold_spans"]);proposed=get(pred)
                supports[role]+=np.array([len(gold&proposed),len(proposed),len(gold)],int)
        stats[name]={"known_input_rows":row_summary(row),"exact_spans":{}}
        for k,s in supports.items():
            a,b,c=map(int,s);stats[name]["exact_spans"][k]={"exact_matches":a,"predicted":b,"gold":c,
                 "f1":2*a/(b+c),"precision":a/b if b else 0.,"recall":a/c if c else 0.}
    save_json(ROOT/"results"/"transition-per-role-audit.json",stats)
    for k,v in stats.items():
        print(k,v["known_input_rows"]["macro_f1"],[(r,round(m["f1"],4)) for r,m in v["exact_spans"].items()],flush=True)
if __name__=="__main__":main()
