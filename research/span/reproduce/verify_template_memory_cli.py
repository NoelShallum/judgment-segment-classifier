"""Verify the template memory's CPU command on a judgment with an accepted revision."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

import compare_broad_edits_to_bert as broad
import sentence_pairwise_consensus as consensus
import sentence_pairwise_repair as repair
import sentence_phrase_sequence as seq
import sentence_template_memory_write as memory
from hier_model import LABELS
from sentence_document_graph_write import file_hash,save_json

P=seq.phrase


def main():
    _,w,digest=memory.read_weights()
    parent,pair_w,_=consensus.inputs()
    bank,base,u,m=memory.get_view("confirm",parent,pair_w,repair.load_recipe(),w)
    freeze=repair.receipt(memory.TAG)
    plan=json.loads(freeze.read_text())
    changed=memory.propose(base,u,m,*plan["choice"])
    overlay_path=repair.receipt(broad.overlay.TAG)
    overlay=json.loads(overlay_path.read_text())
    cfg=broad.B.read(broad.B.TAG+"-frozen")
    bert=broad.bert_reference("confirm",bank,base,cfg,overlay["arms"]["BERT only"]["choice"])
    combined=np.where(bert!=base,bert,changed)
    examples=sorted((not np.any(bert[a:b]!=base[a:b]),int(b-a),str(k),int(a),int(b))
                    for k,a,b in zip(bank[0],bank[1],bank[1][1:])
                    if np.any(combined[a:b]!=bert[a:b]))
    _,n,k,a,b=examples[0]
    docs={d["id"]:d for d in P.docs("confirm")}
    doc=docs[k]
    texts=[doc["text"][s:e].strip() for s,e in doc["reference_units"]]
    with TemporaryDirectory(dir=P.ROOT/"qa") as directory:
        dest=Path(directory)/"input.json"
        dest.write_text(json.dumps(texts))
        command=[sys.executable,str(P.ROOT/"sentence_template_memory_write.py"),"predict","--input",str(dest)]
        for extra,expect in (([],changed[a:b]),(["--bert-device","CPU"],combined[a:b])):
            result=json.loads(subprocess.check_output(command+extra))
            assert result["labels"]==[LABELS[r] for r in expect]
            edges=seq.runs(np.array([0,n]),expect)
            assert result["spans"]==[{"start_sentence":int(s),"end_sentence":int(t),"label":LABELS[r]}
                                     for s,t,r in zip(*edges)]
    proof={"case":k,"sentences":n,"changed":int(np.count_nonzero(changed[a:b]!=base[a:b])),
           "bert_revisions":int(np.count_nonzero(bert[a:b]!=base[a:b])),
           "template_revisions_after_bert":int(np.count_nonzero(combined[a:b]!=bert[a:b])),
           "memory_source_sha256":digest,"freeze_sha256":file_hash(freeze),
           "local_overlay_freeze_sha256":file_hash(overlay_path),"labels_and_spans_exact_parity":True,
           "composed_bert_labels_and_spans_exact_parity":True}
    save_json(P.RESULT/"sentence-template-memory-cli-qa.json",proof)
    print(proof,flush=True)


if __name__=="__main__":
    main()
