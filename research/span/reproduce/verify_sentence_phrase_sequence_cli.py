"""Check the exact supplied-sentence and raw-text CPU sequence interfaces."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
from safetensors.numpy import load_file

import sentence_phrase_sequence as seq
import sentence_phrase_sequence_raw as raw
from hier_model import LABELS
from sentence_document_graph_write import OUT,file_hash,save_json
from span_eval import counts

P=seq.phrase
PROFILE="spans"


def command(path,*flags):
    args=[sys.executable,str(P.ROOT/"sentence_phrase_sequence.py"),"predict",
          "--input",str(path),"--profile",PROFILE,*flags]
    return json.loads(subprocess.check_output(args,text=True))


def main():
    loc=P.RESULT/(seq.TAG+"-frozen.json")
    plan=json.loads(loc.read_text())
    mode,g,b,guard=plan["rules"][PROFILE]
    weights=load_file(str(OUT/(seq.TAG+".safetensors")))
    bank=P.dense.prepare("confirm")
    lex=P.cached_scores("confirm")
    base,u=seq.source_unary(bank,lex)
    edge=seq.edge_arrays(bank,lex,mode,weights)
    y=seq.decode(seq.load_decoder(),bank[1],u,base,edge,g,b,guard)
    candidates=sorted((e-s,k,s,e) for k,s,e in zip(bank[0],bank[1],bank[1][1:]) if np.any(y[s:e]!=base[s:e]))
    n,k,a,e=candidates[0]
    docs={d["id"]:d for d in P.docs("confirm")}
    unit=docs[k]
    text=[unit["text"][a:b].strip() for a,b in unit["reference_units"]]
    raw_docs=sorted(P.docs("test"),key=lambda d:len(d["units"]))
    raw_doc=raw_docs[len(raw_docs)//3]
    offsets=raw_doc["units"]
    expected=raw.work([raw_doc])[0][2][PROFILE]
    with TemporaryDirectory(dir=P.ROOT/"qa") as directory:
        place=Path(directory)
        json_path=place/"sentences.json"
        json_path.write_text(json.dumps(text))
        output=command(json_path)
        assert output["labels"]==[LABELS[i] for i in y[a:e]]
        s,t,v=seq.runs(np.array([0,len(text)]),y[a:e])
        assert output["spans"]==[{"start_sentence":int(i),"end_sentence":int(j),"label":LABELS[k]} for i,j,k in zip(s,t,v)]
        src=place/"judgment.txt"
        src.write_text(raw_doc["text"])
        output=command(src,"--raw")["spans"]
    assert output[0]["start"]==0 and output[-1]["end"]==len(raw_doc["text"])
    assert all(x["end"]==y["start"] for x,y in zip(output,output[1:]))
    j=np.searchsorted([o["start"] for o in output],[i for i,_ in offsets],side="right")-1
    yraw=np.asarray([LABELS.index(output[i]["label"]) for i in j])
    assert np.array_equal(counts(raw_doc,offsets,yraw),expected)
    proof={"sequence_freeze_sha256":file_hash(loc),"profile":PROFILE,
           "supplied":{"case":k.item() if hasattr(k,"item") else k,"sentences":int(n),"label_revisions":int(np.count_nonzero(y[a:e]!=base[a:e]))},
           "raw":{"case":raw_doc["id"],"units":len(offsets),"spans":len(output),"text_characters":len(raw_doc["text"])}}
    save_json(P.RESULT/(seq.TAG+"-cli-qa.json"),proof)
    print(proof,flush=True)


if __name__=="__main__":
    main()
