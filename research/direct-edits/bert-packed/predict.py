"""CPU prediction with a frozen packed whole-judgment readout."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from safetensors.numpy import load_file
import torch
from transformers import BertTokenizerFast

from encode import HERE, TopTwo, case_windows
from fit import EXTRA, auxiliary, loader, values
from native import donor_at, read, sha


def predict(texts, checkpoint, tokenizer, readout, mode, threads):
    if not texts or any(not isinstance(x,str) or not x.strip() for x in texts):
        raise ValueError("Supply one JSON string or an array of nonempty strings.")
    torch.set_num_threads(threads)
    frozen = read(readout/"freeze.json")
    assert frozen["serialized_weights_roundtrip"]
    cfg = read(HERE/"protocol.json")
    assert frozen["plan_sha256"] == sha(HERE/"protocol.json")
    file = readout/"analytic-heads.safetensors"
    assert frozen["weights_sha256"] == sha(file)
    params = load_file(str(file))
    model = TopTwo(donor_at(HERE,checkpoint,"cpu"),None)
    token = BertTokenizerFast.from_pretrained(tokenizer,local_files_only=True)
    arrays = token(texts,add_special_tokens=False,truncation=False,
                   return_attention_mask=False,return_token_type_ids=False,verbose=False)["input_ids"]
    arrays = [x or [token.unk_token_id] for x in arrays]
    ends = np.r_[0,np.cumsum([len(x) for x in arrays])]
    ids = np.concatenate(arrays)
    special = token.cls_token_id,token.sep_token_id,token.pad_token_id
    t = time.perf_counter()
    w = case_windows(ids,ends,special,"prediction","ordered")
    context,n,_,charged = model.run(w,12,"cpu",token.pad_token_id,len(texts))
    np.testing.assert_array_equal(n,np.diff(ends))
    x = dict(tokens=ids,ends=ends,ordered=context)
    aux = auxiliary([x],params["lexical.counts"],False)
    rule = frozen["settings"]["ordered"]["selected"][mode]
    hidden = values([x],"ordered",rule["fraction_last"],aux)
    key = "ordered."+mode
    logits = hidden[:,EXTRA[mode]]@params[key+".weight"]+params[key+".bias"]
    tags = loader()(logits,params["crf.transitions"],[0,len(texts)],rule["transition"])
    labels = [cfg["roles"][i] for i in tags]
    begin = np.flatnonzero(np.r_[True,tags[1:]!=tags[:-1]])
    spans = [{"start_sentence_inclusive":int(a),"end_sentence_exclusive":int(b),"role":labels[a]}
             for a,b in zip(begin,np.r_[begin[1:],len(tags)])]
    return dict(labels=labels,spans=spans,wordpieces=len(ids),windows=len(w),padded_positions=charged,
                inference_seconds=time.perf_counter()-t,readout=mode)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("document",type=Path,help="JSON string for one unit or array of ordered sentence strings")
    cli.add_argument("checkpoint",type=Path)
    cli.add_argument("tokenizer",type=Path)
    cli.add_argument("frozen_readout",type=Path)
    cli.add_argument("--mode",choices=list(EXTRA),default="native")
    cli.add_argument("--threads",type=int,default=2)
    args = cli.parse_args()
    raw = json.loads(args.document.read_text())
    texts = [raw] if isinstance(raw,str) else raw
    if not isinstance(texts,list):
        raise ValueError("Supply one JSON string or an array of nonempty strings.")
    print(json.dumps(predict(texts,args.checkpoint,args.tokenizer,args.frozen_readout,args.mode,args.threads),indent=2))


if __name__=="__main__":
    main()
