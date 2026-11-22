"""Compare a frozen BERT review window that includes a following sentence."""
import argparse
from hashlib import sha256
from itertools import product
import json

import numpy as np
from safetensors.numpy import load_file, save_file

import sentence_bert_reviewer_gain as gain
import sentence_pairwise_repair as repair
import sentence_phase_readout_write as phase
from sentence_document_graph_write import OUT, cm, file_hash, paired, row_summary, save_json


B = gain.B
P = repair.P
TAG = "sentence-bert-future-context"
TARGET_BUDGET = 66
RIGHT_BUDGET = 18
LEFT_BUDGET = 18
PATCH = OUT/(TAG+".safetensors")


def pack(tokenizer,rows):
    output = []
    for _,_,_,q,_,_,texts in rows:
        if not len(q):
            continue
        wanted = sorted({j for i in q for j in (int(i)-1,int(i),int(i)+1) if 0<=j<len(texts)})
        token_rows = tokenizer([texts[j] for j in wanted],add_special_tokens=False,verbose=False)["input_ids"]
        words = dict(zip(wanted,token_rows))
        for i in q:
            target = words[i][-(TARGET_BUDGET if i-1 in words or i+1 in words else B.LIMIT-2):]
            remaining = B.LIMIT-len(target)-2
            context = []
            # Keep the target last, matching the previous-only review convention.
            left = words.get(i-1,())
            right = words.get(i+1,())
            for side,cap,from_end in ((right,RIGHT_BUDGET,False),(left,LEFT_BUDGET,True)):
                use = min(len(side),cap,max(remaining-1,0))
                if use:
                    context[:0] = [*(side[-use:] if from_end else side[:use]),tokenizer.sep_token_id]
                    remaining -= use+1
            output.append([tokenizer.cls_token_id,*context,*target,tokenizer.sep_token_id])
    assert max(map(len,output))<=B.LIMIT
    return output


def encode(split,device):
    from openvino_worker import embed_ids,get_engine

    rows,tokenizer = B.load_views(split)
    ids = pack(tokenizer,rows)
    exported = B.BERT_ART/"source.json"
    config = json.loads(exported.read_text())
    assert file_hash(B.BERT_ART/"encoder.xml")==config["xml_sha256"]
    assert file_hash(B.BERT_ART/"encoder.bin")==config["encoder_bin_sha256"]
    source = sha256(json.dumps([ids,config,device]).encode()).hexdigest()
    folder = P.ROOT/"cache"/TAG/(split+"-"+source[:16])
    folder.mkdir(parents=True,exist_ok=True)
    engine = None
    embeddings = []
    for start in range(0,len(ids),128):
        block = ids[start:start+128]
        file = folder/f"{start:06d}.npy"
        if file.is_file():
            vectors = np.load(file,allow_pickle=False)
        else:
            if engine is None:
                engine,_ = get_engine(device)
            vectors = embed_ids(engine,block,max_length=B.LIMIT)
            part = file.with_suffix(".part")
            with part.open("wb") as f:
                np.save(f,vectors,allow_pickle=False)
            part.replace(file)
        assert vectors.shape==(len(block),768) and np.isfinite(vectors).all()
        embeddings.append(vectors)
        print(split,"Encoded",min(start+128,len(ids)),"/",len(ids),flush=True)
    vectors = np.concatenate(embeddings)
    return rows,vectors,{"input_sha256":source,"vectors_sha256":sha256(vectors.tobytes()).hexdigest(),
                         "windows":len(ids),"device":device}


def view(split,device,context):
    rows,vectors,trace = encode(split,device)
    features = B.design(rows,vectors,"hidden",{})
    parent,w,recipe = context[:3]
    data,_,base,_ = phase.view(split,parent,w,recipe)
    np.testing.assert_array_equal(data[0],[r[0] for r in rows])
    np.testing.assert_array_equal(data[2],np.concatenate([r[1] for r in rows]))
    loc = np.concatenate([start+r[3] for start,r in zip(data[1],rows)])
    prior = np.concatenate([r[5] for r in rows])
    old = np.concatenate([r[2][r[3]] for r in rows])
    old_shape = gain.get_features(split,context)
    assert np.array_equal(old_shape[3],loc)
    assert np.array_equal(old_shape[1],base)
    return data,base,old_shape[2],loc,prior,old,features,trace


def predict(view,w,setting):
    _,base,_,loc,prior,old,feature,_ = view
    gain_factor,gap,veto = setting
    response = B.response(feature,w)
    return gain.overlay.edit(base,loc,prior,response,gain_factor,gap,old if veto else None)


def run(device):
    context = gain.plans()
    rows,x,receipt = encode("fit",device)
    design = B.design(rows,x,"hidden",{})
    labels = np.concatenate([r[1][r[3]] for r in rows])
    fit_cases = [r[0] for r in rows]
    dev = view("select",device,context)
    assert not set(fit_cases) & set(dev[0][0])
    base = row_summary(cm(dev[0][2],dev[2]))
    best = (base["macro_f1"],base["accuracy"])
    choice,weights = None,None
    results = []
    for rarity in (0.,.28,.55):
        for regularization,w in B.fit(design,labels,rarity):
            for option in product((.75,1.),(.10,.16,.22), (True,False)):
                y = predict(dev,w,option)
                q = row_summary(cm(dev[0][2],y))
                grade = (q["macro_f1"],q["accuracy"])
                results.append([rarity,regularization,option,grade])
                if grade>best and q["accuracy"]>=base["accuracy"]-.001:
                    best,choice,weights = grade,(rarity,regularization,option),w
    signature = None
    if weights is not None:
        save_file(weights,str(PATCH))
        restored = load_file(str(PATCH))
        for name,t in weights.items():
            np.testing.assert_array_equal(restored[name],t)
        weights = restored
        signature = file_hash(PATCH)
    frozen = repair.receipt(TAG)
    save_json(frozen,{"frozen_sources":context[5],"training":receipt,"training_cases":fit_cases,
                      "selection_cases":dev[0][0].tolist(),"choice":choice,"grid":results,
                      "training_rows":len(labels),"gradient_steps":0,"weights_sha256":signature})
    result = {"freeze_sha256":file_hash(frozen),"splits":{}}
    for split in ("select","confirm"):
        z = dev if split=="select" else view(split,device,context)
        bank,_,old,loc,*_ = z
        assert not set(bank[0]) & set(fit_cases)
        if split=="confirm":
            assert not set(bank[0]) & set(dev[0][0])
        y = predict(z,weights,choice[2]) if weights is not None else old
        prev,now = (P.dense.paired_confusion(bank,seq) for seq in (old,y))
        result["splits"][split]={"cases":len(bank[0]),"referred":len(loc),"previous":row_summary(prev.sum(0)),
                                 "candidate":row_summary(now.sum(0)),"paired":paired(now,prev),"encoding":z[-1]}
        save_json(P.RESULT/(TAG+".json"),result)
        print(split,choice,result["splits"][split]["candidate"]["macro_f1"],flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",choices=("CPU","GPU"),default="CPU")
    args = parser.parse_args()
    run(args.device)
