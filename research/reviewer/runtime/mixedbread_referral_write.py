"""Write a contextual referral head for the 24M Mixedbread encoder."""
from itertools import product
import json
from time import perf_counter

import numpy as np
from safetensors.numpy import save_file
from transformers import AutoModel, AutoTokenizer

import legal_bert_small_referral as target_pool
import sentence_bert_referral_write as bert
from sentence_document_graph_write import OUT, ROOT, file_hash, paired, row_summary, save_json, torch

TAG = "mixedbread-referral"
BASE = ROOT / "references" / "mxbai-embed-xsmall"
CACHE = ROOT / "cache" / TAG
RESULT = ROOT / "results"
FIT_SPLITS = ("fit", "select")
EVAL_SPLITS = ("select", "confirm", "holdout", "test")


def encoder():
    model, info = AutoModel.from_pretrained(BASE, local_files_only=True, add_pooling_layer=False,
                                            output_loading_info=True)
    assert not info["missing_keys"] and not info["mismatched_keys"]
    assert all(k.startswith("pooler.") for k in info["unexpected_keys"])
    pooled = target_pool.Pooled(model.float().eval())

    @torch.inference_mode()
    def run(inputs):
        args = [torch.from_numpy(inputs[name]) for name in ("input_ids", "attention_mask", "target")]
        return (pooled(*args).numpy(),)

    return run


def bank(split, compiled, tokenizer, proof):
    rows, _ = bert.load_views(split)
    path = CACHE / (split + "-" + proof["sha256"]["model.safetensors"][:12] + ".npy")
    old = np.concatenate([r[4] for r in rows])
    if not path.is_file():
        ids = bert.pack_ids(tokenizer, rows, bert.MODES[1])
        assert len(ids) == len(old)
        vec = []
        start = perf_counter()
        for lo in range(0,len(ids),256):
            vec.append(target_pool.infer(compiled, tokenizer, ids[lo:lo+256], 384))
            print(split, min(lo+256,len(ids)), "/", len(ids), round(perf_counter()-start, 1), flush=True)
        stage = path.with_suffix(".part")
        with stage.open("wb") as f:
            np.save(f,np.concatenate(vec),allow_pickle=False)
        stage.replace(path)
    embedding = np.load(path,allow_pickle=False)
    assert len(old) == len(embedding) and embedding.shape[1] == 384
    return rows, (old, np.column_stack((old, embedding)))


def fit(proof, data):
    train, valid = (data[k] for k in FIT_SPLITS)
    assert not set(r[0] for r in train[0]) & set(r[0] for r in valid[0])
    tags = np.concatenate([r[1][r[3]] for r in train[0]])
    base = bert.current_metrics(valid[0])[1]
    output = {"checkpoint": proof, "window": bert.LIMIT, "gradient_steps": 0,
              "fit_cases": [r[0] for r in train[0]], "select_cases": [r[0] for r in valid[0]],
              "fit_rows": len(tags), "selection_parent": base, "heads": {}}
    for i,name in enumerate(("structure only", "Mixedbread target pooled")):
        best = (base["macro_f1"],base["accuracy"])
        best_recipe = None
        payload = None
        for rare in (0., .28, .55):
            for damp, w in bert.fit(train[1][i], tags, rare):
                scores = bert.response(valid[1][i],w)
                for gain,margin in product((.45,.65,.85,1.),(.02,.08,.15)):
                    pred = bert.apply(valid[0],scores,gain,margin)
                    s = row_summary(bert.aggregate(valid[0],pred).sum(0))
                    candidate = (s["macro_f1"],s["accuracy"])
                    if candidate > best and candidate[1] >= base["accuracy"]-.001:
                        best,payload,best_recipe = candidate,w,(rare,damp,gain,margin)
        dest = OUT / (TAG+f"-{i}.safetensors")
        if payload is not None:
            save_file(payload,str(dest))
        output["heads"][name] = {"params": best_recipe, "macro_accuracy": best,
                                  "sha256": file_hash(dest) if payload else None}
    save_json(RESULT/(TAG+"-frozen.json"), output)
    return output


def main():
    torch.set_num_threads(3)
    CACHE.mkdir(exist_ok=True)
    proof = json.loads((BASE/"provenance.json").read_text())
    assert all(file_hash(BASE/k)==v for k,v in proof["sha256"].items())
    compiled = encoder()
    tokenizer = AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    data = {split: bank(split,compiled,tokenizer,proof) for split in FIT_SPLITS}
    config = fit(proof,data)
    freeze = RESULT/(TAG+"-frozen.json")
    out = {"freeze_sha256":file_hash(freeze),"splits":{}}
    for split in EVAL_SPLITS:
        rows,x = data[split] if split in data else bank(split,compiled,tokenizer,proof)
        if split != "select":
            assert not set(config["fit_cases"] + config["select_cases"]) & set(r[0] for r in rows)
        base,summary = bert.current_metrics(rows)
        rec=out["splits"][split]={"cases":len(rows),"referrals":len(x[0]),"baseline":summary,"heads":{}}
        reference=bert.read(bert.TAG)["splits"][split]["arms"]["BERT previous two and target"]["scores"]
        rec["InLegalBERT referral"]=reference
        for i,(name, recipe) in enumerate(config["heads"].items()):
            p=[r[2] for r in rows]
            if recipe["sha256"]:
                path=OUT/(TAG+f"-{i}.safetensors")
                assert file_hash(path)==recipe["sha256"]
                w=bert.load_file(str(path))
                pred=bert.response(x[i],w)
                p=bert.apply(rows,pred,*recipe["params"][-2:])
            cm=bert.aggregate(rows,p)
            labels=row_summary(cm.sum(0))
            if split=="select":
                assert (labels["macro_f1"], labels["accuracy"]) == recipe["macro_accuracy"]
            rec["heads"][name]={"labels":labels,"paired_baseline":paired(cm,base)}
        save_json(RESULT/(TAG+".json"),out)
        print(split, {k:v["labels"]["macro_f1"] for k,v in rec["heads"].items()},
              "InLegalBERT",reference["macro_f1"],flush=True)


if __name__ == "__main__":
    main()
