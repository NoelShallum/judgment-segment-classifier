"""Replay frozen public seven-role excerpt inputs against the saved Gemini decisions."""
import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(HERE/"priors"))
import experiment as prior
from predict_document import Predictor
import sentence_bert_referral_write as B
import sentence_phrase_sequence as seq
import sentence_pairwise_repair as rep
import sentence_referral_utility_write as util
import predict_pairwise_judgment as pair
import mixedbread_referral_overlay as overlay
from safetensors.numpy import load_file
from openvino_worker import get_engine,embed_ids


PANELS={"balanced isolated sentences":"gemini-seven-pilot","independent singleton case sample":"gemini-confirmation",
        "single human-labeled multi-statement rows":"gemini-seven-paragraphs","joined same-role paragraphs":"gemini-seven-joined"}
SPECIALIST={"gemini-seven-pilot":"pilot","gemini-confirmation":"confirmation","gemini-seven-paragraphs":"paragraphs","gemini-seven-joined":"homogeneous"}


def source(folder):
    where=ROOT/folder
    approved=json.loads((where/"approved.json").read_text())
    answers=json.loads((where/"answers.json").read_text())
    archive_path=ROOT/"results"/("gemini-seven-"+SPECIALIST[folder]+".json")
    archive=json.loads(archive_path.read_text())
    archived={r["id"]:r for r in archive["rows"]}
    assert len(approved)==len(answers)==len(archived)
    for row in approved:
        answer=answers[row["id"]]
        assert answer["model"]=="gemini-decision-1" and hashlib.sha256(row["text"].encode()).hexdigest()==row["sha256"]==answer["input_sha256"]
        row["gemini"]=prior.LABELS.index(answer["choice"])
        earlier=archived[row["id"]]
        assert earlier["text_sha256"]==row["sha256"] and earlier["doc"]==row["doc"] and earlier["gold"]==row["gold"] and earlier["gemini"]==answer["choice"]
        row["specialist"]=prior.LABELS.index(earlier["internal_mlp"])
    return approved,{"approved_sha256":prior.file_hash(where/"approved.json"),"answers_sha256":prior.file_hash(where/"answers.json"),
                     "archived_specialist_predictions_sha256":prior.file_hash(archive_path),"specialist_checkpoint_sha256":archive["candidates"]["internal_mlp"]}


def routed(sentences):
    # Reuse the immutable public assets for every call as a long-running service would.
    seq.phrase.load_predictor=lru_cache(maxsize=1)(seq.phrase.load_predictor)
    score_pairs=pair.predictor("direct")
    collected=[]
    def capture(bank,lex,unary,initial,item,state,model,cfg,vocab):
        base=score_pairs(bank,lex,unary,initial)
        confidence=np.exp(unary-unary.max(1,keepdims=True));confidence/=confidence.sum(1,keepdims=True)
        pos=util.route(bank[1],1.-confidence[np.arange(len(base)),base],.2)
        assert pos.tolist()==[0]
        row=B.case_view(item,state,model,cfg,vocab,pos)
        np.testing.assert_array_equal(row[2],bank[3])
        collected.append((row,base))
        return base
    for t in sentences:
        seq.run_input([t],False,"spans",revise_with_context=capture)
    rows=[x for x,b in collected]
    engine,tok=get_engine("CPU")
    ids=B.pack_ids(tok,rows,B.MODES[1])
    vec=embed_ids(engine,ids,max_length=B.LIMIT)
    config=B.read(B.TAG+"-frozen")
    name=overlay.BERT
    weights=B.OUT/"sentence-bert-referral-2.safetensors"
    assert prior.file_hash(weights)==config["checkpoints"][name]
    w=load_file(str(weights))
    post=B.response(B.design(rows,vec,config["frozen_modes"][name][0],w),w)
    g,margin,veto=B.read(overlay.TAG+"-frozen")["arms"]["BERT only"]["choice"]
    pred=[]
    for i,(row,base) in enumerate(collected):
        p=overlay.edit(base,np.asarray([0]),row[5],post[i:i+1],g,margin,row[2] if veto else None)
        pred.append(int(p[0]))
    # The normal unmodified entry point and the batched path must agree on actual inputs.
    import predict_wide_bert_review as runtime
    native=runtime.predictor("CPU",.20)
    for j in (0,len(sentences)-1):
        got=native([sentences[j]])["labels"]
        assert got==[prior.LABELS[pred[j]]]
    return np.asarray(pred)


def bootstrap(gold,a,b,cases):
    ids=list(dict.fromkeys(cases))
    lookup={k:i for i,k in enumerate(ids)}
    ix=np.asarray([lookup[k] for k in cases])
    ac=np.bincount(49*ix+7*gold+a,minlength=49*len(ids)).reshape(-1,7,7)
    bc=np.bincount(49*ix+7*gold+b,minlength=49*len(ids)).reshape(-1,7,7)
    draw=np.random.default_rng(18152).integers(len(ids),size=(2000,len(ids)))
    def stats(c):
        m=c[draw].sum(1)
        acc=np.diagonal(m,axis1=1,axis2=2).sum(1)/m.sum((1,2))
        return acc
    return np.quantile(stats(ac)-stats(bc),[.025,.975]).tolist()


def main():
    torch.set_num_threads(1)
    panels={k:source(n) for k,n in PANELS.items()}
    texts=list(dict.fromkeys(row["text"] for rows,_ in panels.values() for row in rows))
    truth={t:i for i,t in enumerate(texts)}
    where=HERE/"gemini"
    where.mkdir(exist_ok=True)
    frozen=where/"frozen-excerpt-predictions.npz"
    signature=hashlib.sha256(json.dumps([texts,B.read(B.TAG+"-frozen")["backbone_weights_sha256"],prior.file_hash(HERE/"priors/priors.safetensors"),prior.file_hash(HERE/"hierarchy/uniform-retention-plus-head.safetensors")]).encode()).hexdigest()
    if frozen.exists():
        with np.load(frozen,allow_pickle=False) as f:
            assert f["identity"].item()==signature
            allpred={k:f[k] for k in f.files if k!="identity"}
    else:
        allpred={}
        for mode in ("source","document-prior","retention","retention-prior"):
            obj=Predictor(HERE/"portable-smoke",mode)
            allpred[mode]=np.asarray([prior.LABELS.index(obj.predict(t)[0]["label"]) for t in texts])
            print("native excerpts",mode,len(texts),flush=True)
        allpred["current-20-percent"]=routed(texts)
        tmp=frozen.with_suffix(".part")
        with tmp.open("wb") as stream:np.savez_compressed(stream,identity=signature,**allpred)
        tmp.replace(frozen)
    out={}
    ledger={}
    for name,(rows,sources) in panels.items():
        gold=np.asarray([prior.LABELS.index(r["gold"]) for r in rows]); gem=np.asarray([r["gemini"] for r in rows])
        pos=[truth[r["text"]] for r in rows]
        cases=[r["doc"] for r in rows]
        scores={}
        methods={"Gemini Decisions":gem,"archived InLegalBERT excerpt MLP":np.asarray([r["specialist"] for r in rows]),**{k:v[pos] for k,v in allpred.items()}}
        ledger[name]=[{"id":r["id"],"case":r["doc"],"input_sha256":r["sha256"],"gold":r["gold"],
                      "predictions":{tag:prior.LABELS[int(pred[i])] for tag,pred in methods.items()}} for i,r in enumerate(rows)]
        for tag,pred in methods.items():
            c=prior.cm(gold,pred)
            scores[tag]={**prior.stat(c[None,:,:]),"correct":int((pred==gold).sum()),"agrees_with_gemini":int((pred==gem).sum()),
                         "accuracy_delta_vs_gemini_ci95":bootstrap(gold,pred,gem,cases)}
        out[name]={"n":len(rows),"cases":len(set(cases)),"supports":dict(zip(prior.LABELS,np.bincount(gold,minlength=7).tolist())),
                   "sources":sources,"scores":scores}
    report={"model":"gemini-decision-1","api_calls_in_this_replay":0,"local_input":"Exactly the archived normalized string submitted to Gemini, one independent upload at a time. The current BERT reviewer applies its 96-token window.",
            "note":"Absent roles receive F1 zero in the fixed seven-class average. Use accuracy and class support in sparse panels. The four panels have different sampling procedures. The joined paragraph gold is derived from same-role sentences.",
            "roles":prior.LABELS,"panels":out,"local_predictions_sha256":prior.file_hash(frozen)}
    prior.write(where/"matched-excerpt-ledger.json",ledger)
    report["per_item_ledger_sha256"]=prior.file_hash(where/"matched-excerpt-ledger.json")
    prior.write(where/"results.json",report)
    print("matched excerpt accuracy",{k:{w:round(s["accuracy"],4) for w,s in v["scores"].items()} for k,v in out.items()},flush=True)


if __name__=="__main__":main()
