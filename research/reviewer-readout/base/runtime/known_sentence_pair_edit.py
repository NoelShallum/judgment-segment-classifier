"""Tune fixed decoder coefficients on macro-F1 for supplied sentence boundaries."""
import json

import numpy as np
from safetensors.numpy import load_file

from role_pair_weight_search import (EDGE,CRF,DONOR,ROOT,HierModel,Edit,ONE,ZERO,Packed,
    document,file_hash,gold_cases,independent_validation,load_decoder,moments,sample_whole,
    save_json,search,torch,write_profile)
from hier_full import CACHE,cm,row_summary
from hier_trial import OUT
from preserved_emitter_edit import edit_documents,embeddings
from role_edge_runtime import transition_scores
from run_entry import prepare_score
from score_frozen_edge_sentence_rows import pair


class KnownSentences(Packed):
    def bind(self, docs):
        self.gold=np.concatenate([np.asarray(d["reference_roles"],dtype=int) for d in docs])
        assert len(self.gold)==self.units
        return self

    def measure(self,bias,multiplier):
        labels=self.decode(bias,multiplier)
        summary=row_summary(cm(self.gold,labels))
        return summary["macro_f1"],summary["accuracy"]

    def confusion(self,bias,multiplier):
        labels=self.decode(bias,multiplier)
        return np.stack([cm(self.gold[a:b],labels[a:b]) for a,b in zip(self.sizes,self.sizes[1:])])


def main():
    torch.set_num_threads(2)
    engine=load_decoder()
    model=HierModel(DONOR)
    base=load_file(str(EDGE/"role-edge-balanced.safetensors"))
    vocab=json.loads((EDGE/"vocab.json").read_text())
    def pack(docs,raw):
        ds=[{**d,"units":d["reference_units"]} for d in docs]
        entries=[(prepare_score(d,x,{"exact":base}),transition_scores(d["text"],d["units"],vocab,base))
                 for d,x in zip(ds,raw)]
        return KnownSentences(entries,base,engine).bind(ds)

    train=edit_documents()
    data=pack(train,[model.logits(x) for x in embeddings(model,train,"aligned-source-train")])
    direction,ref,obt=moments(data,train)
    result=[]
    trace={}
    for goal in ("exact","balanced"):
        candidate,t=search(data,ref,goal,"known-train")
        trace[goal]=t
        for proportion in (.25,.5,.75,1.):
            result.append(Edit(f"{goal}-{proportion}",proportion*candidate.bias,
                               ONE+proportion*(candidate.gain-ONE)))
    options=[Edit("parent",ZERO,ONE),*(Edit(f"frequency-{gain}",gain*direction,ONE) for gain in (.1,.25,.5)),*result]
    groups=sample_whole()

    def prepare(s):
        ds=groups[s]
        # The aligned mode uses supplied reference boundaries for each case.
        raw=[model.logits(x) for x in aligned_embeddings(model,ds,s)]
        return pack(ds,raw)

    candidate=prepare("select")
    table={e.name:candidate.measure(e.bias,e.gain) for e in options}
    choice=max(options,key=lambda e:(table[e.name],e.name=="parent"))
    output=OUT/"known-sentence-role-pair.safetensors"
    write_profile(output,base,choice)
    freeze=ROOT/"results/known-sentence-pair-edit-frozen.json"
    save_json(freeze,{"training_cases":[d["id"] for d in train],"training_rows":data.units,"coordinate_trace":trace,
                      "selection_cases":[d["id"] for d in groups["select"]],"values":table,"choice":choice.name,
                      "bias":choice.bias.tolist(),"gain":choice.gain.tolist(),"patch_sha256":file_hash(output)})
    print("Known-sentence choice",choice.name,table[choice.name],table["parent"],flush=True)
    used={d["id"] for ds in groups.values() for d in ds}
    omitted=set(json.loads((EDGE/"overlap.json").read_text())["independent_exclusions_for_text_disjoint_check"])
    groups["independent"]=[d for d in independent_validation(used) if d["id"] not in omitted]
    output={"protocol":"English LegalSeg with official sentence offsets, full judgment context, and the seven original labels. Search optimizes known-row macro-F1 on 144 training cases using a fixed donor, then selects shrinkage before any official test replay. Only CRF and lexical transition weights are updated.",
            "freeze_sha256":file_hash(freeze),"choice":choice.name,"splits":{}}
    for s in ("select","confirm","independent","test"):
        if s=="test":
            docs=[]; raw=[]
            for k,r in gold_cases().items():
                docs.append(document(k,r))
                selected=np.asarray([i for i,(_,line,_) in enumerate(r) if line.strip()])
                with np.load(CACHE/(k+".npz")) as f:raw.append(f["context"][selected])
            bank=pack(docs,raw)
        else:
            bank=candidate if s=="select" else prepare(s)
        a=bank.confusion(choice.bias,choice.gain)
        b=bank.confusion(ZERO,ONE)
        held={"candidate":row_summary(a.sum(0)),"parent":row_summary(b.sum(0)),
              "gap_ci95_macro_f1_accuracy":pair(a,b),"cases":len(a)}
        if s=="test":
            checked=json.loads((ROOT/"results/role-edge-given-sentences.json").read_text())["profiles"]["balanced"]["candidate"]
            assert held["parent"]==checked
        output["splits"][s]=held
        save_json(ROOT/"results/known-sentence-pair-edit.json",output)
        print(s,held["candidate"]["macro_f1"],held["parent"]["macro_f1"],flush=True)


def aligned_embeddings(model,docs,split):
    source=ROOT/"cache/known-sentence-pair-edit"
    source.mkdir(exist_ok=True,parents=True)
    for d in docs:
        path=source/(d["id"]+".npy")
        if not path.is_file():
            tmp=path.with_suffix(".part")
            x=model.contextualize(model.words([d["text"][a:b] for a,b in d["reference_units"]]))
            with tmp.open("wb") as f:np.save(f,x,allow_pickle=False)
            tmp.replace(path)
        yield np.load(path,allow_pickle=False)


if __name__=="__main__":
    main()
