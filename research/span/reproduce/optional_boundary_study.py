"""Write boundary likelihood and evaluate optional cuts with an unchanged donor encoder."""
import csv
import json
from collections import Counter, defaultdict
from itertools import groupby, product
import hashlib

import numpy as np
from safetensors.numpy import load_file,save_file

from role_edge_interpolation import (EDGE, CRF, DONOR, ROOT, HierModel, bootstrap,
    document, file_hash, full_score, gold_cases, independent_validation, metric,
    sample_whole,save_json,torch)
from discourse_span_edit import text_features
from hier_full import CACHE
from program_trial import TO_ID
from role_cues import cue_features
from role_edge_runtime import decompose,signatures,transition_scores
from run_entry import prepare_score
from optional_boundary_runtime import proposals, select_cuts, decode_blocks

RESULT=ROOT/"results"
PATCH=ROOT/"checkpoints/hierarchical/optional-boundaries"
VARIANTS=[None,*product((.75,1.5),(0.,.30),(.0,.7,1.4))]
FEATURE_WEIGHTS={"type":1.,"shape":.35,"left":.22,"right":.38,"prefix":.35,"length":.22}


def read_train():
    source=ROOT/"references/official/train.csv"
    hit=Counter()
    seen=Counter()
    n=positive=cases=0
    for k, rows in training_cases(source):
        d=document(k,rows)
        boundaries={s["start"] for s in d["gold_spans"]}
        for _,cut,features in proposals(d["text"],d["units"]):
            good=cut in boundaries
            n+=1
            positive+=good
            seen.update(features)
            if good:
                hit.update(features)
        cases+=1
        if cases%800==0:
            print("Counted training cases",cases,flush=True)
    probability=positive/n
    logit=np.log(probability/(1-probability))
    vocab={k:i+1 for i,k in enumerate(sorted(k for k in seen if seen[k]>=16))}
    delta=np.zeros(len(vocab)+1,np.float32)
    for k,i in vocab.items():
        p=(hit[k]+35*probability)/(seen[k]+35)
        delta[i]=FEATURE_WEIGHTS[k.split(":",1)[0]]*(np.log(p/(1-p))-logit)
    PATCH.mkdir(exist_ok=True,parents=True)
    save_file({"cut.base":np.asarray(logit,np.float32),"cut.delta":delta},str(PATCH/"prior.safetensors"))
    save_json(PATCH/"vocab.json",vocab)
    report={"source_sha256":file_hash(source),"cases":cases,"options":n,"gold_role_changes":positive,
            "probability":probability,"counts_by_feature_type":{k:[v,hit[k]] for k,v in seen.items() if k.startswith("type:")},
            "weights_sha256":file_hash(PATCH/"prior.safetensors"),"vocab_sha256":file_hash(PATCH/"vocab.json")}
    save_json(RESULT/"optional-boundary-train.json",report)
    print("Froze boundary counts",n,positive,flush=True)


def training_cases(source):
    with source.open(encoding="utf-8-sig",newline="") as fh:
        for k, rows in groupby(csv.DictReader(fh),key=lambda r:r["Index"]):
            yield k,[(i,r["Text"],TO_ID[r["Label"]]) for i,r in enumerate(rows)]


def package(d, model, parent, raw, prior_vocab, prior, lex_vocab):
    text=d["text"]
    offsets=d["units"]
    best=select_cuts(text,offsets,prior_vocab,prior)
    chosen=defaultdict(list)
    for b,c,p in best:
        a,z=offsets[b]
        chosen[b].append((c,p,(a,c),(c,z)))
    pieces=[text[a:z] for _,vals in chosen.items() for _,_,left,right in vals for a,z in (left,right)]
    local=model.logits(model.contextualize(model.words(pieces),independent=True)) if pieces else np.empty((0,7))
    if pieces:
        local=local+cue_features(pieces)@parent["cue_head"]
    cue,position=text_features(d)
    emit=raw+cue@parent["cue_head"]+position@parent["position_head"]
    memory=parent["edge.log_ratio"]
    pair_texts=[]
    for vals in chosen.values():
        for _,_,l,r in vals:
            pair_texts.append([lex_vocab.get(k,0) for k in next(signatures(text,[l,r]))])
    relative=decompose(memory[np.asarray(pair_texts)].sum(1),"interaction") if pair_texts else []
    options=[]
    atoms=[]
    coordinates=[]
    i=0
    for b,(a,z) in enumerate(offsets):
        cursor=len(atoms)
        full=[a]+[c for c,*_ in chosen[b]]+[z]
        atoms.extend(zip(full,full[1:]))
        variants=[]
        for k,(c,p,*_) in enumerate(chosen[b],start=1):
            left=(c-a)/(z-a)
            difference=(local[i*2]-local[i*2+1])*.5
            difference-=difference.mean()
            variants.append((p,left,difference,relative[i],cursor+k))
            i+=1
        options.append(variants)
        coordinates.append((cursor,len(atoms)))
    aug={**d,"units":atoms}
    scorer=prepare_score(aug,np.zeros((len(atoms),7),np.float32),{})
    edge=parent[CRF][3:,3:]+float(parent["edge.gain"])*transition_scores(text,offsets,lex_vocab,parent)
    gold={x["start"] for x in d["gold_spans"]}
    found={c for _,c,_ in best}
    return scorer,emit,edge,options,coordinates,len(found),len(found&gold)


def arrays(compiled, parent, spec):
    scored=[]
    recovered=0
    added=0
    for scorer,em,edge,options,coordinates,_,_ in compiled:
        alternatives=[[] for _ in options]
        if spec is not None:
            gain,prior_gain,cost=spec
            for i,variants in enumerate(options):
                for p,left,diff,lex,_ in variants:
                    delta=gain*diff
                    first=left*em[i]+delta
                    second=(1-left)*em[i]-delta
                    pair=first[:,None]+second[None,:]+.2*lex + prior_gain*p-cost
                    np.fill_diagonal(pair,-np.inf)
                    alternatives[i].append(pair)
        path=decode_blocks(em,edge,parent[CRF],alternatives)
        roles=np.empty(len(scorer.offsets)-1,np.int8)
        for b,(first,last,choice) in enumerate(path):
            a,z=coordinates[b]
            if choice>=0:
                cut=options[b][choice][-1]
                roles[a:cut]=first
                roles[cut:z]=last
                recovered+=scorer.gold_label[cut]>=0
                added+=1
            else:
                roles[a:z]=last
        scored.append(scorer.score(roles))
    return np.stack(scored),added,int(recovered)


def main():
    torch.set_num_threads(2)
    if not (RESULT/"optional-boundary-train.json").exists():
        read_train()
    prior=load_file(str(PATCH/"prior.safetensors"))
    prior_vocab=json.loads((PATCH/"vocab.json").read_text())
    lex=json.loads((EDGE/"vocab.json").read_text())
    parent=load_file(str(EDGE/"role-edge-exact.safetensors"))
    groups=sample_whole()
    model=HierModel(DONOR)

    def prepare(cases, split):
        out=[]
        for i,d in enumerate(cases,1):
            if split=="test":
                with np.load(CACHE/(d["id"]+".npz")) as fp:
                    raw=fp["raw"]
            elif split=="independent":
                sig=hashlib.sha256(d["text"].encode()).hexdigest()
                raw=np.load(ROOT/"astra-run-penalty/validation-cache"/(sig+".npy"))
            else:
                raw=model.logits(model.contextualize(model.words([d["text"][a:b] for a,b in d["units"]])))
            out.append(package(d,model,parent,raw,prior_vocab,prior,lex))
            if i%72==0 or i==len(cases):
                print("Encoded alternate cuts",split,i,flush=True)
        return out

    chosen=None
    grid=[]
    selection=prepare(groups["select"],"select")
    baseline=metric(arrays(selection,parent,None)[0].sum(0))
    best=baseline
    for spec in VARIANTS:
        v,opened,matched=arrays(selection,parent,spec)
        ranking=metric(v.sum(0))
        grid.append([spec,*ranking,opened,matched])
        if ranking>best:
            best=ranking
            chosen=spec
    freeze=RESULT/"optional-boundary-frozen.json"
    save_json(freeze,{"choice":chosen,"grid":grid,"selection_case_ids":[d["id"] for d in groups["select"]],
                      "prior_sha256":file_hash(PATCH/"prior.safetensors"),"parent_sha256":file_hash(EDGE/"role-edge-exact.safetensors")})
    print("Selected",chosen,best,"parent",baseline,flush=True)
    if chosen:
        save_file({**prior,"cut.policy":np.asarray(chosen,np.float32)},str(PATCH/"selected.safetensors"))
    used={d["id"] for docs in groups.values() for d in docs}
    excluded=set(json.loads((EDGE/"overlap.json").read_text())["independent_exclusions_for_text_disjoint_check"])
    groups["independent"]=[d for d in independent_validation(used) if d["id"] not in excluded]
    report={"method":"Each baseline unit becomes a block with its unchanged emission, or a split with distinct roles. Internal cut evidence is estimated from gold role-change counts on the official training split. Frozen donor local left-versus-right scores and the existing CRF score both choices. The block decoder preserves adjacent context and is exact over the available choices. Zero gradient steps.",
            "selected":chosen,"frozen_sha256":file_hash(freeze),"splits":{}}
    for split in ("select","confirm","independent","test"):
        if split=="test":
            groups["test"]=[document(k,rows) for k,rows in gold_cases().items()]
        bank=selection if split=="select" else prepare(groups[split],split)
        cand,opened,matched=arrays(bank,parent,chosen)
        null,_,_=arrays(bank,parent,None)
        report["splits"][split]={"cases":len(bank),"candidate":full_score(cand),"parent":full_score(null),
                                 "options":sum(r[-2] for r in bank),"reachable_extra_gold_changes":sum(r[-1] for r in bank),
                                 "opened":opened,"opened_at_gold_change":matched,"gap_ci95":bootstrap(cand,null)}
        if split=="test":
            reference=json.loads((EDGE/"results.json").read_text())["splits"]["test"]["exact"]["candidate"]
            assert full_score(null)==reference
        save_json(RESULT/"optional-boundary-lattice.json",report)
        print(split,metric(cand.sum(0)),metric(null.sum(0)),"opened",opened,"at gold",matched,flush=True)


if __name__=="__main__":
    main()
