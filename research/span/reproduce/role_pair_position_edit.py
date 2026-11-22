"""Write position-dependent role-pair frequency corrections into the lexical table."""
import json

import numpy as np
from safetensors.numpy import load_file,save_file

from role_pair_weight_search import (EDGE,CRF,DONOR,ROOT,OUT,HierModel,Packed,ONE,ZERO,
    bootstrap,document,file_hash,full_score,gold_cases,independent_validation,
    load_decoder,metric,moments,prepare,sample_whole,save_json,torch)
from preserved_emitter_edit import edit_documents,embeddings
from role_edge_runtime import transition_scores
from run_entry import prepare_score

PHASES=4


def phases(packed):
    return np.minimum(PHASES-1,packed.start_lookup*PHASES//packed.case_end)


def conditional_counts(packed,labels):
    eligible=~packed.start_mask[1:]
    combos=PHASES*49
    start,end=labels[:-1].astype(np.int64),labels[1:].astype(np.int64)
    y=phases(packed)[1:]*49+7*start+end
    return np.bincount(y[eligible],minlength=combos).reshape(PHASES,7,7)


def profile(base, delta, g, vocab):
    t=base["edge.log_ratio"].copy()
    gain=float(base["edge.gain"])
    for key,index in vocab.items():
        if key.startswith("zone:"):
            phase=int(key.removeprefix("zone:"))//(16//PHASES)
            t[index]+=(g/gain)*delta[phase]
    return {**base,"edge.log_ratio":t}


def fast_score(packed,g,delta,phase,per_case=False):
    # A position effect enters the same term as the lexical table at each edge.
    old=packed.edge
    try:
        packed.edge=np.ascontiguousarray(old+g/packed.scale*delta[phase],dtype=np.float32)
        return packed.per_case(ZERO,ONE) if per_case else packed.measure(ZERO,ONE)
    finally:
        packed.edge=old


def main():
    torch.set_num_threads(2)
    engine=load_decoder()
    parent=load_file(str(EDGE/"role-edge-exact.safetensors"))
    vocabulary=json.loads((EDGE/"vocab.json").read_text())
    model=HierModel(DONOR)
    train=edit_documents()
    embeddings_train=embeddings(model,train,"raw-emitter-train")
    samples=[(prepare_score(d,model.logits(x),{"exact":parent}),transition_scores(d["text"],d["units"],vocabulary,parent))
             for d,x in zip(train,embeddings_train)]
    fit=Packed(samples,parent,engine)
    train_gold=fit.token_counts.argmax(1)
    train_pred=fit.decode(ZERO,ONE)
    actual=conditional_counts(fit,train_gold)
    estimate=conditional_counts(fit,train_pred)
    def normalize(values):
        total=values.sum(0)+8
        whole=total/total.sum(1,keepdims=True)
        count=values+60*whole[None,:,:]+2.
        return count/count.sum(-1,keepdims=True)
    local=np.clip(np.log(normalize(actual)/normalize(estimate)),-1.15,1.15).astype(np.float32)
    shared,*_=moments(fit,train)
    options={"pooled":np.repeat(shared[None,:,:],PHASES,axis=0),
             "position":local,
             "position-centered":local-(local.mean(0))}
    options["none"]=np.zeros_like(local)
    groups=sample_whole()
    def packed(split):
        return Packed(prepare(groups[split],split,model,{"exact":parent},vocabulary,parent),parent,engine)
    data=packed("select")
    location=phases(data)
    settings=[("none",0.),*((k,g) for k in ("position","position-centered","pooled") for g in (.1,.2,.35,.5))]
    table={e:fast_score(data,e[1],options[e[0]],location) for e in settings}
    chosen=max(settings,key=lambda e:(table[e],e==("none",0.)))
    print("Position choice",chosen,table[chosen],table[("none",0.)],flush=True)
    p=OUT/"whole-role-pair-position.safetensors"
    save_file(profile(parent,options[chosen[0]],chosen[1],vocabulary),str(p))
    record=ROOT/"results/role-pair-position-frozen.json"
    save_json(record,{"choice":chosen,"weights_sha256":file_hash(p),"profile":options[chosen[0]].tolist(),
                      "train_case_ids":[d["id"] for d in train],"gold_direction_counts":actual.tolist(),
                      "predicted_direction_counts":estimate.tolist(),"grid":[[*e,*v] for e,v in table.items()],
                      "selection_cases":[d["id"] for d in groups["select"]]})
    result={"protocol":"Estimate conditional role-pair frequency residuals in four relative-length bins using 144 training cases. Add the resulting 7×7 corrections to the existing positional features in the lexical table. Compare phase-specific, centered and pooled variants. Validation freezes one setting without gradients.","choice":chosen,"freeze_sha256":file_hash(record),"splits":{}}
    selected=load_file(str(p))
    used={d["id"] for values in groups.values() for d in values}
    exclusions=set(json.loads((EDGE/"overlap.json").read_text())["independent_exclusions_for_text_disjoint_check"])
    groups["independent"]=[d for d in independent_validation(used) if d["id"] not in exclusions]
    for split in ("select","confirm","independent","test"):
        if split=="test":
            groups[split]=[document(k,r) for k,r in gold_cases().items()]
        bank=data if split=="select" else packed(split)
        control=bank.per_case(ZERO,ONE)
        # Replay the physical table before drawing conclusions.
        bank.edge=np.ascontiguousarray(np.concatenate([
            np.concatenate((np.zeros((1,7,7),np.float32),
                 transition_scores(d["text"],d["units"],vocabulary,selected))) for d in groups[split]]),dtype=np.float32)
        q=bank.per_case(ZERO,ONE)
        result["splits"][split]={"candidate":full_score(q),"parent":full_score(control),"cases":len(q),
                                 "gap_ci95":bootstrap(q,control)}
        save_json(ROOT/"results/role-pair-position-edit.json",result)
        print(split,metric(q.sum(0)),metric(control.sum(0)),flush=True)


if __name__=="__main__":
    main()
