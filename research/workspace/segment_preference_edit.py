"""Write a residual segment score from local and complete-path preference pairs."""
from itertools import product
import json

import numpy as np
from safetensors.numpy import save_file

from role_edge_interpolation import (
    EDGE, DONOR, HierModel, ROOT, bootstrap, document, file_hash, full_score, gold_cases,
    independent_validation, load_file, metric, prepare, sample_whole, save_json, torch,
)
from role_pair_weight_search import edit_documents, embeddings, prepare_score, transition_scores
from segment_span_runtime import learned_scores, span_features
from segment_weight_write import engine
from span_posterior_decode import spans


def span_ids(arcs, path, units):
    dim=(units,units+1,7)
    return np.isin(np.ravel_multi_index(arcs.T,dim),np.ravel_multi_index(spans(path).T,dim))


def compile_row(item,edge,parents,incumbent,decoder):
    total=item.token_role_counts.sum(1)
    options=(item.logits,edge,parents,total,incumbent["proposals"],float(incumbent["temperature"]))
    original,*_=span_features(*options)
    arcs,_,x,prior,_=span_features(*options,posterior_ends=True)
    n=len(total)
    old=np.isin(np.ravel_multi_index(arcs.T,(n,n+1,7)),
                np.ravel_multi_index(original.T,(n,n+1,7)),assume_unique=True)
    assert old.sum()==len(original)
    u=prior+float(incumbent["share"])*(learned_scores(x[:,:69],incumbent)-prior)-float(incumbent["span_cost"])
    current=decoder(arcs[old],u[old],n)
    # Inference features depend only on the model, candidate offsets, and token totals.
    a,b,k=arcs.T
    present=span_ids(arcs,current,n)
    border=np.zeros(n+1)
    border[0]=border[-1]=1.
    border[1:-1]=current[:-1]!=current[1:]
    features=np.column_stack((x,u,old,present,border[a],border[b],current[a]==k,current[b-1]==k))
    exact=((item.offsets[b]==item.gold_end[a])&(k==item.gold_label[a]))
    return item,arcs,features,u,old,exact,present


def support(row,decoder):
    item,arcs,_,u,old,exact,chosen=row
    n=len(item.offsets)-1
    a,b,k=arcs.T
    preferences=[]
    for index in np.flatnonzero(exact):
        related=(a==a[index]) | (b==b[index]) | ((k==k[index]) & (np.abs(a-a[index])+np.abs(b-b[index])<=2))
        rivals=np.flatnonzero(related&~exact)
        top=rivals[np.argsort(u[rivals],kind="stable")[-3:]]
        preferences.extend((index,int(r)) for r in top)
    good=decoder(arcs,exact.astype(float)-.20+.25*u,n)
    better=span_ids(arcs,good,n)
    return np.asarray(preferences,dtype=np.int64).reshape(-1,2),better


def fit(rows,decoder):
    x=np.concatenate([r[2] for r in rows])
    center=x.mean(0)
    scale=1/np.maximum(x.std(0),.06)
    local=[];local_target=[];complete=[];complete_target=[];anchor=[];anchor_target=[];anchor_mass=[]
    for row in rows:
        item,arcs,f,u,old,y,current=row
        z=np.column_stack(((f-center)*scale,np.ones(len(y))))
        anchor.append(z)
        anchor_mass.append(np.where(y,2.,np.where(current,1.,.28)))
        anchor_target.append(np.where(y,.67,-.035)-u-.24)
        rival,oracle=support(row,decoder)
        if len(rival):
            good,bad=rival.T
            local.append(z[good]-z[bad])
            local_target.append(np.maximum(.35-u[good]+u[bad],0))
        add=oracle&~current
        remove=current&~oracle
        used=add.sum()+remove.sum()
        if used and y[oracle].sum()>y[current].sum():
            norm=used**.5
            complete.append((z[add].sum(0)-z[remove].sum(0))/norm)
            target=.52*(y[oracle].sum()-y[current].sum())-.20*(add.sum()-remove.sum())
            complete_target.append((target-u[add].sum()+u[remove].sum())/norm)
    x=np.concatenate(anchor)
    mass=np.concatenate(anchor_mass)
    regularization=np.r_[np.ones(x.shape[1]-1),.12]
    absolute=(x.T@(mass[:,None]*x)/mass.sum(),x.T@(mass*np.concatenate(anchor_target))/mass.sum())
    local_x=np.concatenate(local)
    complete_x=np.stack(complete)
    count={"point_targets":int(len(x)),"local_pairs":int(len(local_x)),"preferred_paths":int(len(complete_x))}
    options={}
    for name,lm,cm in (("point",0.,0.),("local",.55,0.),("paths",0.,.32),("joint",.50,.24)):
        gram,rhs=(v.copy() for v in absolute)
        if lm:
            gram+=lm*(local_x.T@local_x)/len(local_x)
            rhs+=lm*(local_x.T@np.concatenate(local_target))/len(local_x)
        if cm:
            gram+=cm*(complete_x.T@complete_x)/len(complete_x)
            rhs+=cm*(complete_x.T@np.asarray(complete_target))/len(complete_x)
        for ridge in (.12,.48):
            coef=np.linalg.solve(gram+np.diag(regularization*ridge*np.trace(gram)/len(gram)),rhs)
            options[name,ridge]={"center":center,"scale":scale,"coefficient":coef[:-1].copy(),"bias":np.asarray(coef[-1])}
    return options,count


def adjustments(rows,weights):
    return [np.clip(((r[2]-weights["center"])*weights["scale"])@weights["coefficient"]+weights["bias"],-.6,.6)
            for r in rows]


def score(rows,decoder,residual=None,amount=0.,expand=False,penalty=0.):
    stats=[]
    edits=residual if residual is not None else [0.]*len(rows)
    for row,delta in zip(rows,edits):
        item,arcs,_,u,old,_,current=row
        keep=slice(None) if expand else old
        score=u+amount*delta-penalty*~old
        labels=decoder(arcs[keep],score[keep],len(item.offsets)-1)
        if residual is None:
            assert np.array_equal(current,span_ids(arcs,labels,len(labels)))
        stats.append(item.score(labels))
    return np.stack(stats)


def main():
    torch.set_num_threads(2)
    parents={k:load_file(str(EDGE/f"role-edge-{k}.safetensors")) for k in ("exact","balanced")}
    vocab=json.loads((EDGE/"vocab.json").read_text())
    incumbent_path=ROOT/"checkpoints/hierarchical/whole-segment-weights.safetensors"
    incumbent=load_file(str(incumbent_path))
    model=HierModel(DONOR)
    decode=engine()
    groups=sample_whole()

    def build(bank,name):
        rows=[compile_row(item,edge,parents,incumbent,decode) for item,edge in bank]
        print("Prepared preference features",name,len(rows),sum(len(r[1]) for r in rows),flush=True)
        return rows

    train=edit_documents()
    states=embeddings(model,train,"raw-emitter-train")
    raw=[(prepare_score(d,model.logits(x),parents),transition_scores(d["text"],d["units"],vocab,parents["exact"]))
         for d,x in zip(train,states)]
    specimens=build(raw,"train")
    weights,counts=fit(specimens,decode)
    print("Preference evidence",counts,flush=True)
    select=build(prepare(groups["select"],"select",model,parents,vocab,parents["exact"]),"select")
    values=[]
    parent=metric(score(select,decode).sum(0))
    best=parent
    choice=None
    for name,ridge in weights:
        delta=adjustments(select,weights[name,ridge])
        for amount,extra in product((.06,.15,.35,.65),(None,.0,.055,.14)):
            q=metric(score(select,decode,delta,amount,extra is not None,extra or 0.).sum(0))
            values.append([name,ridge,amount,extra,*q])
            if q>best and q[1]>=parent[1]-.008:
                choice=[name,ridge,amount,extra]
                best=q
    # Include an untouched-score expanded lattice to isolate the trained score's contribution.
    expansion_controls={}
    for cost in (.0,.055,.14):
        q=metric(score(select,decode,[0.]*len(select),0.,True,cost).sum(0))
        expansion_controls[str(cost)]=q
    freeze=ROOT/"results/segment-preference-frozen.json"
    out=ROOT/"checkpoints/hierarchical/whole-segment-preference.safetensors"
    if choice:
        name,ridge,amount,extra=choice
        write={**weights[name,ridge],"gain":np.asarray(amount),"extend":np.asarray(extra is not None),
               "proposal_cost":np.asarray(extra or 0.)}
        save_file(write,str(out))
        physical=load_file(str(out))
        assert all(np.array_equal(v,physical[k]) for k,v in write.items())
    save_json(freeze,{"choice":choice,"grid":values,"parent":parent,"selected_scores":best,"counts":counts,
             "selection_cases":[d["id"] for d in groups["select"]],"train_cases":[d["id"] for d in train],
             "incumbent_sha256":file_hash(incumbent_path),"expansion_controls":expansion_controls,
             "weights_sha256":file_hash(out) if choice else None})
    print("Selected preference edit",choice,best,parent,flush=True)
    used={d["id"] for v in groups.values() for d in v}
    excluded=set(json.loads((EDGE/"overlap.json").read_text())["independent_exclusions_for_text_disjoint_check"])
    groups["independent"]=[d for d in independent_validation(used) if d["id"] not in excluded]
    report={"choice":choice,"freeze_sha256":file_hash(freeze),"splits":{},
       "method":"Closed-form residual on the exact segment scorer. Pair correct spans with hard negatives sharing either endpoint or nearby boundaries. Compare local preferences separately and alongside paths with more correct spans. Inference retains the frozen encoder, proposes candidates using raw text, and decodes valid tilings with one of seven roles per span."}
    for s in ("select","confirm","independent","test"):
        if s=="test":
            groups[s]=[document(k,r) for k,r in gold_cases().items()]
        rows=select if s=="select" else build(prepare(groups[s],s,model,parents,vocab,parents["exact"]),s)
        base=score(rows,decode)
        candidate=base
        plain=base
        if choice:
            extra=bool(physical["extend"])
            cost=float(physical["proposal_cost"])
            candidate=score(rows,decode,adjustments(rows,physical),float(physical["gain"]),extra,cost)
            plain=score(rows,decode,[0.]*len(rows),0.,extra,cost)
        if s=="select":
            np.testing.assert_allclose(metric(candidate.sum(0)),best,rtol=0,atol=1e-12)
        q={"cases":len(rows),"candidate":full_score(candidate),"parent":full_score(base),
           "unchanged_scores_on_selected_lattice":full_score(plain),"gap_ci95":bootstrap(candidate,base)}
        report["splits"][s]=q
        save_json(ROOT/"results/segment-preference-edit.json",report)
        print(s,metric(candidate.sum(0)),metric(base.sum(0)),flush=True)


if __name__=="__main__":
    main()
