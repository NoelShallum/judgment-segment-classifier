"""Directly write emitter corrections that exactly preserve a predeclared subspace."""
import itertools
import numpy as np
from safetensors.numpy import load_file,save_file
from scipy.linalg import solve
import torch

from discourse_span_edit import text_features
from hier_model import CRF,EMITTER,HierModel,path_labels
from hier_trial import DONOR,OUT
from preserved_emitter_edit import edit_documents,embeddings
from program_trial import save_json
from seven_paths import ROOT
from span_eval import counts,sets,summarize
from transition_span_edit import sample_whole
from weight_edit import file_hash


def vectors(x):
    return np.column_stack((x,np.ones(len(x))))


def calculate(x,y,base):
    x=vectors(x).astype(np.float64)
    out=x@base
    rival=out.copy()
    rival[np.arange(len(x)),y]=-np.inf
    other=rival.argmax(1)
    correct=y==out.argmax(1)
    margin=out[np.arange(len(x)),y]-out[np.arange(len(x)),other]
    # Protect the high-confidence subspace, with case-disjoint tuning downstream.
    select=correct&(margin>=np.median(margin[correct]))
    protected=x[select]
    cov=protected.T@protected/len(protected)
    eig,basis=np.linalg.eigh(cov)
    ordered=basis[:,np.argsort(eig)[::-1]]
    wrong=x[~correct];t=y[~correct];f=other[~correct]
    step=np.zeros((len(wrong),7),dtype=float)
    gain=.5*np.maximum(0,1.-margin[~correct])
    step[np.arange(len(wrong)),t]=gain
    step[np.arange(len(wrong)),f]=-gain
    rnd,_=np.linalg.qr(np.random.default_rng(27714).normal(size=(len(base),len(base))))
    yield ("parent",0.,0.,0.,base,0.)
    for kind,rank in itertools.product(("protect","random_control"),(8,24,48)):
        u=(ordered if kind=="protect" else rnd)[:,:rank]
        p=np.eye(len(base))-u@u.T
        z=wrong@p
        c=z.T@z/len(z);rhs=z.T@step/len(z)
        for regularizer,scale in itertools.product((.05,.2),(.2,.5,1.)):
            update=p@solve(c+regularizer*np.trace(c)/len(c)*np.eye(len(c)),rhs,assume_a="pos")
            update*=scale
            drift=float(np.max(abs(u.T@update)))
            assert drift<2e-7
            energy=float(np.linalg.norm(z)/np.linalg.norm(wrong))
            yield (kind,rank,regularizer,scale,base+update,drift,energy)


def main():
    torch.set_num_threads(2)
    m=HierModel(DONOR)
    source=edit_documents()
    xx=np.concatenate(embeddings(m,source,"aligned-source-train"))
    y=np.concatenate([d["reference_roles"] for d in source])
    before=np.vstack((m.weight[3:].T,m.bias[3:]))
    options=list(calculate(xx,y,before))
    print("Frozen training support",len(y),"gradient-free settings",len(options),flush=True)
    groups=sample_whole();groups["screen"]=sets()["test"]
    controls={label:load_file(str(OUT/f"whole-discourse-{label}.safetensors")) for label in ("exact","balanced")}
    data={}
    for name,docs in groups.items():
        if name=="screen":
            states=embeddings(m,docs,"whole-test")
        else:
            states=embeddings(m,docs,"raw-boundary-write-"+name)
        prepare=[]
        for d,xx in zip(docs,states):
            a,b=text_features(d)
            x=vectors(xx)
            cue={k:a@v["cue_head"]+b@v["position_head"] for k,v in controls.items()}
            prepare.append((d,x,cue))
        data[name]=prepare
    def score(group,label,weights):
        stats=[]
        for d,x,cue in data[group]:
            z=path_labels(x@weights+cue[label],controls[label][CRF])
            stats.append(counts(d,d["units"],z))
        return summarize(np.sum(stats,axis=0))
    report={}
    for label,g in controls.items():
        baseline=score("select",label,before)
        best={};grid=[]
        token_floor=baseline["token_accuracy"]-.01
        for option in options:
            kind,rank,r,scale,values,drift,*extra=option
            t=score("select",label,values)
            k=t["exact_labeled_spans"]["f1"],t["token_accuracy"]
            grid.append([kind,rank,r,scale,drift,*k])
            if label=="balanced" and k[1]<token_floor:continue
            for subgroup in ("candidate","random_control"):
                if kind=="protect" and subgroup!="candidate":continue
                if kind=="random_control" and subgroup!="random_control":continue
                if subgroup not in best or k>best[subgroup][0]:best[subgroup]=(k,option)
        summary={}
        for subgroup,(_,opt) in best.items():
            kind,rank,r,scale,values,err,*extra=opt
            weights=m.weight.copy();bias=m.bias.copy()
            weights[3:]=values[:-1].T;bias[3:]=values[-1]
            path=OUT/f"whole-hard-preservation-{label}-{subgroup}.safetensors"
            save_file({**g,EMITTER+"weight":weights,EMITTER+"bias":bias},str(path))
            summary[subgroup]={"edit":[kind,rank,r,scale],"protected_direction_max_absolute_drift":err,
                 "projected_error_key_energy_fraction":extra[0] if extra else None,"weights_sha256":file_hash(path),
                 "scores":{k:{"cases":len(ds),"candidate":score(k,label,values),"incumbent":score(k,label,before)} for k,ds in groups.items()}}
        report[label]={"grid":grid,"choices":summary}
        v=summary["candidate"]["scores"]["screen"];print(label,summary["candidate"]["edit"],v["candidate"]["exact_labeled_spans"]["f1"],v["incumbent"]["exact_labeled_spans"]["f1"],flush=True)
    save_json(ROOT/"results/hard-preservation-emitter.json",{"cases_in_edit_estimation":len(source),"training_rows":len(y),
        "method":"Fit a gradient-free residual head to classification errors on the official training split, restricted to the complement of a leading subspace of confident correct train activations. Verify the projected update has zero drift on protected directions. Compare a random protected subspace under the same candidate grid. Validation chooses coefficient; case-disjoint confirmation and a 20-test-case screen are untouched by selection.",
        "profiles":report})
if __name__=="__main__":
    main()
