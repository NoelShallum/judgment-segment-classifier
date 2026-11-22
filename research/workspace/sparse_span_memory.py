"""Test an explicitly written key/value role memory on cached encoder representations."""
import argparse
import itertools
import json

import numpy as np
from safetensors.numpy import load_file,save_file
import torch

import encoder_span_edit as trial
from hier_model import CRF, HierModel,path_labels
from hier_trial import DONOR,OUT
from program_trial import fixture_rows,save_json
from role_fixtures import ANCHORS,CHECKS
from seven_paths import ROOT
from span_eval import counts,summarize
from transition_span_edit import features
from weight_edit import file_hash


def retrieve(x,keys,memory,k,temperature):
    similarity=x@keys.T
    chosen=np.argpartition(similarity,-k,axis=1)[:,-k:]
    sim=np.take_along_axis(similarity,chosen,axis=1)
    weight=np.exp((sim-sim.max(1,keepdims=True))/temperature)
    weight/=weight.sum(1,keepdims=True)
    return np.einsum("nk,nkr->nr",weight,memory[chosen])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("encoder",choices=trial.MODELS)
    args=p.parse_args();torch.set_num_threads(2)
    name=args.encoder
    path=ROOT/"results"/f"encoder-span-{name}.json"
    if not path.is_file():path=path.with_name(path.stem+"-accelerated.json")
    meta=json.loads(path.read_text())
    trial.CACHE=trial.CACHE/(name if meta.get("backend","torch")=="torch" else name+"-gpu")
    model=trial.Encoder(name)
    corpus,bank=trial.make_documents()
    fixtures=fixture_rows(ANCHORS)+fixture_rows(CHECKS)
    y=np.asarray([r[3] for r in fixtures])
    keys=model.cache({"keys":[r[2] for r in fixtures]},"isolated")["keys"]
    memory=np.eye(7,dtype=np.float32)[y]
    hier=HierModel(DONOR)
    output={}
    pathout=ROOT/"checkpoints"/f"encoder-span-{name}"
    with torch.inference_mode():
        for task,splits in corpus.items():
            selected=meta["tasks"][task]["best"]
            mode=selected["mode"]
            enc=model.cache({k:v for k,v in bank.items() if k.startswith(task+"/")},mode)
            specs=list(itertools.product((1,3,7),(.04,.12),(.25,.5,1.)))
            tr_original=load_file(str(OUT/(("fragment" if task=="fragment" else "document")+".safetensors")))
            baseline=tr_original["transition"] if task=="fragment" else tr_original[CRF]
            no_edge=baseline.copy();no_edge[1,3:]*=0;no_edge[3:,2]*=0
            if task=="fragment" and not np.isclose(tr_original["classifier.weight"][:,200:],0).all():
                raise ValueError("Fragment baseline needs independent states")
            raw={split:features(hier,list(docs.values())) for split,docs in splits.items()}
            z={}
            for k,t in ((1,.04),(1,.12),(3,.04),(3,.12),(7,.04),(7,.12)):
                z[k,t]={idx:retrieve(v,keys,memory,k,t) for idx,v in enc.items()}
            def eval_spec(group,k,t,alpha,transition):
                values=[]
                for i,(doc_id,d) in enumerate(splits[group].items()):
                    score=raw[group][i]
                    if alpha:
                        # Centered log probabilities provide finite additive evidence for each role.
                        direction=np.log(np.maximum(z[k,t][f"{task}/{group}/{doc_id}"],.01))
                        score=(1-alpha)*score+alpha*direction
                    pred=path_labels(score,transition)
                    values.append(counts(d,d["units"],pred))
                return summarize(np.sum(values,0))
            grid=[];best=None;pure=None
            for k,t,alpha in specs:
                for edges,transition in (("incumbent",baseline),("zero",no_edge)):
                    q=eval_spec("select",k,t,alpha,transition)
                    metric=q["exact_labeled_spans"]["f1"],q["token_accuracy"]
                    grid.append([k,t,alpha,edges,*metric])
                    cand=metric,k,t,alpha,edges,transition
                    if best is None or metric>best[0]:best=cand
                    if alpha==1. and (pure is None or metric>pure[0]):pure=cand
            results={}
            for name2,opt in (("best",best),("pure",pure)):
                _,k,t,a,e,tr=opt
                target=pathout/f"{task}-sparse-{name2}.safetensors"
                save_file({"keys":keys,"roles":memory,"weights":np.array([k,t,a],np.float32),CRF:tr},str(target))
                results[name2]={"k":k,"temperature":t,"share":a,"edges":e,"mode":mode,"sha256":file_hash(target),
                               "scores":{f:{"candidate":eval_spec(f,k,t,a,tr),"hierarchy":eval_spec(f,1,.04,0.,baseline),"cases":len(ds)} for f,ds in splits.items()}}
            output[task]={"grid":grid,**results}
            print(name,task,results["best"]["scores"]["test"]["candidate"]["exact_labeled_spans"]["f1"],flush=True)
    save_json(ROOT/"results"/f"encoder-sparse-memory-{name}.json",{
        "method":"Physically stored key vectors and one-hot values, retrieved with top-k cosine scores. Compare a standalone memory against a weighted hierarchy contribution. All keys are authored role examples. No gradient steps.",
        "tasks":output})

if __name__=="__main__":
    main()
