"""Direct, role-specific segment length weights in an augmented Viterbi CRF."""
import itertools
import json
import numpy as np
from safetensors.numpy import load_file,save_file
import torch

from fragment_trial import passages
from hier_full import CACHE,gold_cases
from hier_model import CRF,HierModel
from hier_trial import DONOR,OUT
from program_trial import save_json
from seven_paths import ROOT
from span_eval import counts,document,summarize
from transition_span_edit import features,sample_whole
from weight_edit import file_hash

# The four columns correspond to lengths of one, two, three, or at least four units.
DURATION=np.array([[0,0,0,0],
                   [-1,-.25,0,0],
                   [0,-.1,-.25,-.25],
                   [-1,-.15,0,0],
                   [-1,-.15,0,0],
                   [-1,-.35,0,0],
                   [0,-.25,-.5,-.5]],dtype="f4")

def decode(emissions, transitions, duration):
    x=np.asarray(emissions,dtype=np.float32)
    labels,ages=duration.shape
    sts=labels*ages
    roles=np.repeat(np.arange(labels),ages)
    age=np.tile(np.arange(ages),labels)
    different=roles[:,None]!=roles[None,:]
    valid_switch=different&(age[None,:]==0)
    valid_stay=~different&(np.minimum(age[:,None]+1,ages-1)==age[None,:])
    rel=transitions[3:,3:][roles[:,None],roles[None,:]]
    end=duration[roles,age]
    w=np.where(valid_switch,rel+end[:,None],np.where(valid_stay,rel,-np.inf))
    start=np.where(age==0,transitions[1,3:][roles]+x[0,roles],-np.inf)
    link=np.zeros((len(x),sts),dtype=np.int16)
    score=start;ids=np.arange(sts)
    for i in range(1,len(x)):
        step=score[:,None]+w
        best=step.argmax(0)
        link[i]=best
        score=step[best,ids]+x[i,roles]
    at=np.argmax(score+transitions[3:,2][roles]+end)
    out=np.empty(len(x),dtype=np.int64)
    for i in range(len(out)-1,-1,-1):
        out[i]=roles[at];at=link[i,at]
    return out

def main():
    torch.set_num_threads(2)
    enc=HierModel(DONOR)
    result={}
    previous=ROOT/"results/role-specific-transitions.json"
    for task,groups in (("fragment",passages()),("whole",sample_whole())):
        basepath=OUT/(("fragment" if task=="fragment" else "document")+".safetensors")
        base=load_file(str(basepath))
        if task=="fragment" and not np.isclose(base["classifier.weight"][:,200:],0).all():
            raise ValueError("Incumbent fragment readout requires independent features")
        tr=base["transition"] if task=="fragment" else base[CRF]
        if previous.exists() and task in json.loads(previous.read_text())["tasks"]:
            alt=load_file(str(OUT/("role-specific-"+task+".safetensors")))[CRF]
        else:
            alt=tr
        if task=="whole":
            groups["test"]=[document(k,rows) for k,rows in gold_cases().items()]
        inputs={k:features(enc,d) for k,d in groups.items() if task!="whole" or k!="test"}
        if task=="whole":
            inputs["test"]=[]
            for d in groups["test"]:
                with np.load(CACHE/(d["id"]+".npz")) as f:inputs["test"].append(f["raw"])
        def run(fold,t,delta):
            ds=groups[fold]
            return np.stack([counts(doc,doc["units"],decode(e,t,delta)) for doc,e in zip(ds,inputs[fold])])
        grid=[];best=None
        for version,signal,scale in itertools.product(("incumbent","role-specific"),("none","role-length","shorten"),(0.,.35,.8,1.6)):
            if (scale==0.)!=(signal=="none"):continue
            matrix={"none":np.zeros_like(DURATION),"role-length":DURATION,"shorten":-DURATION}[signal]*scale
            transitions=tr if version=="incumbent" else alt
            m=summarize(run("select",transitions,matrix).sum(0));q=(m["exact_labeled_spans"]["f1"],m["token_accuracy"])
            grid.append([version,signal,scale,*q])
            if best is None or q>best[0]:best=(q,version,signal,scale,transitions,matrix)
        _,version,signal,scale,t,d=best
        path=OUT/("duration-"+task+".safetensors")
        save_file({CRF:t,"length_bias":d},str(path))
        c={}
        for fold in groups:
            current=run(fold,t,d);control=run(fold,tr,np.zeros_like(d))
            delta=[];rng=np.random.default_rng(612)
            for i in rng.integers(len(current),size=(1500,len(current))):
                A=summarize(current[i].sum(0));B=summarize(control[i].sum(0))
                delta.append(A["exact_labeled_spans"]["f1"]-B["exact_labeled_spans"]["f1"])
            c[fold]={"candidate":summarize(current.sum(0)),"baseline":summarize(control.sum(0)),"exact_gap_ci95":np.quantile(delta,[.025,.975]).tolist()}
        result[task]={"profile":[version,signal,scale],"grid":grid,"patch_sha256":file_hash(path),"scores":c}
        save_json(ROOT/"results/duration-weights.json",result)
        print(task,version,signal,scale,c["test"]["candidate"]["exact_labeled_spans"]["f1"],flush=True)

if __name__=="__main__":
    main()
