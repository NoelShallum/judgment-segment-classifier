"""Write pair-specific transition coefficients with gradient-free sequence-level search."""
import ctypes
from dataclasses import dataclass
import json
import subprocess

import numpy as np
from safetensors.numpy import load_file,save_file

from role_edge_interpolation import (EDGE,CRF,DONOR,ROOT,HierModel,bootstrap,decode,
    document,file_hash,full_score,gold_cases,independent_validation,metric,
    prepare,sample_whole,save_json,torch)
from preserved_emitter_edit import edit_documents,embeddings
from role_edge_runtime import transition_scores
from run_entry import prepare_score

RESULTS=ROOT/"results"
OUT=ROOT/"checkpoints/hierarchical"


def load_decoder():
    source=ROOT/"packed_role_decode.cpp"
    digest=file_hash(source)[:16]
    lib=ROOT/"cache"/f"packed-role-decode-{digest}.so"
    if not lib.is_file():
        tmp=lib.with_suffix(".part")
        subprocess.run(["g++","-std=c++17","-shared","-fPIC","-O3","-ffp-contract=off",
                        str(source),"-o",str(tmp)],check=True)
        tmp.replace(lib)
    nD=np.ctypeslib.ndpointer
    decoder=ctypes.CDLL(str(lib)).decode_judgments
    decoder.argtypes=[nD(np.float64,flags="C_CONTIGUOUS"),nD(np.float32,flags="C_CONTIGUOUS"),
                      nD(np.int64,flags="C_CONTIGUOUS"),ctypes.c_size_t,
                      nD(np.float32,flags="C_CONTIGUOUS"),nD(np.float32,flags="C_CONTIGUOUS"),
                      nD(np.int8,flags="C_CONTIGUOUS")]
    decoder.restype=None
    return decoder


class Packed:
    def __init__(self,bank,parent,engine):
        self.items=[q for q,_ in bank]
        self.sizes=np.r_[0,np.cumsum([len(q.logits["exact"]) for q in self.items],dtype=np.int64)]
        self.logits=np.ascontiguousarray(np.concatenate([q.logits["exact"] for q in self.items]),dtype=np.float64)
        self.edge=np.ascontiguousarray(np.concatenate([np.concatenate((np.zeros((1,7,7),np.float32),v)) for _,v in bank]),dtype=np.float32)
        self.token_counts=np.concatenate([q.token_role_counts for q in self.items])
        self.units=len(self.logits)
        self.start_mask=np.zeros(self.units,bool)
        self.start_mask[self.sizes[:-1]]=1
        self.final_index=np.repeat(self.sizes[1:],np.diff(self.sizes))
        self.start_lookup=np.concatenate([q.offsets[:-1] for q in self.items])
        self.case_end=np.repeat([q.offsets[-1] for q in self.items],np.diff(self.sizes))
        self.gold_ends=np.concatenate([q.gold_end for q in self.items])
        self.gold_roles=np.concatenate([q.gold_label for q in self.items])
        self.gold_total=sum(x.truth_run_counts.sum() for x in self.items)
        self.total=self.token_counts.sum()
        self.transition=parent[CRF]
        self.scale=float(parent["edge.gain"])
        self.engine=engine

    def decode(self,bias,multiplier):
        out=np.empty(self.units,np.int8)
        matrix=self.transition.copy()
        matrix[3:,3:]+=bias
        self.engine(self.logits,self.edge,self.sizes,len(self.items),matrix,
                    np.ascontiguousarray(self.scale*multiplier,dtype=np.float32),out)
        return out

    def measure(self,bias,multiplier):
        y=self.decode(bias,multiplier)
        changed=self.start_mask.copy()
        changed[1:] |= y[1:]!=y[:-1]
        left=np.flatnonzero(changed)
        after=np.r_[left[1:],self.units]
        lookup=self.start_lookup[np.minimum(after,self.units-1)]
        right=np.where(after>=self.final_index[left],self.case_end[left],lookup)
        hits=((right==self.gold_ends[left])&(y[left]==self.gold_roles[left])).sum()
        correct=self.token_counts[np.arange(self.units),y].sum()
        return 2*hits/(len(left)+self.gold_total),correct/self.total

    def per_case(self,bias,multiplier):
        y=self.decode(bias,multiplier)
        return np.stack([d.score(q) for d,q in zip(self.items,np.split(y,self.sizes[1:-1]))])


ZERO=np.zeros((7,7),np.float32)
ONE=np.ones((7,7),np.float32)


@dataclass
class Edit:
    name: str
    bias: np.ndarray
    gain: np.ndarray


def moments(bank, data):
    predicted=bank.decode(ZERO,ONE)
    annotated=bank.token_counts.argmax(1)
    use=~bank.start_mask[1:]
    reference=np.bincount(7*annotated[:-1][use]+annotated[1:][use],minlength=49).reshape(7,7)
    obtained=np.bincount(7*predicted[:-1][use]+predicted[1:][use],minlength=49).reshape(7,7)
    conditional=lambda x:(x+8)/(x+8).sum(1,keepdims=True)
    update=np.clip(np.log(conditional(reference)/conditional(obtained)),-1.,1.).astype(np.float32)
    return update,reference.tolist(),obtained.tolist()


def search(bank, support, objective, train_name):
    score_weight=.15 if objective=="balanced" else .03
    bias=ZERO.copy()
    gain=ONE.copy()
    def value(d,g):
        f1,acc=bank.measure(d,g)
        ridge=.008*np.square(d).mean()+.004*np.square(g-1).mean()
        return float(f1+score_weight*acc-ridge),float(f1),float(acc)
    best=value(bias,gain)
    trace=[list(best)]
    coords=list(zip(*np.nonzero(np.array(support)>=16)))
    coords.sort(key=lambda ij:-support[ij[0]][ij[1]])
    # Single-coefficient changes are written and rescored against the actual path metric.
    for radius,scale in ((.36,.45),(.20,.25),(.12,.15)):
        before=best
        for rows in (list(diagonal_masks())+[(ij,) for ij in coords]):
            for weight,step,lo,hi in ((bias,radius,-.65,.65),(gain,scale,.35,1.65)):
                idx=tuple(np.asarray(rows).T)
                original=weight[idx].copy()
                chosen=original
                for delta in (-step,step):
                    weight[idx]=np.clip(original+delta,lo,hi)
                    test=value(bias,gain)
                    if test>best:
                        chosen=weight[idx].copy()
                        best=test
                weight[idx]=chosen
        trace.append(list(best))
        print(train_name,objective,"coordinate pass",radius,best,flush=True)
        if before==best:
            break
    return Edit(train_name+"-"+objective,bias,gain),trace


def diagonal_masks():
    yield tuple((i,i) for i in range(7))
    yield ((3,3),(4,4))
    yield tuple((i,5) for i in range(7) if i!=5)


def write_profile(path,parent,selected):
    t=parent[CRF].copy()
    t[3:,3:]+=selected.bias
    save_file({**parent,CRF:t,"edge.log_ratio":(parent["edge.log_ratio"]*selected.gain).astype("f4")},str(path))


def main():
    torch.set_num_threads(2)
    engine=load_decoder()
    parent=load_file(str(EDGE/"role-edge-exact.safetensors"))
    vocab=json.loads((EDGE/"vocab.json").read_text())
    model=HierModel(DONOR)
    train=edit_documents()
    vec=embeddings(model,train,"raw-emitter-train")
    training=[(prepare_score(d,model.logits(x),{"exact":parent}),transition_scores(d["text"],d["units"],vocab,parent))
              for d,x in zip(train,vec)]
    packed=Packed(training,parent,engine)
    run=packed.decode(ZERO,ONE)
    for (q,phi),r in zip(training,np.split(run,packed.sizes[1:-1])):
        np.testing.assert_array_equal(r,decode(q.logits["exact"],parent[CRF],phi,float(parent["edge.gain"])))
    np.testing.assert_allclose(packed.measure(ZERO,ONE),metric(packed.per_case(ZERO,ONE).sum(0)))
    direction,ref,obt=moments(packed,training)
    options=[Edit("parent",ZERO,ONE)]
    for a in (.1,.2,.35):
        options.append(Edit(f"moment-{a}",a*direction,ONE))
    training_fit={}
    for goal in ("exact","balanced"):
        e,trace=search(packed,ref,goal,"train")
        training_fit[e.name]=trace
        for amount in (.25,.5,.75,1.):
            options.append(Edit(f"{e.name}-{amount}",e.bias*amount,ONE+amount*(e.gain-ONE)))
    recipe=RESULTS/"role-pair-search-train.json"
    save_json(recipe,{"train_ids":[d["id"] for d in train],"train_units":packed.units,
                      "reference_counts":ref,"parent_counts":obt,"coordinate_trace":training_fit})
    print("Frozen training edits",len(options),packed.units,flush=True)
    groups=sample_whole()
    end=Packed(prepare(groups["select"],"select",model,{"exact":parent},vocab,parent),parent,engine)
    base=end.measure(ZERO,ONE)
    table={}
    for e in options:
        table[e.name]=end.measure(e.bias,e.gain)
    target={}
    for goal in ("exact","balanced"):
        target[goal]=max((e for e in options if goal=="exact" or table[e.name][1]>=base[1]),
                         key=lambda e:(table[e.name],e.name=="parent"))
        print("Validation",goal,target[goal].name,table[target[goal].name],base,flush=True)
    manifests={}
    for key,e in target.items():
        path=OUT/f"whole-role-pair-search-{key}.safetensors"
        write_profile(path,parent,e)
        manifests[key]={"name":e.name,"sha256":file_hash(path),"bias":e.bias.tolist(),"gain":e.gain.tolist()}
    freeze=RESULTS/"role-pair-search-frozen.json"
    save_json(freeze,{"options":table,"profiles":manifests,"selection_case_ids":[x["id"] for x in groups["select"]],
                      "train_sha256":file_hash(recipe)})
    used={d["id"] for cases in groups.values() for d in cases}
    excluded=set(json.loads((EDGE/"overlap.json").read_text())["independent_exclusions_for_text_disjoint_check"])
    groups["independent"]=[d for d in independent_validation(used) if d["id"] not in excluded]
    result={"method":"Finite coordinate search with sparse role-pair transition biases and lexical gains. Tune the actual exact-span output and optionally weight token accuracy. Warm starting from the existing deployed role pair weights. Train search uses 144 public training judgments, then selection chooses shrinkage on development. No gradients.",
            "frozen_sha256":file_hash(freeze),"profiles":manifests,"splits":{}}
    for split in ("select","confirm","independent","test"):
        if split=="test":
            groups[split]=[document(k,v) for k,v in gold_cases().items()]
        data=end if split=="select" else Packed(prepare(groups[split],split,model,{"exact":parent},vocab,parent),parent,engine)
        base=data.per_case(ZERO,ONE)
        result["splits"][split]={}
        for key,e in target.items():
            y=data.per_case(e.bias,e.gain)
            result["splits"][split][key]={"candidate":full_score(y),"parent":full_score(base),
                                         "cases":len(base),"gap_ci95":bootstrap(y,base)}
            print(split,key,metric(y.sum(0)),metric(base.sum(0)),flush=True)
        save_json(RESULTS/"role-pair-search.json",result)


if __name__=="__main__":
    main()
