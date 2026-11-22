"""Convert a three-output entailment checkpoint into a programmed seven-role readout."""
from argparse import ArgumentParser
import hashlib
import itertools
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file,save_file
import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer

from fragment_trial import passages
from hier_model import CRF,HierModel,path_labels
from hier_trial import DONOR,OUT
from program_trial import save_json
from seven_paths import ROOT
from span_eval import counts,summarize
from transition_span_edit import features
from weight_edit import file_hash

SOURCE=ROOT/"references/nli-minilm"
CACHE=ROOT/"cache/nli-span-minilm-v1"
HYPOTHESES=(
    "The selected text is a heading, case caption, list of appearances, or formal transition.",
    "The selected text recounts the facts or earlier procedural history of the case.",
    "The selected text identifies a legal issue or question that the court must resolve.",
    "The selected text presents the submissions of the petitioner or appellant.",
    "The selected text presents the submissions of the respondent or opposing party.",
    "The selected text gives the court's own analysis of the law or evidence.",
    "The selected text announces the court's operative ruling or disposition.",
)
NULL=(
    "The selected text discusses some material from a judgment.",
    "A section of a court judgment is shown here.",
    "The selected text is part of the written opinion.",
)

def limited(s,n):
    words=s.split()
    if len(words)<=n:return " ".join(words)
    a=n//2
    return " ".join(words[:a]+words[-(n-a):])

def inputs(d,context):
    snippets=[d["text"][a:b].strip() for a,b in d["units"]]
    for i,x in enumerate(snippets):
        if not context:
            yield limited(x,110)
        else:
            before=limited(snippets[i-1],30) if i else ""
            after=limited(snippets[i+1],30) if i+1<len(snippets) else ""
            yield f"Previous text: {before}\nSelected text: {limited(x,90)}\nFollowing text: {after}"


class Entailment:
    def __init__(self):
        self.runtime=None
        self.tokenizer=AutoTokenizer.from_pretrained(SOURCE,local_files_only=True)
        self.model=AutoModelForSequenceClassification.from_pretrained(SOURCE,local_files_only=True).eval()
        labels={str(v).lower():int(k) for k,v in self.model.config.id2label.items()}
        self.order=[labels[x] for x in ("contradiction","neutral","entailment")]
        self.parameters=sum(p.numel() for p in self.model.parameters())

    @torch.inference_mode()
    def encode(self,items):
        first=[x for x in items for _ in HYPOTHESES]
        second=list(HYPOTHESES)*len(items)
        tok=self.tokenizer(first,second,add_special_tokens=True,truncation="only_first",max_length=224)
        lengths=list(map(len,tok["input_ids"]));order=sorted(range(len(first)),key=lengths.__getitem__)
        outputs=np.empty((len(first),3),dtype=np.float32)
        for i in range(0,len(first),16):
            here=order[i:i+16]
            vals=[{k:v[j] for k,v in tok.items()} for j in here]
            batch=self.tokenizer.pad(vals,padding=True,return_tensors="pt")
            if self.runtime is not None:
                pred=self.runtime({k:batch[k].numpy() for k in ("input_ids","attention_mask")})[0]
            else:
                pred=self.model(**batch).logits.numpy()
            outputs[here]=pred[:,self.order]
        return outputs.reshape(len(items),7,3)


def vectors(model,docs,context):
    base=CACHE/("context" if context else "isolated")
    base.mkdir(parents=True,exist_ok=True)
    outputs={}
    todo=[]
    for key,d in docs.items():
        sig=hashlib.sha256(d["text"].encode()).hexdigest()
        path=base/(sig+".npy")
        if path.exists():outputs[key]=np.load(path,allow_pickle=False)
        else:todo.append((key,d,path))
    for i in range(0,len(todo),6):
        block=todo[i:i+6];texts=[list(inputs(d,context)) for _,d,_ in block]
        res=model.encode(list(itertools.chain.from_iterable(texts)))
        off=0
        for (k,d,path),t in zip(block,texts):
            matrix=res[off:off+len(t)];outputs[k]=matrix;off+=len(t)
            stage=path.with_suffix(".part")
            with stage.open("wb") as f:np.save(f,matrix,allow_pickle=False)
            stage.replace(path)
        print("contextual" if context else "isolated",min(i+6,len(todo)),"/",len(todo),flush=True)
    return outputs


def main():
    ap=ArgumentParser(description=__doc__)
    ap.add_argument("--threads",type=int,default=2)
    ap.add_argument("--openvino",action="store_true")
    ap.add_argument("--parity-only",action="store_true")
    cfg=ap.parse_args()
    torch.set_num_threads(cfg.threads)
    model=Entailment()
    if cfg.openvino:
        import openvino as ov
        folder=ROOT/"runtime/nli-span-minilm"
        graph=ov.Core().read_model(folder/"model.xml")
        compiled=ov.Core().compile_model(graph,"GPU",{"INFERENCE_PRECISION_HINT":"f32","CACHE_DIR":str(folder/"compiled")})
        probes=list(HYPOTHESES)+["Earlier substantive proceedings are relevant. " * 70]+list(NULL)
        parent=model.encode(probes)
        model.runtime=compiled
        actual=model.encode(probes)
        diff=float(np.max(np.abs(parent-actual)))
        print("Backend parity maximum absolute NLI logit difference",diff,flush=True)
        if diff>.003 or not np.all(parent.argmax(2)==actual.argmax(2)):
            raise ValueError("NLI OpenVINO critical parity criteria not met")
        save_json(ROOT/"results/nli-accelerated-parity.json",{"input_items":len(probes),"max_absolute_logit_difference":diff,"same_three_way_argmax":True})
    if cfg.parity_only:return
    groups=passages()
    # Fixed engineering screen. If this family is competitive, extend to the complete tests with frozen weights.
    cohorts={k:v[:{"select":30,"confirm":30,"test":54,"cloud":12}[k]] for k,v in groups.items()}
    bank={f"{k}/{d['id']}":d for k,ds in cohorts.items() for d in ds}
    out={mode:vectors(model,bank,bool(mode)) for mode in (0,1)}
    prior=model.encode(NULL).mean(0)
    hier=HierModel(DONOR)
    emissions={fold:features(hier,docs) for fold,docs in cohorts.items()}
    original=load_file(str(OUT/"fragment.safetensors"))
    if not np.isclose(original["classifier.weight"][:,200:],0).all():
        raise ValueError("Incumbent fragment needs independent features")
    t=original["transition"]
    def score(fold,mode,w,b,mix,transition):
        hits=[]
        for d,reference in zip(cohorts[fold],emissions[fold]):
            q=out[mode][f"{fold}/{d['id']}"]
            decoded=np.einsum("lrc,rc->lr",q,w)+b
            vals=(1-mix)*reference+mix*decoded
            hits.append(counts(d,d["units"],path_labels(vals,transition)))
        return summarize(np.sum(hits,axis=0))
    all_scores=[];best=None;pure=None
    for mode,neg,neutral,debias,mix,sticky in itertools.product((0,1),(0.,.5),(0.,.5),(0.,.75),(.25,.5,1.),(.5,1.)):
        w=np.tile(np.array([-neg,-neutral,1.],np.float32),(7,1))
        b=-debias*np.einsum("lc,lc->l",prior,w)
        transition=t.copy();transition[3:,3:]*=sticky
        q=score("select",mode,w,b,mix,transition)
        rank=q["exact_labeled_spans"]["f1"],q["token_accuracy"]
        spec=(mode,neg,neutral,debias,mix,sticky)
        all_scores.append([*spec,*rank])
        record=(rank,spec,w,b,transition)
        if best is None or rank>best[0]:best=record
        if mix==1 and (pure is None or rank>pure[0]):pure=record
    save={}
    for key,record in (("selected",best),("standalone",pure)):
        _,settings,w,b,tr=record;mode,_,_,_,mix,_=settings
        p=OUT/f"nli-fragment-{key}.safetensors"
        save_file({"role_nli_weights":(w*mix).astype("f4"),"role_nli_bias":(b*mix).astype("f4"),"crf.transitions":tr,
                   "hierarchy_mix":np.array(1-mix,np.float32)},str(p))
        save[key]={"settings":list(settings),"sha256":file_hash(p),"scores":{}}
        for fold,ds in cohorts.items():
            q=score(fold,mode,w,b,mix,tr);r=score(fold,mode,w,b,0,t)
            save[key]["scores"][fold]={"cases":len(ds),"candidate":q,"incumbent":r}
        print(key,save[key]["scores"]["test"]["candidate"]["exact_labeled_spans"]["f1"],flush=True)
    save_json(ROOT/"results/nli-minilm-span-edit.json",{"parameters":model.parameters,"backend":"openvino-source" if cfg.openvino else "torch",
        "source_revision":"b95119ce93d3e065de6214e38cd4a97b0f2f2c6d","hypotheses":HYPOTHESES,"grid":all_scores,"model_profiles":save,
        "method":"Directly write seven comparisons of the pinned entailment head's outputs. The score maps from contradiction, neutral and entailment into one scalar per role. Compare sentence input to explicitly marked local context. Null passages estimate fixed bias. No gradients, API supervision, or gold role information in inputs."})
if __name__=="__main__":
    main()
