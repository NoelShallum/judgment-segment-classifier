"""Gradient-free seven-role classifiers on public encoders with gold-blind document context."""
import argparse
from collections import defaultdict
import csv
import hashlib
import itertools
from pathlib import Path
import random

import numpy as np
from safetensors.numpy import load_file, save_file
from scipy.linalg import solve
import torch
from transformers import AutoModel, AutoTokenizer

from fragment_trial import passages
from hier_model import CRF, HierModel, path_labels
from hier_trial import DONOR, OUT as HIER
from program_trial import fixture_rows, save_json
from role_fixtures import ANCHORS, CHECKS
from seven_paths import LABELS, ROOT
from span_eval import counts, sets, summarize
from weight_edit import file_hash

MODELS = {
    "gte": ("Alibaba-NLP/gte-modernbert-base", "e7f32e3c00f91d699e8c43b53106206bcc72bb22"),
    "granite": ("ibm-granite/granite-embedding-small-english-r2", "2ab6fa8ea2d674564defd37171ae19079b864b33"),
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"),
    "deberta": ("cross-encoder/nli-deberta-v3-xsmall", "a150876415327c80daeff35ca6f68f5ed8cf5c24"),
}
CACHE = ROOT / "cache/encoder-span-edit-v2"
CHUNK = 384
MODES = ("isolated", "bidirectional")
NEUTRAL_SAMPLE = 192


def normalized(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def chunk_units(token_units, context, cap):
    if not context:
        return [(tokens[i:i+cap], [(j, 0, len(tokens[i:i+cap]))])
                for j, tokens in enumerate(token_units) for i in range(0, len(tokens), cap)]
    stream = list(itertools.chain.from_iterable(token_units))
    offsets = np.cumsum([0] + list(map(len, token_units)))
    # Each unit is assigned to one or more disjoint blocks with a surrounding halo.
    core = cap // 2
    chunks = []
    for start in range(0, len(stream), core):
        end = min(len(stream), start+core)
        left = max(0, min(start-(cap-core)//2, len(stream)-cap))
        right = min(len(stream), left+cap)
        first = np.searchsorted(offsets, start, side="right")-1
        stop = np.searchsorted(offsets, end, side="left")
        pools = [(j, max(start,offsets[j])-left, min(end,offsets[j+1])-left)
                 for j in range(first,stop)]
        chunks.append((stream[left:right], pools))
    return chunks


class Encoder:
    def __init__(self, key, backend="torch"):
        model, revision = MODELS[key]
        local = {"gte": "gte-modernbert", "deberta": "deberta-xsmall"}
        source = ROOT / "references" / local[key] if key in local else model
        self.tok = AutoTokenizer.from_pretrained(source, revision=revision, local_files_only=True)
        attention = "eager" if key == "deberta" else "sdpa"
        self.model = AutoModel.from_pretrained(source, revision=revision, local_files_only=True,
                                               attn_implementation=attention).float().eval()
        self.runtime = None
        if backend in ("openvino","openvino-source"):
            if key != "gte" and backend == "openvino":
                raise ValueError("The public OpenVINO-compatible ONNX graph is for GTE")
            from modernbert_openvino import ModernBertOpenVINO
            self.runtime = ModernBertOpenVINO(ROOT,from_source=backend=="openvino-source",key=key)
        self.width = self.model.config.hidden_size
        self.params = sum(x.numel() for x in self.model.parameters())
        probe = self.tok("judgment", return_special_tokens_mask=True)
        plain = np.flatnonzero(np.asarray(probe["special_tokens_mask"]) == 0)
        self.prefix = probe["input_ids"][:plain[0]]
        self.suffix = probe["input_ids"][plain[-1]+1:]
        self.content_start = len(self.prefix)
        self.cap = CHUNK-len(self.prefix)-len(self.suffix)

    @torch.inference_mode()
    def process(self, items, mode):
        groups = [self.tok(ss, add_special_tokens=False, verbose=False)["input_ids"] for ss in items]
        shapes = [len(x) for x in groups]
        accum = np.zeros((sum(shapes),self.width),np.float32)
        mass = np.zeros(sum(shapes),np.int64)
        jobs = []
        first = 0
        for group in groups:
            group = [ids or [self.tok.unk_token_id] for ids in group]
            for ids, pools in chunk_units(group, mode=="bidirectional", self.cap):
                jobs.append((ids, [(u+first,l,r) for u,l,r in pools]))
            first += len(group)
        jobs.sort(key=lambda j:len(j[0]))
        for offset in range(0,len(jobs),12):
            batch = jobs[offset:offset+12]
            formatted = [{"input_ids": self.prefix + ids + self.suffix} for ids,_ in batch]
            inputs = self.tok.pad(formatted, padding=True, return_tensors="pt")
            if self.runtime is not None:
                out = self.runtime(inputs)
            else:
                out = self.model(**inputs).last_hidden_state.numpy()
            for z, (ids, pools) in zip(out,batch):
                states=z[self.content_start:self.content_start+len(ids)]
                for i,a,b in pools:
                    accum[i] += states[a:b].sum(0)
                    mass[i] += b-a
        if np.any(mass==0):
            raise RuntimeError("Token coverage gap")
        return np.split(normalized(accum/mass[:,None]).astype(np.float32), np.cumsum(shapes)[:-1])

    def cache(self, bank, mode):
        dest = CACHE / mode
        dest.mkdir(parents=True, exist_ok=True)
        outputs = {}
        todo = []
        for name,seq in bank.items():
            sig=hashlib.sha256(repr(seq).encode()).hexdigest()
            target=dest/(sig+".npy")
            if target.exists():
                outputs[name]=np.load(target,allow_pickle=False)
            else:
                todo.append((name,seq,target))
        for start in range(0,len(todo),24):
            group=todo[start:start+24]
            values=self.process([seq for _,seq,_ in group],mode)
            for (name,_,path),value in zip(group,values):
                temp=path.with_suffix(".part")
                with temp.open("wb") as f: np.save(f,value,allow_pickle=False)
                temp.replace(path)
                outputs[name]=value
            print(mode, min(start+24,len(todo)), "/",len(todo),flush=True)
        return outputs


def make_documents():
    whole = sets()
    clipped = passages()
    result = {}
    for task, groups in (("whole",whole),("fragment",clipped)):
        result[task] = {k:{str(d["id"]):d for d in ds} for k,ds in groups.items()}
    bank = {f"{task}/{kind}/{key}":[d["text"][a:b] for a,b in d["units"]]
            for task,groups in result.items() for kind,ds in groups.items() for key,d in ds.items()}
    return result,bank


def neutral_texts():
    rng = random.Random(64458)
    reservoir = []
    with (ROOT/"references/official/train.csv").open(newline="",encoding="utf-8-sig") as stream:
        n=0
        for r in csv.DictReader(stream):
            t=r["Text"].strip()
            if not 6<=len(t.split())<=48: continue
            n+=1
            if len(reservoir)<NEUTRAL_SAMPLE: reservoir.append(t)
            else:
                k=rng.randrange(n)
                if k<NEUTRAL_SAMPLE: reservoir[k]=t
    return reservoir


def candidate_heads(keys, roles, neutral):
    center = keys.mean(0)
    x = keys-center
    rows = normalized(np.stack([keys[roles==j].mean(0) for j in range(7)]))
    yield "prototypes",rows,np.zeros(7)
    for retained in (0,16,48):
        _,_,axes=np.linalg.svd(neutral-neutral.mean(0),full_matrices=False)
        axes=axes[:retained]
        projected=x-(x@axes.T)@axes
        desired=np.eye(7)[roles]-(keys@rows.T)
        gram=projected@projected.T
        for damping in (.1,.5):
            delta=solve(gram+np.eye(len(keys))*damping*(gram.trace()/len(keys)),desired,assume_a="pos").T@projected
            for gain in (.5,1.):
                yield f"association_r{retained}_d{damping}_g{gain}",rows+gain*delta,-gain*delta@center


def patch_for(task, transitions, scale):
    base=transitions.copy()
    base[3:,3:] *= scale
    base[1,3:] *= (1. if task=="whole" else 0.) * scale
    base[3:,2] *= (1. if task=="whole" else 0.) * scale
    return base


def evaluate(modelname, groups, banks, heads):
    model=HierModel(DONOR)
    reports={}
    out=ROOT/"checkpoints"/("encoder-span-"+modelname)
    out.mkdir(exist_ok=True,parents=True)
    for task, splits in groups.items():
        existing=load_file(str(HIER/(("document" if task=="whole" else "fragment")+".safetensors")))
        base_t=existing[CRF] if task=="whole" else existing["transition"]
        donor_tr=model.transitions
        ref_scores={}
        for fold,docs in splits.items():
            for key,d in docs.items():
                units=[d["text"][a:b] for a,b in d["units"]]
                words=model.words(units)
                ctxt=model.contextualize(words)
                if task=="whole":
                    ref=model.logits(ctxt)
                else:
                    x=np.column_stack((ctxt,model.contextualize(words,independent=True)))
                    ref=x@existing["classifier.weight"].T+existing["classifier.bias"]
                ref_scores[fold,key]=ref
        def collect(split, mode, w, b, tr, share):
            rows=[]
            for key,d in splits[split].items():
                ref=ref_scores[split,key]
                x=banks[mode][f"{task}/{split}/{key}"] if share else None
                logits=(1-share)*ref+share*(x@w.T+b) if share else ref
                rows.append(counts(d,d["units"],path_labels(logits,tr)))
            return np.stack(rows)
        baseline={k:collect(k,"isolated",None,None,base_t,0.) for k in splits}
        rows=[]
        best=None
        best_pure=None
        settings=((mode, row, share, scale) for mode in MODES
                  for row,share,scale in itertools.product(heads[mode],(.25,.5,1.),(0.,.5,1.))) if isinstance(heads,dict) else itertools.product(MODES,heads,(.25,.5,1.),(0.,.5,1.))
        for mode,(name,w,b),share,scale in settings:
            tr=patch_for(task,donor_tr,scale)
            values=collect("select",mode,10*w,10*b,tr,share)
            m=summarize(values.sum(0))
            score=(m["exact_labeled_spans"]["f1"],m["token_accuracy"])
            rows.append([mode,name,share,scale,*score])
            record=(score,mode,name,share,scale,10*w,10*b,tr)
            if best is None or score>best[0]:best=record
            if share==1 and (best_pure is None or score>best_pure[0]):best_pure=record

        def evaluate_profile(tag, record):
            _,mode,name,share,scale,w,b,tr=record
            dest=out/(task+"-"+tag+".safetensors")
            save_file({"classifier.weight":(share*w).astype("f4"),"classifier.bias":(share*b).astype("f4"),
                       "crf.transitions":tr,"donor_emission_scale":np.array(1-share,np.float32)},str(dest))
            report={"mode":mode,"edit":name,"new_encoder_fraction":share,"transition_scale":scale,"weights_sha256":file_hash(dest),"scores":{}}
            for split,docs in splits.items():
                q=collect(split,mode,w,b,tr,share);p=baseline[split];rng=np.random.default_rng(1181)
                draws=[]
                for ix in rng.integers(len(q),size=(1500,len(q))):
                    a=summarize(q[ix].sum(0));z=summarize(p[ix].sum(0))
                    draws.append(a["exact_labeled_spans"]["f1"]-z["exact_labeled_spans"]["f1"])
                report["scores"][split]={"cases":len(q),"candidate":summarize(q.sum(0)),"hierarchy":summarize(p.sum(0)),
                                         "exact_span_gap_95pct":np.quantile(draws,[.025,.975]).tolist()}
            return report
        reports[task]={"grid":rows,"best":evaluate_profile("selected",best),"standalone_encoder":evaluate_profile("pure",best_pure)}
    return reports


def main():
    global CACHE
    ap=argparse.ArgumentParser()
    ap.add_argument("encoder",choices=MODELS)
    ap.add_argument("--threads",type=int,default=3)
    ap.add_argument("--backend",choices=("torch","openvino","openvino-source"),default="torch")
    ap.add_argument("--parity-only",action="store_true")
    args=ap.parse_args()
    torch.set_num_threads(args.threads)
    CACHE = CACHE / (args.encoder if args.backend=="torch" else args.encoder+"-gpu")
    enc=Encoder(args.encoder, args.backend)
    if enc.runtime is not None:
        from engine_parity_span import compare
        compare(enc,args.encoder)
    if args.parity_only:return
    groups,bank=make_documents()
    keys=fixture_rows(ANCHORS)+fixture_rows(CHECKS)
    roles=np.asarray([r[3] for r in keys])
    authors={"keys":[r[2] for r in keys],"unlabeled":neutral_texts()}
    diag=enc.cache(authors,"isolated")
    heads=list(candidate_heads(diag["keys"],roles,diag["unlabeled"]))
    vectors={mode:enc.cache(bank,mode) for mode in MODES}
    print("Validation selection",flush=True)
    results=evaluate(args.encoder,groups,vectors,heads)
    record={"model":MODELS[args.encoder],"parameters":enc.params,"backend":args.backend,"chunk_limit_including_special_tokens":CHUNK,
            "authored_keys":len(keys),"unlabeled_preservation_examples":NEUTRAL_SAMPLE,
            "method":"Seven-role linear tensor from authored prototypes; centered ridge associations optionally restricted against principal directions of a fixed unlabeled LegalSeg train sample. Physical readout plus CRF tensors; no gradient steps. Block bidirectional context compared to isolated embeddings. Development exact span F1 selects one pure encoder and one eligible blend per input contract. Earlier results motivated these families.",
            "tasks":results}
    save_json(ROOT/"results"/("encoder-span-"+args.encoder+("" if args.backend=="torch" else "-accelerated")+".json"),record)
    for k,t in results.items():
        for n in ("best","standalone_encoder"):
            m=t[n]["scores"]["test"]
            print(k,n,t[n]["mode"],t[n]["edit"],m["candidate"]["exact_labeled_spans"]["f1"],m["hierarchy"]["exact_labeled_spans"]["f1"],flush=True)

if __name__=="__main__":
    main()
