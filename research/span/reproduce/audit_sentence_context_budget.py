"""Measure supplied-sentence profiles on identical rows with controlled context."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np

from confirm_sentence_graph_holdout import gather_validation
from sentence_document_graph_write import (
    STEM,OUT,ROOT,EDGE,PARENT,DONOR,HierModel,torch,document,load_file,file_hash,row_summary,cm,paired,save_json,
)
from role_cues import cue_features
from role_edge_runtime import transition_scores
from predict_role_edge import decode_profile,infer_sentences
from sentence_joint_runtime import joint_labels
from sentence_local_readout_runtime import labels
from hier_model import LABELS

NAME="sentence-context-budget"
WINDOWS=(1,4,16,64,None)
PROFILES={
    "sentence-joint":PARENT,
    "sentence-local":OUT/"sentence-local-readout-portable.safetensors",
    "sentence-document":OUT/"sentence-document-readout-portable.safetensors",
}


def capped_states(model,words,bounds):
    groups=defaultdict(list)
    for a,b in bounds:
        groups[b-a].append((a,b))
    bank=np.empty((len(words),model.doc.hidden_size*2),np.float32)
    with torch.inference_mode():
        for n,spans in groups.items():
            for j in range(0,len(spans),256):
                part=spans[j:j+256]
                x=torch.stack([words[a:b] for a,b in part])
                h,_=model.doc(x)
                for (a,b),z in zip(part,h.numpy()):
                    bank[a:b]=z
    return bank


def emissions(text,bounds,raw,cues,states,lengths,weights,vocab):
    coord=(np.asarray(bounds).mean(1)-bounds[0][0])/(bounds[-1][1]-bounds[0][0])
    p=np.eye(5)[np.digitize(coord,[.05,.25,.60,.91])]
    base=weights["sentence-joint"]
    logits=raw+cues@base["cue_head"]+p@base["position_head"]
    edge=transition_scores(text,bounds,vocab,base)
    gain=float(base["edge.gain"])
    first=decode_profile(logits,edge,base,gain)
    args=(logits,edge)
    tail=(gain,first,lengths,states)
    return {name:(joint_labels if name=="sentence-joint" else labels)(*args,w,*tail)
            for name,w in weights.items()}


def process(args):
    docs,live=args
    torch.set_num_threads(1)
    model=HierModel(DONOR)
    vocab=json.loads((EDGE/"vocab.json").read_text())
    weights={k:load_file(str(v)) for k,v in PROFILES.items()}
    common=weights["sentence-joint"]
    for name,w in weights.items():
        assert all(np.array_equal(t,w[k]) for k,t in common.items())
    audit=[]
    source=ROOT/"cache/known-sentence-pair-edit"
    matrices={str(width):{name:[] for name in PROFILES} for width in WINDOWS}
    for d in docs:
        t=d["text"]
        spans=d["reference_units"]
        pieces=[t[a:b].strip() for a,b in spans]
        cues=cue_features([t[a:b] for a,b in spans])
        lengths=np.asarray([len(x) for x in pieces])
        y=np.asarray(d["reference_roles"])
        words=model.words(pieces)
        for width in WINDOWS:
            if width is None:
                states=np.load(source/(d["id"]+".npy"),allow_pickle=False)
                blocks=[(0,len(pieces))]
            else:
                i=range(0,len(pieces),width)
                blocks=[(a,min(a+width,len(pieces))) for a in i]
                states=capped_states(model,words,blocks)
            raw=model.logits(states)
            stats={name:np.zeros((7,7),np.int64) for name in PROFILES}
            for a,b in blocks:
                offset=spans[a][0]
                # Strip the delimiter after the last retained sentence, as the CLI does.
                end=spans[b-1][0]+len(pieces[b-1])
                sub=[(x-offset,min(y0,end)-offset) for x,y0 in spans[a:b]]
                use=emissions(t[offset:end],sub,raw[a:b],cues[a:b],states[a:b],lengths[a:b],weights,vocab)
                if d["id"] in live and a==0:
                    for name,prediction in use.items():
                        w=weights[name]
                        expected=infer_sentences(pieces[a:b],model,w,vocab,w,"joint",float(w["edge.gain"]))
                        assert expected["labels"]==[LABELS[j] for j in prediction],(d["id"],width,name)
                    audit.append([d["id"],width,b-a])
                for name,prediction in use.items():
                    stats[name]+=cm(y[a:b],prediction)
            for name,s in stats.items():
                matrices[str(width)][name].append(s)
    return [d["id"] for d in docs],matrices,audit


def main():
    root=ROOT/"results"
    source=root/"sentence-graph-holdout-frozen.json"
    prior=root/"sentence-graph-holdout.json"
    fixed=json.loads(source.read_text())
    hold=json.loads(prior.read_text())
    assert hold["freeze_sha256"]==file_hash(source)
    data=ROOT/"references/official/val.csv"
    assert fixed["val_csv_sha256"]==file_hash(data)
    table=gather_validation(data)
    docs=[document(key,table[key][0]) for key in fixed["cases"]]
    check_live={docs[i]["id"] for i in (0,100,200,300)}
    print("Matched context audit",len(docs),fixed["sentences"],flush=True)
    nworkers=4
    with ProcessPoolExecutor(nworkers) as pool:
        parts=list(pool.map(process,[(docs[i::nworkers],check_live) for i in range(nworkers)]))
    index={key:i for i,key in enumerate(x for ids,_,_ in parts for x in ids)}
    order=[index[x] for x in fixed["cases"]]
    out={}
    scores={}
    for width in reversed(WINDOWS):
        key=str(width)
        out[key]={}
        for name in PROFILES:
            m=np.concatenate([p[key][name] for _,p,_ in parts])[order]
            scores[key,name]=m
            q=out[key][name]={"scores":row_summary(m.sum(0))}
            assert q["scores"]["rows"]==fixed["sentences"]
            if width is None:
                expect=hold["baseline"] if name=="sentence-joint" else hold["arms"][{"sentence-local":"local","sentence-document":"two-hop"}[name]]["scores"]
                assert q["scores"]==expect,name
            else:
                q["paired_to_full"]=paired(m,scores["None",name])
        print(width,[(k,v["scores"]["macro_f1"],v["scores"]["accuracy"]) for k,v in out[key].items()],flush=True)
    live=[x for _,_,q in parts for x in q]
    assert len(live)==len(WINDOWS)*len(check_live)
    save_json(root/(NAME+".json"),{"holdout_sha256":file_hash(source),"cases":len(docs),"source_sentences":fixed["sentences"],
             "profiles_sha256":{k:file_hash(v) for k,v in PROFILES.items()},"scores":out,"checked_live_chunks":live,
             "protocol":"Re-encode contiguous disjoint blocks of up to 1, 4, 16 or 64 supplied sentences as independent inputs. Score the same original sentence rows and compare with full-judgment context. Excerpts preserve source boundaries and need not coincide with author paragraphs."})


if __name__=="__main__":
    main()
