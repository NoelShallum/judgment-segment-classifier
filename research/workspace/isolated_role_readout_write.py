"""Fit CPU sentence-only role weights without changing the frozen backbone."""
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np
from safetensors.numpy import save_file

from sentence_ngram_residual_write import STEM as NGRAM,OUT,ROOT,VECTORIZER,load_file,file_hash
from sentence_document_graph_write import (
    EDGE,DONOR,HierModel,cm,paired,row_summary,save_json,torch,
    independent_validation,sample_whole,training_documents,
)
from confirm_sentence_graph_holdout import gather_validation
from hier_full import CACHE,gold_cases
from hier_model import CRF,EMITTER
from role_cues import cue_features
from span_eval import document

TAG="isolated-role-readout"
LEGACY=OUT/"excerpt.safetensors"
MODES=("states","lexical","combined")


def design(text,stat,table):
    x=VECTORIZER.transform(text)
    strength=(x@table)/np.sqrt(np.maximum(np.diff(x.indptr),1))[:,None]
    tokens=np.asarray([len(n.split()) for n in text])
    cues=cue_features(text)
    surface=np.column_stack((np.log1p(tokens),np.log1p([len(t) for t in text]),cues))
    return (np.column_stack((stat,surface)),np.column_stack((strength,surface)),
            np.column_stack((stat,strength,surface)))


def block(args):
    docs,official=args
    torch.set_num_threads(1)
    model=HierModel(DONOR)
    orig=load_file(str(LEGACY))
    table=load_file(str(OUT/(NGRAM+"-memory.safetensors")))["lexical"][0]
    output=[]
    for d in docs:
        words=[d["text"][a:b].strip() for a,b in d["reference_units"]]
        if official:
            doc,mask=d["id"],d["source_mask"]
            with np.load(CACHE/(doc+".npz")) as f:
                assert len(f["excerpt"])==len(mask)
                h=f["excerpt"][mask]
        else:
            h=model.contextualize(model.words(words),independent=True)
        p=(h@orig[EMITTER+"weight"][3:].T+orig[EMITTER+"bias"][3:]+orig[CRF][1,3:]+orig[CRF][3:,2]).argmax(1)
        output.append((d["id"],np.asarray(d["reference_roles"]),p,design(words,h,table)))
    return output


def prepare(docs,official=False):
    with ProcessPoolExecutor(4) as pool:
        parts=list(pool.map(block,[(docs[i::4],official) for i in range(4)]))
    index={k:(k,y,p,x) for group in parts for k,y,p,x in group}
    return [index[d["id"]] for d in docs]


def fits(train,index):
    x=np.vstack([d[3][index] for d in train])
    mean=x.mean(0)
    scale=1/np.maximum(x.std(0),.035)
    z=(x-mean)*scale
    y=np.concatenate([d[1] for d in train])
    p=np.eye(7)[y]
    mass=p.sum(0)
    freq=mass/mass.sum()
    means=(p.T@z)/mass[:,None]
    residual=z-means[y]
    sigma=residual.T@residual/len(z)
    for reg in (.05,.25):
        precision=np.linalg.solve(sigma+np.eye(sigma.shape[0])*sigma.trace()/len(sigma)*reg,means.T)
        p0=-.5*np.sum(means*precision.T,axis=1)
        yield ("lda",reg,0.),{"center":mean,"scale":scale,"weight":precision,"bias":p0,"prior":np.log(freq)}
    a=np.column_stack((z,np.ones(len(z))))
    for rarity in (.0,.28,.56):
        w=freq[y]**-rarity
        gram=a.T@(a*w[:,None])
        target=a.T@(p*w[:,None])
        for reg in (.06,.3):
            coef=np.linalg.solve(gram+np.diag(np.r_[np.ones(z.shape[1]),.05]*reg*gram.trace()/len(gram)),target)
            yield ("ridge",reg,rarity),{"center":mean,"scale":scale,"weight":np.ascontiguousarray(coef[:-1]),
                                       "bias":np.ascontiguousarray(coef[-1]),"prior":np.log(freq)}


def prediction(rows,mode,weights,method,adjustment):
    i=MODES.index(mode)
    # Uniform addition to all seven scores cancels in the selected label.
    out=[]
    for k,y,b,x in rows:
        p=((x[i]-weights["center"])*weights["scale"])@weights["weight"]+weights["bias"]
        if method=="ridge":
            p=np.log(np.maximum(p,1.e-7))
        out.append((p+adjustment*weights["prior"]).argmax(1))
    return out


def scores(rows,labels):
    return np.stack([cm(row[1],y) for row,y in zip(rows,labels)])


def main():
    torch.set_num_threads(1)
    root=ROOT/"results"
    legacy=json.loads((root/"hierarchical-official-test.json").read_text())
    assert legacy["excerpt_sha256"]==file_hash(LEGACY)
    frozen=json.loads((root/(NGRAM+"-frozen.json")).read_text())
    assert frozen["memory_sha256"]==file_hash(OUT/(NGRAM+"-memory.safetensors"))
    lexical_cases=set(frozen["memory_training_cases"])
    train=[v for v in training_documents() if v["id"] not in lexical_cases]
    print("Preparing isolated training rows",len(train),flush=True)
    fit=prepare(train)
    options={(name,*spec):w for i,name in enumerate(MODES) for spec,w in fits(fit,i)}
    print("Isolated heads solved",sum(len(d[1]) for d in fit),flush=True)
    del fit
    groups=sample_whole()
    dev=prepare(groups["select"])
    m=baseline=row_summary(scores(dev,[d[2] for d in dev]).sum(0))
    winner=(m["macro_f1"],m["accuracy"],("legacy",))
    table=[]
    for (name,method,reg,rarity),w in options.items():
        grid=(.0,.2,.6,1.) if method=="lda" else (-.22,0.,.22,.45)
        for adj in grid:
            y=prediction(dev,name,w,method,adj)
            m=row_summary(scores(dev,y).sum(0))
            choice=(name,method,reg,rarity,adj)
            table.append([choice,m["macro_f1"],m["accuracy"]])
            if m["accuracy"]>=baseline["accuracy"]-.01:
                winner=max(winner,(m["macro_f1"],m["accuracy"],choice))
    choice=winner[-1]
    selected=None
    written=None
    if choice[0]!="legacy":
        target=OUT/(TAG+".safetensors")
        selected=options[choice[:4]]
        save_file(selected,str(target))
        got=load_file(str(target))
        assert all(np.array_equal(v,got[k]) for k,v in selected.items())
        selected=got
        written=file_hash(target)
    receipt=root/(TAG+"-frozen.json")
    save_json(receipt,{"choice":choice,"grid":table,"training_cases":[d["id"] for d in train],
             "selection_cases":[d["id"] for d in groups["select"]],"weights_sha256":written,
             "legacy_weights_sha256":file_hash(LEGACY),"lexical_weights_sha256":frozen["memory_sha256"]})
    print("Frozen isolated readout",choice,winner[:2],(baseline["macro_f1"],baseline["accuracy"]),flush=True)
    val=ROOT/"references/official/val.csv"
    hold_path=root/"sentence-graph-holdout-frozen.json"
    hold=json.loads(hold_path.read_text())
    assert hold["val_csv_sha256"]==file_hash(val)
    records=gather_validation(val)
    groups["holdout"]=[document(k,records[k][0]) for k in hold["cases"]]
    result={"freeze_sha256":file_hash(receipt),"holdout_sha256":file_hash(hold_path),"splits":{}}
    for split in ("select","confirm","holdout","test"):
        if split=="test":
            docs=[]
            for k,v in gold_cases().items():
                d=document(k,v)
                d["source_mask"]=np.asarray([bool(txt.strip()) for _,txt,_ in v])
                docs.append(d)
            rows=prepare(docs,official=True)
        else:
            rows=dev if split=="select" else prepare(groups[split])
        base=[d[2] for d in rows]
        use=base if selected is None else prediction(rows,choice[0],selected,choice[1],choice[-1])
        a,b=scores(rows,use),scores(rows,base)
        q=result["splits"][split]={"cases":len(rows),"candidate":row_summary(a.sum(0)),
                 "baseline":row_summary(b.sum(0)),"paired":paired(a,b)}
        if split=="test":
            assert q["baseline"]==legacy["row_classification"]["excerpt_programmed"]
        if split=="holdout":
            assert q["baseline"]["rows"]==hold["sentences"]
        save_json(root/(TAG+".json"),result)
        print(split,q["candidate"]["macro_f1"],q["baseline"]["macro_f1"],q["candidate"]["accuracy"],flush=True)


if __name__=="__main__":
    main()
