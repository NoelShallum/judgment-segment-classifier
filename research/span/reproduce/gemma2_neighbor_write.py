"""Test within-judgment context with frozen EmbeddingGemma 2 and associative output weights."""
import argparse
from contextlib import contextmanager
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import socket
import subprocess
from time import perf_counter,sleep
from urllib.error import URLError

import numpy as np
from safetensors.numpy import save_file,load_file

from probe_embeddinggemma2_q8 import MODEL,PROMPT,request
from embeddinggemma_routed_write import (
    DONOR,EDGE,OUT,PARENT,ROOT,HierModel,aligned_embeddings,case_inputs,edit_documents,
    embeddings,file_hash,independent_validation,measure,sample_whole,save_json,torch,cm,
    predict,summarize,
)

STEM="gemma2-neighbor"
LIMIT=3
WIDTH=128
KINDS=("hierarchy","target","neighbors")
BUCKET=32
FORMAT="neighbor-window-v1"


@contextmanager
def backend(binary):
    revision=json.loads((MODEL/"provenance.json").read_text())
    source=MODEL/revision["file"]
    assert file_hash(source)==revision["sha256"]
    version=subprocess.run([str(binary),"--version"],capture_output=True,text=True,check=True)
    with socket.socket() as s:
        s.bind(("127.0.0.1",0))
        port=s.getsockname()[1]
    host=f"http://127.0.0.1:{port}"
    cmd=[str(binary),"-m",str(source),"--embeddings","--pooling","mean","--embd-normalize","2",
         "--host","127.0.0.1","--port",str(port),"-ngl","0","-np","4","-c","1024",
         "-b","512","-ub","512","-t","4","-tb","4"]
    with (ROOT/(STEM+"-server.log")).open("w") as log:
        job=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        try:
            for _ in range(75):
                if job.poll() is not None:
                    raise RuntimeError(f"llama-server exited {job.returncode}")
                try:
                    if request(host+"/health").get("status")=="ok":
                        break
                except URLError:
                    sleep(1)
            else:
                raise TimeoutError("Embedding server did not become healthy")
            yield host,{"source":revision,"runtime":(version.stdout+version.stderr).strip(),"settings":cmd[2:3]+cmd[4:]}
        finally:
            job.terminate()
            try:
                job.wait(timeout=8)
            except subprocess.TimeoutExpired:
                job.kill()
                job.wait()


def format_input(content,i,arm):
    def crop(text,n):
        words=text.split()
        if len(words)<=n:
            return " ".join(words)
        return " ".join(words[:n//2]+["…"]+words[-(n-n//2):])
    target=crop(content[i],42)
    if arm=="target":
        return f"Target: {target}"
    before=crop(content[i-1],16) if i>0 else ""
    after=crop(content[i+1],16) if i+1<len(content) else ""
    # Keep the target identical in both arms. Context comes only from this judgment.
    return f"Target: {target}\nPrevious: {before}\nNext: {after}"


def vector_bank(host,cases,rows,tag,checkpoint):
    directory=ROOT/"cache/gemma2-neighbor"
    directory.mkdir(exist_ok=True)
    features={kind:np.empty((len(rows),WIDTH),dtype=np.float32) for kind in ("target","neighbors")}
    pending=[]
    for kind in features:
        for pos,(j,i) in enumerate(rows):
            text=PROMPT+format_input(cases[j].texts,i,kind)
            pending.append((kind,pos,text))
    now=perf_counter()
    print("EmbeddingGemma 2",tag,len(pending),"inputs",flush=True)
    for start in range(0,len(pending),BUCKET):
        work=pending[start:start+BUCKET]
        inputs=[row[2] for row in work]
        key=sha256(json.dumps([FORMAT,checkpoint,inputs],ensure_ascii=False).encode()).hexdigest()
        path=directory/(key+".npy")
        if path.is_file():
            vectors=np.load(path,allow_pickle=False)
        else:
            payload=request(host+"/v1/embeddings",{"input":inputs})["data"]
            ids=[p["index"] for p in sorted(payload,key=lambda p:p["index"])]
            assert ids==list(range(len(inputs)))
            full=np.asarray([p["embedding"] for p in sorted(payload,key=lambda p:p["index"])],dtype=np.float32)
            assert full.shape==(len(inputs),768) and np.isfinite(full).all()
            assert np.allclose(np.linalg.norm(full,axis=1),1,atol=1e-5)
            vectors=full[:,:WIDTH]/np.maximum(np.linalg.norm(full[:,:WIDTH],axis=1,keepdims=True),1.e-8)
            part=path.with_suffix(".part")
            with part.open("wb") as f:
                np.save(f,vectors)
            part.replace(path)
        assert vectors.shape==(len(work),WIDTH) and np.isfinite(vectors).all()
        for v,(kind,i,_) in zip(vectors,work):
            features[kind][i]=v
        if start%128==0 or start+BUCKET>=len(pending):
            print(tag,min(start+BUCKET,len(pending)),len(pending),round(perf_counter()-now,1),flush=True)
    return features


def inputs(cases,rows,v,kind):
    base=np.stack([cases[j].structure[i] for j,i in rows])
    if kind=="hierarchy":
        return base
    return np.column_stack((base,v[kind]))


def kernel(a,b,order):
    k=a@b.T
    if order==1:
        return 1+k
    return 1+k+.5*np.square(k)


def write(x,y,prior,order,ridge):
    mean=x.mean(0)
    scale=1/np.maximum(x.std(0),.02)
    key=np.ascontiguousarray(((x-mean)*scale)/np.sqrt(x.shape[1]),dtype="f4")
    k=kernel(key,key,order).astype(np.float64)
    val=np.eye(7)[y]-prior
    coefficient=np.linalg.solve(k+np.eye(len(k))*ridge*k.trace()/len(k),val)
    return {"center":mean,"scale":scale,"keys":key,"values":coefficient.astype("f4")}


def correction(x,weights,order):
    q=((x-weights["center"])*weights["scale"])/np.sqrt(x.shape[1])
    return np.clip(kernel(q,weights["keys"],order)@weights["values"],-.7,.7)


def class_probs(prior,update,gain):
    z=np.maximum(prior+gain*update,0)
    return z/np.maximum(z.sum(1,keepdims=True),1.e-9)


def fit_routes(cases):
    pools=[[] for _ in range(7)]
    for j,c in enumerate(cases):
        for i in c.eligible[:max(8,len(c.eligible)//3)]:
            pools[c.gold[i]].append((j,i))
    rng=np.random.default_rng(916743)
    return [pool[i] for pool in pools for i in rng.choice(len(pool),min(44,len(pool)),replace=False)]


def eval_routes(cases):
    return [(j,i) for j,c in enumerate(cases) for i in c.eligible[:LIMIT]]


def picked(cases,rows):
    return (np.asarray([cases[j].gold[i] for j,i in rows]),
            np.asarray([cases[j].current[i] for j,i in rows]),
            np.stack([cases[j].prior[i] for j,i in rows]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-server",required=True,type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1)
    p=load_file(str(PARENT))
    vocab=json.loads((EDGE/"vocab.json").read_text())
    model=HierModel(DONOR)
    groups=sample_whole()
    fit_docs=edit_documents()

    def prepare(docs,group):
        states=embeddings(model,docs,"aligned-source-train") if group=="fit" else aligned_embeddings(model,docs,group)
        return [case_inputs(d,h,model,p,vocab) for d,h in zip(docs,states)]

    with backend(args.llama_server.resolve(strict=True)) as (host,engine):
        checkpoint=engine["source"]["sha256"]
        source=prepare(fit_docs,"fit")
        tr=fit_routes(source)
        train=vector_bank(host,source,tr,"fit",checkpoint)
        y,_,prior=picked(source,tr)
        written={}
        for kind,order,ridge in product(KINDS,(1,2),(.12,.55)):
            it=inputs(source,tr,train,kind)
            written[kind,order,ridge]=write(it,y,prior,order,ridge)
        del source,train
        dev=prepare(groups["select"],"select")
        routes=eval_routes(dev)
        emb=vector_bank(host,dev,routes,"select",checkpoint)
        gold,current,known=picked(dev,routes)
        best={}
        grid=[]
        for (kind,order,ridge),w in written.items():
            val=correction(inputs(dev,routes,emb,kind),w,order)
            for gain,gap in product((0.,.30,.60,.95),(.025,.075,.15)):
                q=predict(dev,routes,class_probs(known,val,gain),1.,gap) if gain else [c.current for c in dev]
                labels=np.asarray([q[j][i] for j,i in routes])
                m=measure(cm(gold,labels)).tolist()
                choice=(kind,order,ridge,gain,gap)
                grid.append([*choice,*m,int((labels!=current).sum())])
                total=(m[0],m[1],gain==0.,choice)
                if kind not in best or total>best[kind]:
                    best[kind]=total
        chosen={k:t[-1] for k,t in best.items()}
        written_files={}
        fixed={}
        for k,choice in chosen.items():
            candidate=OUT/f"{STEM}-{k}.safetensors"
            w={**written[choice[:3]],"settings":np.asarray(choice[1:])}
            save_file(w,str(candidate))
            fixed[k]=load_file(str(candidate))
            for name,value in w.items():
                np.testing.assert_array_equal(fixed[k][name],value)
            written_files[k]=file_hash(candidate)
        freeze=ROOT/"results"/(STEM+"-frozen.json")
        save_json(freeze,{"choices":chosen,"grid":grid,"engine":engine,"training_rows":len(tr),
            "training_labels":np.bincount(y,minlength=7).tolist(),"selection_cases":[c.key for c in dev],
            "parent_sha256":file_hash(PARENT),"weights_sha256":written_files,"per_case_route":LIMIT,"width":WIDTH,
            "train_cases":[q["id"] for q in fit_docs],"selection_parent":measure(cm(gold,current)).tolist()})
        print("Frozen",chosen,"selection",best,flush=True)
        disjoint=set(json.loads((EDGE/"overlap.json").read_text())["independent_exclusions_for_text_disjoint_check"])
        already={d["id"] for group in groups.values() for d in group}
        groups["independent"]=[d for d in independent_validation(already) if d["id"] not in disjoint]
        known_rows=json.loads((ROOT/"results/sentence-joint-edit.json").read_text())["splits"]
        result={"freeze_sha256":file_hash(freeze),"splits":{}}
        for group in ("select","confirm","independent"):
            cases,rr,vv=(dev,routes,emb) if group=="select" else (prepare(groups[group],group),None,None)
            if group!="select":
                rr=eval_routes(cases)
                vv=vector_bank(host,cases,rr,group,checkpoint)
            _,_,prior=picked(cases,rr)
            row={}
            for k,w in fixed.items():
                order,_,gain,gap=w["settings"]
                delta=correction(inputs(cases,rr,vv,k),w,int(order))
                values=class_probs(prior,delta,gain)
                yy=predict(cases,rr,values,1.,gap) if gain else [c.current for c in cases]
                row[k]=stats=summarize(cases,rr,yy)
                assert stats["full_parent"]==known_rows[group]["candidate"]
            result["splits"][group]=row
            save_json(ROOT/"results"/(STEM+".json"),result)
            print(group,[(k,q["routed_scores"]["macro_f1"],q["full"]["macro_f1"],q["changed"]) for k,q in row.items()],flush=True)


if __name__=="__main__":
    main()
