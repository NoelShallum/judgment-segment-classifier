"""Safely precompute still-distant fit shards in reverse without changing the main summation order."""
import json
import os
from pathlib import Path

from scale_program import FOLDER,SHARD,SIZES,Corpus,empty,update,file_hash,save_np,dump

def main():
    path=FOLDER/"plan.json";plan=json.loads(path.read_text());proof=file_hash(path);ids=plan["fit"]
    corpus=Corpus(plan);folder=FOLDER/"shards";folder.mkdir(exist_ok=True)
    edges=sorted(set([0,*range(SHARD,len(ids),SHARD),*SIZES]))
    seen=done=0
    pieces=list(zip(edges,edges[1:]))
    # Finish the 2,400-case checkpoint early, then work inward from the full-coverage edge.
    for wave in ([ab for ab in pieces if ab[1]<=2400],[ab for ab in pieces if ab[1]>2400]):
        for a,b in reversed(wave):
            prior=FOLDER/"progress.json"
            forward=json.loads(prior.read_text()).get("fit_judgments",0) if prior.exists() else 0
            if a<forward+3*SHARD:break
            target=folder/f"{a:04d}-{b:04d}.npz"
            if target.exists():continue
            # Coordinate with a possible restarted secondary worker. The normal forward worker remains at least three shards behind.
            lock=target.with_suffix(".lock")
            try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
            except FileExistsError:continue
            try:
                os.write(fd,str(os.getpid()).encode());os.close(fd);block=empty()
                for k in ids[a:b]:
                    gt,views=corpus.extract(k);update(block,views,gt,corpus.prior)
                if not target.exists():save_np(target,source_plan=proof,keys=ids[a:b],**block)
                seen+=b-a;done+=1
                dump(FOLDER/"reverse-progress.json",dict(shards=done,training_judgments=seen,lowest_completed_index=a,cache_usage=corpus.stats,source_plan=proof))
                print("reverse completed",a,b,"total this run",seen,flush=True)
            finally:
                if lock.exists():lock.unlink()
if __name__=="__main__":main()
