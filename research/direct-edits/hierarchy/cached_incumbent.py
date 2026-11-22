"""Replay the already-frozen 20% reviewer cache without GPU access."""
import argparse
import sys
import types
import numpy as np
from embedding_write import ROOT,HERE,metric,confusion
import sentence_wide_bert_multiclass_write as system

class Serial:
    def __init__(self,*a,**kw):pass
    def __enter__(self):return self
    def __exit__(self,*a):pass
    def map(self,fn,args):return map(fn,args)

def make(split):
    system.wide.ProcessPoolExecutor=Serial
    worker=types.ModuleType("openvino_worker")
    def offline_only(*args,**kwargs): raise RuntimeError("Missing a required cached incumbent feature; no device inference authorized in this lane")
    worker.embed_ids=worker.get_engine=offline_only
    sys.modules["openvino_worker"]=worker
    obj=system.inputs(split,system.gain.plans(),system.read_wide())
    k,offset,gold=obj.bank[:3]
    tmp=HERE/("strong-"+split+".part")
    with tmp.open("wb") as fp:
        np.savez(fp,keys=k,ends=offset,gold=gold,pred=obj.incumbent)
    tmp.replace(HERE/("strong-"+split+".npz"))
    print(split,metric(confusion(gold,obj.incumbent)),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('split',choices=('select','confirm'));make(a.parse_args().split)
