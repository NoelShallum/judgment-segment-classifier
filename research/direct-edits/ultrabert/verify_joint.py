"""Replay a serialized edited backbone in PyTorch and verify its row-aligned classifier outputs."""
import argparse
import json
import os
import numpy as np
import torch
from safetensors.numpy import load_file
from extract import Encoder, groups
from joint import position
from study import SOURCE, data, h, load_source
from prepare import ROOT, save

def main():
    p=argparse.ArgumentParser();p.add_argument("--stage",default="quick");p.add_argument("--cap",default=384,type=int)
    p.add_argument("--mode",default="context");p.add_argument("--method",default="joint");f=p.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","1")));torch.set_num_interop_threads(1)
    folder=ROOT/f"{f.method}-{f.stage}-{f.mode}-{f.cap}"
    report=json.loads((folder/"report.json").read_text())
    item=groups(f.stage)["fit"][0];ds=data([item],f.cap,f.mode)
    source=load_source();enc=Encoder()
    mat=enc.model.layers[23].mlp.down_proj.weight
    assert h(mat.detach().cpu().numpy())==h(source)
    for name,rec in report["experiments"].items():
        tensors=load_file(str(folder/f"{name}.safetensors"))
        with torch.no_grad():mat.copy_(torch.from_numpy(tensors[SOURCE]))
        observed=enc.run(item,f.cap,f.mode)
        delta=tensors[SOURCE]-source
        expected=(ds["e"]+ds["k"]@delta.astype(float).T)
        wh=tensors["classifier.weight"].T
        bias=position(ds)@tensors["position.weight"].T+tensors["classifier.bias"]
        pred=expected@wh+bias;actual=observed["e"]@wh+bias
        rec["fresh_forward"]={"case":item["id"],"csv_rows":ds['row'].tolist(),"installed_tensor_sha256":h(mat.detach().cpu().numpy()),
                             "max_abs_logit_difference":float(np.max(np.abs(pred-actual))),
                             "relative_hidden_l2_error":float(np.linalg.norm(expected-observed["e"])/np.linalg.norm(observed["e"])),
                             "same_argmax":bool(np.array_equal(pred.argmax(1),actual.argmax(1)))}
        print(name,rec["fresh_forward"],flush=True)
        save(folder/"report.json",report)
if __name__=="__main__":main()
