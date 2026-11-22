"""Test a validation-selected dense write that permits revisions to all seven roles."""
import argparse

import numpy as np
from safetensors.numpy import save_file

import sentence_dense_readout_write as dense
from sentence_document_graph_write import OUT, file_hash, paired, row_summary, save_json

TAG = "sentence-dense-unrestricted"


def main(splits):
    result_dir = dense.FOLDER
    parent_freeze = result_dir/(dense.TAG+"-frozen.json")
    parent = dense.read(dense.TAG+"-frozen")
    floor = parent["validation_parent"]["accuracy"]-parent["accuracy_floor_points"]/100
    trials = [(m,a,arm,setting) for arm,option in parent["arms"].items()
              for setting,m,a,_ in option["search"] if not setting[-1] and a>=floor]
    macro,accuracy,arm,setting = max(trials)
    rarity,focus,damping,*action = setting
    assert not action[-1]
    tr = dense.prepare("fit")
    labels = tr[2]
    bias = (1.-tr[4][np.arange(len(labels)),labels])**2
    freq = np.maximum(np.bincount(labels,minlength=7)/len(labels),.013)
    basis = dense.learn_basis(tr[-1],arm)
    x = dense.infer_basis(tr[-1],arm,basis)
    center = x.mean(0,dtype=np.float64).astype(np.float32)
    scale = (1./np.maximum(x.std(0,dtype=np.float64),.025)).astype(np.float32)
    x -= center
    x *= scale
    z = np.column_stack((x,np.ones(len(x),np.float32)))
    w = (freq[labels]**-rarity*(1+focus*bias)).astype(np.float32)
    gram,links = dense.fit_moments(z,labels,tr[4],w)
    reg = np.r_[np.ones(len(gram)-1),.02]*gram.trace()/len(gram)
    beta = np.linalg.solve(gram+np.diag(damping*reg),links).astype(np.float32)
    weights = {**basis,"center":center,"scale":scale,"weight":beta[:-1].copy(),"bias":beta[-1].copy()}
    target = OUT/(TAG+".safetensors")
    save_file(weights,str(target))
    del tr,x,z
    receipt = result_dir/(TAG+"-frozen.json")
    save_json(receipt,{"dense_freeze_sha256":file_hash(parent_freeze),"weights_sha256":file_hash(target),
                       "arm":arm,"setting":setting,"select_macro_f1":macro,"select_accuracy":accuracy})
    result = {"freeze_sha256":file_hash(receipt),"splits":{}}
    for split in splits:
        v = dense.prepare(split)
        estimate = dense.output(dense.infer_basis(v[-1],arm,weights),weights)
        pred = dense.replace(v[3],v[4],estimate,*action)
        matrix = dense.paired_confusion(v,pred)
        scores = row_summary(matrix.sum(0))
        if split=="select":
            assert scores["macro_f1"]==macro and scores["accuracy"]==accuracy
        result["splits"][split] = {
            "cases":len(v[0]),"scores":scores,"baseline":dense.baseline(v,split),
            "paired_parent":paired(matrix,dense.paired_confusion(v,v[3])),
            "changed":int(np.count_nonzero(v[3]!=pred)),
            "reasoning_predictions_revised":int(np.count_nonzero((v[3]==5)&(pred!=5)))}
        print(split,arm,scores["macro_f1"],scores["accuracy"],result["splits"][split]["reasoning_predictions_revised"],flush=True)
    save_json(result_dir/(TAG+".json"),result)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("splits",nargs="+")
    main(parser.parse_args().splits)
