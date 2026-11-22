"""Audit contiguous lexical saves and verify saved weights against the fit equations."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from safetensors.numpy import load_file,save_file

import sentence_phrase_readout as phrase
from sentence_document_graph_write import OUT,file_hash,save_json


def recover():
    q=phrase.ROOT/"qa"/"phrase-layout-invalid"
    q.mkdir()
    model=OUT/(phrase.TAG+"-lexical.safetensors")
    receipt=phrase.RESULT/(phrase.TAG+"-lexical-frozen.json")
    source=json.loads(receipt.read_text())
    old_hash=file_hash(model)
    assert source["weights_sha256"]==old_hash and "layout_recovery" not in source
    x=np.arange(60,dtype=np.float32).reshape(10,6)
    raw=np.asfortranarray(x)
    with TemporaryDirectory(dir=q) as tmp:
        check=Path(tmp)/"matrix.safetensors"
        save_file({"weight":raw},str(check))
        roundtrip=load_file(str(check))["weight"]
        assert not np.array_equal(roundtrip,x)
        assert np.array_equal(roundtrip.ravel().reshape(x.shape,order="F"),x)
    bad=load_file(str(model))["weight"]
    correct=np.ascontiguousarray(bad.ravel().reshape(bad.shape,order="F"))
    assert not np.array_equal(correct,bad)
    model.replace(q/model.name)
    receipt.replace(q/receipt.name)
    save_file({"weight":correct},str(model))
    np.testing.assert_array_equal(load_file(str(model))["weight"],correct)
    source["weights_sha256"]=file_hash(model)
    source["layout_recovery"]={"invalid_weights_sha256":old_hash,"method":"Decode stored column-major order and resave C contiguous.","test_fixture_shape":list(x.shape)}
    save_json(receipt,source)
    for path in phrase.CACHE.glob("*.npy"):
        if path.stem!="fit":
            path.replace(q/path.name)
    for path in phrase.RESULT.glob("sentence-phrase-*.json"):
        if path!=receipt:
            path.replace(q/path.name)
    for i in range(len(phrase.ARMS)):
        path=OUT/(phrase.TAG+f"-{i}.safetensors")
        if path.is_file():
            path.replace(q/path.name)
    print("Recovered canonical layout",correct.shape,old_hash,file_hash(model),flush=True)


def verify():
    _,_,y,anchor,text=phrase.corpus("fit")
    vocab_path=OUT/(phrase.TAG+"-vocabulary.json")
    path=OUT/(phrase.TAG+"-lexical.safetensors")
    receipt=phrase.RESULT/(phrase.TAG+"-lexical-frozen.json")
    proof=json.loads(receipt.read_text())
    assert proof["weights_sha256"]==file_hash(path)
    assert proof["vocabulary_sha256"]==file_hash(vocab_path)
    v=phrase.vectorizers(json.loads(vocab_path.read_text()))
    w=load_file(str(path))["weight"]
    norm=(1+1.5*(1-anchor[np.arange(len(y)),y])**2).astype(np.float32)
    hot=np.eye(7,dtype=np.float32)[y]
    target=np.column_stack((hot,hot-anchor))
    rhs=np.zeros_like(w[:,:14],dtype=np.float64)
    discrepancy=np.zeros_like(w,dtype=np.float64)
    for a in range(0,len(y),8192):
        b=min(a+8192,len(y))
        x=phrase.sparse(v,text[a:b])
        t=target[a:b]
        sample_weight=norm[a:b,None]
        rhs+=x.T@(sample_weight*t)
        for j in range(len(proof["ridge"])):
            sl=slice(j*14,(j+1)*14)
            discrepancy[:,sl]+=x.T@(sample_weight*(x@w[:,sl]-t))
        print("Audited",b,"/",len(y),flush=True)
    for j,d in enumerate(proof["ridge"]):
        sl=slice(j*14,(j+1)*14)
        discrepancy[:,sl]+=d*w[:,sl]
    residual=np.max(np.abs(discrepancy))/np.max(np.abs(rhs))
    assert residual<2e-5,residual
    save_json(phrase.RESULT/(phrase.TAG+"-lexical-qa.json"),{"file_sha256":file_hash(path),
        "lexical_freeze_sha256":file_hash(receipt),"fit_rows":len(y),
        "normalized_max_normal_equation_residual":float(residual)})
    print("Normal equations verified",residual,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("recover","verify"))
    if parser.parse_args().action=="recover":
        recover()
    else:
        verify()
