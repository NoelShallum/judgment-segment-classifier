"""Audit that candidate edits reach only the intended native LSTM tensor slices."""
import numpy as np

import sentence_recurrent_inverse as study
from sentence_document_graph_write import DONOR,OUT,HierModel,file_hash,load_file,save_json,torch


def main():
    torch.set_num_threads(1)
    base=HierModel(DONOR)
    upstream=study.raw_parameters(base)
    receipt=study.RESULT/(study.TAG+"-frozen.json")
    plan=study.read(study.TAG+"-frozen")
    docs,words=study.prepare("confirm")
    order=sorted(range(len(docs)),key=lambda i:len(words[i]))
    subset=[order[0],order[len(docs)//2],order[-1]]
    natural=[base.contextualize(torch.from_numpy(words[i])) for i in subset]
    suits={}
    for arm in study.ARMS:
        path=OUT/(study.TAG+"-"+str(study.ARMS.index(arm))+".safetensors")
        assert file_hash(path)==plan["arms"][arm]["sha256"]
        patch=load_file(str(path))
        expected=set(plan["arms"][arm]["tensors"])
        assert expected==set(patch)
        changed=0
        for key,val in patch.items():
            delta=val!=upstream[key]
            changed+=np.count_nonzero(delta)
            if arm=="readout-only":
                assert key.startswith("emitter.hidden2tag.") and not delta[:3].any()
            else:
                assert key.startswith("emitter.lstm.")
                n=base.doc.hidden_size
                assert not delta[:2*n].any() and not delta[3*n:].any()
        assert changed==plan["arms"][arm]["values_replaced"] and changed>0
        study.install(base,upstream)
        study.install(base,patch)
        installed=study.raw_parameters(base)
        assert all(np.array_equal(val,installed[key]) for key,val in patch.items())
        drive=[]
        for j,i in enumerate(subset):
            words_i=words[i]
            modified=base.contextualize(torch.from_numpy(words_i))
            delta=float(np.max(np.abs(modified-natural[j])))
            record={"case":docs[i]["id"],"sentences":len(words_i),"max_hidden_drift":delta}
            if arm=="readout-only":
                assert delta==0.
            else:
                assert delta>0.
                error=max(study.inverse_targets(words_i,modified,np.zeros_like(modified),base,k)[2] for k in range(2))
                record["max_gate_reconstruction_error"]=error
            drive.append(record)
        suits[arm]={"verified_replaced":int(changed),"live_cases":drive}
    assert file_hash(study.BASE)==plan["source_sha256"]["backbone"]
    save_json(study.RESULT/(study.TAG+"-weight-qa.json"),{"freeze_sha256":file_hash(receipt),"arms":suits})
    print("Verified native tensor scopes and three live forward passes per arm",flush=True)


if __name__=="__main__":
    main()
