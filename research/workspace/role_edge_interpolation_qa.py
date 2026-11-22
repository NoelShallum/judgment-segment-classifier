"""Replay the physically written intermediate profile and verify offset output."""
import json

import numpy as np

from role_edge_interpolation import (EDGE, CRF, DONOR, ROOT, HierModel, bootstrap, decode,
    document, file_hash, full_score, gold_cases, load_file, prepare, save_json, torch)
from predict_role_edge import infer
from hier_full import CACHE
from hier_model import LABELS
from span_model import merge
from discourse_span_edit import text_features
from role_edge_runtime import transition_scores


def main():
    torch.set_num_threads(2)
    path=ROOT/"checkpoints/hierarchical/whole-role-edge-interpolated-exact.safetensors"
    freeze_file=ROOT/"results/role-edge-interpolation-frozen.json"
    freeze=json.loads(freeze_file.read_text())
    sha=file_hash(path)
    assert sha==freeze["weight_hashes"]["exact"]
    weights=load_file(str(path))
    vocab=json.loads((EDGE/"vocab.json").read_text())
    model=HierModel(DONOR)
    source=gold_cases()
    records=[document(k,r) for k,r in source.items()]
    window=prepare(records,"test",model,{"interpolated":weights},vocab,weights)
    gain=float(weights["edge.gain"])
    scored=np.stack([item.score(decode(item.logits["interpolated"],weights[CRF],x,gain)) for item,x in window])
    aggregate=full_score(scored)
    study=json.loads((ROOT/"results/role-edge-interpolation.json").read_text())["splits"]["test"]["exact"]["candidate"]
    assert aggregate==study
    archived=np.load(EDGE/"per-case-published-comparison.npz")
    np.testing.assert_array_equal(archived["case_ids"],list(source))
    qa_docs=json.loads((EDGE/"qa.json").read_text())["cases"]
    for report in qa_docs:
        d=document(report["case"],source[report["case"]])
        with np.load(CACHE/(str(d["id"])+".npz")) as f:
            raw=f["raw"]
        cues,pos=text_features(d)
        z=raw+cues@weights["cue_head"]+pos@weights["position_head"]
        edges=transition_scores(d["text"],d["units"],vocab,weights)
        expected=merge(d["units"],decode(z,weights[CRF],edges,gain),LABELS)
        actual=infer(d["text"],model,weights,vocab,weights,"joint",gain)["spans"]
        assert actual==expected
    report={"sha256":sha,"cases_replayed":len(scored),"verified_live_case_ids":[d["case"] for d in qa_docs],
            "archive_gap_ci95":bootstrap(scored,archived["archive"]),"scores":aggregate}
    save_json(ROOT/"results/role-edge-interpolation-qa.json",report)
    print("Verified deployed weights on",len(scored),"official cases",flush=True)


if __name__=="__main__":
    main()
