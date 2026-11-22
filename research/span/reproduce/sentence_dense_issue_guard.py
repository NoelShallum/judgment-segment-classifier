"""A prespecified issue-preserving ablation on a new validation cohort."""
import numpy as np

import sentence_dense_readout_write as dense
import sentence_dense_referral_combine as fusion
from sentence_document_graph_write import file_hash, paired, row_summary, save_json

TAG = "sentence-dense-issue-guard"
ARM = "nonlinear"


def scores(bank, edits, ref):
    m = dense.paired_confusion(bank,edits)
    b = dense.paired_confusion(bank,ref)
    return {"scores":row_summary(m.sum(0)),"paired":paired(m,b),
            "changed":int(np.count_nonzero(edits!=ref))}


def main():
    folder = dense.FOLDER
    dense_plan = dense.read(dense.TAG+"-frozen")
    plan = {"dense_weights_sha256":dense_plan["arms"][ARM]["sha256"],
            "arm":ARM,"heldout_sha256":file_hash(folder/(dense.TAG+"-role-guard.json")),
            "hypothesis":"The official-test and first prospective analysis of the nonlinear readout showed that issue corrections contributed little and could hurt issue F1.",
            "rule":"Keep all predictions to or from issue unchanged by the dense edit. For the combined candidate keep the BERT output on referred rows.",
            "frozen_dense_sha256":file_hash(folder/(dense.TAG+"-frozen.json")),
            "frozen_bert_sha256":file_hash(folder/(fusion.bert.TAG+"-frozen.json"))}
    freeze = folder/(TAG+"-frozen.json")
    if freeze.exists():
        assert dense.read(TAG+"-frozen")==plan
    else:
        save_json(freeze,plan)
    result = {"freeze_sha256":file_hash(freeze),"splits":{}}
    for split in ("role-guard","holdout","test"):
        if split=="role-guard":
            bank=dense.prepare(split)
            original=dense.baseline(bank,split)
        else:
            bank,original,expert,routed=fusion.views(split)
        raw=fusion.readout(bank,ARM,dense_plan["arms"][ARM])
        y=dense.preserve_issues(bank[3],raw)
        s={"cases":len(bank[0]),"baseline":original,"unguarded_dense":scores(bank,raw,bank[3]),
           "guarded_dense":scores(bank,y,bank[3]),"guarded_vs_dense":scores(bank,y,raw)}
        assert s["guarded_dense"]["scores"]["per_role_f1"]["issue"]==original["per_role_f1"]["issue"]
        if split!="role-guard":
            hybrid=np.where(routed,expert,y)
            s["bert"]=scores(bank,expert,bank[3])["scores"]
            s["combined"]=scores(bank,hybrid,expert)
            assert s["combined"]["scores"]["per_role_f1"]["issue"]==s["bert"]["per_role_f1"]["issue"]
            if split=="test":
                record = dense.read("known-sentence-numerical-archives")
                archive = folder/"known-sentence-numerical-archives.npz"
                assert file_hash(archive)==record["score_file_sha256"]
                name = record["best_by_macro_f1"]
                with np.load(archive,allow_pickle=False) as f:
                    np.testing.assert_array_equal(f["cases"],bank[0])
                    author = f[name]
                s["aligned_author"]={"name":name,"scores":row_summary(author.sum(0))}
                s["combined_vs_aligned_author"] = paired(dense.paired_confusion(bank,hybrid),author)
        result["splits"][split]=s
        print(split,{k:(round(v["scores"]["macro_f1"],7),round(v["scores"]["accuracy"],7)) for k,v in s.items() if isinstance(v,dict) and "scores" in v},flush=True)
        save_json(folder/(TAG+".json"),result)


if __name__=="__main__":
    main()
