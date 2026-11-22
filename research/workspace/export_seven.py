"""Assemble the user-facing seven-role model, sequence decoder and auditable portable release."""
import hashlib,json,os,shutil,zipfile
from pathlib import Path
import numpy as np
from safetensors.numpy import save_file
from seven_paths import ROOT,LABELS,folder_for
from weight_edit import file_hash,tensor
from span_model import authored_transitions
from package_candidate import install,write
OUT=ROOT.parents[1]/"outputs/judgment-role";KIT=ROOT/"portable-seven-role"
REV="44d7925e5ab422ecf04acd5dcbb0b62a6f8d8f53"
PUBLIC={"a":("InLegalBERT(i)model.safetensors","bbe8af20d1524c7535b1cbe68c6d4c3c844c6cc9fccef46c03c9ff56e7c18cea"),
        "b":("InLegalBERT(i-2%2C%20i-1%2C%20i)model.safetensors","a81f509d250f98b44872e89e10186d5dcc17d37905b69aa1d9a077fb841a62ac")}
def jfile(p,v):write(p,json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def build_profile(name,parent,model,seq,extra=()):
    d=KIT/"profiles"/name;d.mkdir(parents=True,exist_ok=True);src=model/"model.safetensors"
    head=d/"head.npz"
    keys=("classifier.weight","classifier.bias",*extra)
    with head.open("wb") as f:np.savez_compressed(f,**{k:tensor(src,k) for k in keys})
    source,h=PUBLIC[parent];newhash=file_hash(src)
    record={"profile":name,"source_url":f"https://huggingface.co/L-NLProc/LegalSeg_InLegalBERT/resolve/{REV}/InLegalBERT/{source}",
            "source_sha256":h,"target_sha256":newhash,"head_sha256":file_hash(head),"parameters":109487623,"fp32_checkpoint_bytes":src.stat().st_size,
            "head_scalars":5383,"replacement_keys":list(keys),"labels":LABELS,"sequence":bool(seq)}
    if seq:
        tr=authored_transitions(seq["sticky"],seq["order"]);n=seq["neighbor"];ker=np.array([n,1-2*n,n],dtype=np.float32)
        save_file({"transition":tr,"neighbor_kernel":ker},str(d/"sequence.safetensors"))
        q={**seq,"source_head_sha256":newhash,"labels":LABELS,"sha256":file_hash(d/"sequence.safetensors")}
        jfile(d/"sequence.json",q)
    jfile(d/"edit.json",{**record,"source_edit":json.loads((model/"edit.json").read_text())})
    return record
def main():
    KIT.mkdir(parents=True,exist_ok=True);OUT.mkdir(exist_ok=True,parents=True)
    study=json.loads((ROOT/"results/document-segmentation.json").read_text());prog=study["selection"]
    arows=[r for r in study["selection_candidates"] if r[0]=="A isolated"];best=max(arows,key=lambda r:(r[4],r[5]))
    compact={"method":best[0],"context_left":0,"parent":"a","sticky":best[1],"order":best[2],"neighbor":best[3]}
    profiles={};paths={}
    for key,base,dir,seq in (("compact","a","associative-direct",compact),("document",prog["parent"],"associative-direct" if prog["parent"]=="a" else "associative-context",prog),
                              ("centroid","a","selected-direct",compact)):
        p=ROOT/"checkpoints"/dir;profiles[key]=build_profile(key,base,p,seq);paths[key]=p
    graft=ROOT/"results/graft-validation.json"
    if graft.is_file():
        g=json.loads(graft.read_text());p=ROOT/"checkpoints/graft-direct"
        profiles["graft"]=build_profile("graft",g["source_encoder"],p,None)
    mlp=ROOT/"checkpoints/associative-mlp"
    if (mlp/"model.safetensors").is_file():
        profiles["internal"]=build_profile("internal","a",mlp,None,("bert.encoder.layer.11.output.dense.weight","bert.encoder.layer.11.output.dense.bias"))
    jfile(KIT/"profiles.json",{"default":"compact","official_revision":REV,"profiles":profiles})
    for n in ("config.json","tokenizer.json","tokenizer_config.json","special_tokens_map.json","vocab.txt"):install(paths["compact"]/n,KIT/n)
    for n in ("predict.py","span_model.py","windowing.py","weight_edit.py"):install(ROOT/n,KIT/n)
    install(ROOT/"rebuild_profiles.py",KIT/"rebuild.py")
    install(Path("/usr/share/common-licenses/Apache-2.0"),KIT/"LICENSE");install(ROOT/"references/tokenizer/LICENSE-MIT.txt",KIT/"TOKENIZER-LICENSE.txt")
    write(KIT/"NOTICE","AttoFlow modified the classification heads of L-NLProc/LegalSeg_InLegalBERT by Nigam and collaborators. The span decoder adds authored transition and neighbor tensors. Tokenizer from law-ai/InLegalBERT. See LICENSE, TOKENIZER-LICENSE.txt and the source links in README.md.\n")
    write(KIT/"requirements.txt","numpy>=1.26,<3\ntorch>=2.5,<3\ntransformers>=4.48,<5\nsafetensors>=0.5,<1\n")
    readme="""# AttoFlow seven-role judgment classifier

A local research prototype for English Indian judgments. Give `predict` one sentence or paragraph to get one dominant role. Give `segment` a longer text to get contiguous labeled spans. All seven roles are retained: `none`, `facts`, `issue`, `petitioner_argument`, `respondent_argument`, `reasoning`, and `decision`.

## Run on a CPU

With Python 3.11, run the following inside the extracted archive. Reconstruction downloads the public 438 MB donor and verifies SHA-256. An existing donor can be supplied with `--source /path/to/donor.safetensors`.

```sh
python -m pip install -r requirements.txt
python rebuild.py --profile compact --output-dir model
python model/predict.py "Notice was served on 3 May. We therefore allow the appeal."
python rebuild.py --profile document --output-dir document-model
python document-model/predict.py --spans < judgment.txt
```

In a ready-made model directory the predictor can be imported directly.

```python
from predict import JudgmentRoleClassifier

reader = JudgmentRoleClassifier()
one = reader.predict("We therefore dismiss the appeal.")
ranges = reader.segment("The notice was served. Counsel for the appellant challenged it. We dismiss the appeal.")
```

Offsets are Python Unicode character positions, with an inclusive start and an exclusive end. Use `text[start:end]` to recover the exact original characters. Spans cover the original string without gaps. Whitespace-only input raises an error. Raw excerpts that exceed one BERT window are fully covered with overlapping windows. The sequence decoder proposes candidate boundaries from punctuation and paragraph breaks, scores the resulting units, and merges adjacent equal roles. The document profile uses the strongest sequence program selected on local validation. The compact profile also includes its independently selected sequence weights.

## Weights and evaluation

Every BERT profile has 109,487,623 parameters and a 437,974,028-byte FP32 checkpoint. Each BERT classifier head has 5,376 output weights and seven biases. The internal profile also installs a replacement for the final feed-forward projection and its bias. Rebuild it with the internal profile and a separate output directory. The sequence layer adds 49 transition values and three neighbor coefficients. The `centroid` profile reconstructs the original edit, and `graft` reconstructs the exploratory sparse donor-row experiment for single-excerpt classification. When the letter-based instruction readout is included, its folder documents the larger local decoder separately.

See `evidence/AttoFlow-judgment-role-results.pdf` for measured performance, sample construction, uncertainty and limitations. The active legal results use seven-role LegalSeg and separate authored diagnostics. The report also describes historical controls and published references. Some inputs are intact annotations, some are adjacent rows joined under an explicit rule, and fictional examples have authored intended labels. Gemini Decision 1 receives a typed choice question. None of the active weights use provider predictions or gradient updates. The donor checkpoints were previously trained by their publishers.

The public checkpoint named for the standalone model is byte-identical to a file named for the previous sentence and previous gold label. The released prediction CSV and the loaded checkpoint produce different results. The report keeps those systems distinct.

The ready-made classifiers are research candidates. Their accuracy is an empirical result in the report. The JSON response is a role prediction, with uncertainty particularly high when speaker attribution or procedural context is absent.

[LegalSeg authors’ paper](https://aclanthology.org/2025.findings-naacl.63/) · [Pinned donor weights](https://huggingface.co/L-NLProc/LegalSeg_InLegalBERT/tree/44d7925e5ab422ecf04acd5dcbb0b62a6f8d8f53/InLegalBERT) · [Official dataset](https://huggingface.co/datasets/L-NLProc/LegalSeg_CSV) · [Gemini Decisions](https://aistudio.google.com/docs/decisions-api)
"""
    write(KIT/"README.md",readme)
    for key,dirname in (("compact","model"),("document","document-model")):
        dst=OUT/dirname;dst.mkdir(exist_ok=True)
        for n in ("config.json","tokenizer.json","tokenizer_config.json","special_tokens_map.json","vocab.txt","requirements.txt","predict.py","span_model.py","windowing.py","LICENSE","NOTICE","README.md","TOKENIZER-LICENSE.txt"):install(KIT/n,dst/n)
        for n in ("edit.json","sequence.json","sequence.safetensors"):install(KIT/"profiles"/key/n,dst/n)
        install(paths[key]/"model.safetensors",dst/"model.safetensors")
    evidence=KIT/"evidence";evidence.mkdir(exist_ok=True)
    prefixes=("gemini-seven-","document-segmentation","identical-inputs-ceiling","graft-validation","associative-","mixed-encoder-validation","program-validation","direct-official-test",
             "direct-interim-test","all-published-predictions","published-baselines","author-checkpoint-inventory","split-inventory","window-coverage-test","openvino-",
             "instruction-projection-validation","instruction-semantic-validation","gemini-paired-synthetic","seven-profile-qa","seven-instruction-qa","distribution-check","hierarchical-","fragment-validation","fusion-validation","internal-mlp-validation","cue-program-validation","public-document-donor-audit","stocktake-standalone-alignment","pause-snapshot")
    for f in (ROOT/"results").glob("*.json"):
        if f.name.startswith(prefixes):install(f,evidence/f.name)
    install(ROOT/"references/official/verified-local.json",evidence/"official-csv-checksums.json")
    install(OUT/"AttoFlow-judgment-role-results.pdf",evidence/"AttoFlow-judgment-role-results.pdf")
    install(OUT/"experiment-results.html",evidence/"experiment-results.html")
    for group in ("pilot","confirmation","paragraphs","homogeneous"):
        folder=folder_for(group);items=json.loads((folder/"approved.json").read_text());sample=[]
        for row in items:sample.append({k:v for k,v in row.items() if k!="text"})
        jfile(evidence/(group+"-input-manifest.json"),sample)
    rec=KIT/"reproduce"
    files=("encoder.py","role_fixtures.py","program_trial.py","weight_edit.py","mixed_encoder_trial.py","associative_edit.py","graft_rows.py","span_eval.py","span_model.py","windowing.py","seven_paths.py",
           "matched_local.py","matched_metrics.py","span_cloud_metrics.py","cache_full.py","score_official_test.py","import_official.py","replay_seven.py","seven_paragraphs.py","gguf_choice_rows.py",
           "synthetic_review.py","gemini_matched_live.py","gemini_legal_modes.py","program_trial.py","project_hidden.cpp","openvino_worker.py","identical_inputs.py")
    for n in files:
        if (ROOT/n).is_file():install(ROOT/n,rec/n)
    cloud=evidence/"gemini-seven-spans.json"
    if cloud.exists():
        jfile(evidence/"span-input-manifest.json",[{"doc":r["case"],"row_ids":r["source_rows"],"sha256":r["raw_sha256"]} for r in json.loads(cloud.read_text())["records"]])
    write(rec/"README.md","Restore the checksummed source files using python import_official.py --downloads /your/csv/directory. Run python replay_seven.py to reconstruct the exact cloud-benchmark inputs from the packaged manifests, verify the source hashes, and recompute paired metrics offline from recorded choices. The other scripts document the gradient-free candidate programs. All module file paths are relative to this directory. To repeat local forward inference, place the pinned donors and tokenizer under references as specified by encoder.py and weight_edit.py. The source manifests include no judgment text.\n")
    for semantic in (False,True):
        batch="instruction-semantic" if semantic else "instruction-projection";src=ROOT/"checkpoints"/batch;folder=ROOT/("gemma-semantic" if semantic else "gemma-projection")
        if not (src/"edited.safetensors").is_file():continue
        target=KIT/batch;target.mkdir(exist_ok=True)
        for n in ("unchanged.safetensors","edited.safetensors"):install(src/n,target/n)
        for n in ("prefix.txt","suffix.txt"):install(folder/n,target/n)
        install(ROOT/"predict_instruction.py",target/"predict.py");install(ROOT/"windowing.py",target/"windowing.py")
        q=json.loads((folder/"frozen.json").read_text())
        jfile(target/"profile.json",{"labels":LABELS,"head_sha256":file_hash(target/"edited.safetensors"),"source":q["spec"]["model"],"revision":q["spec"]["revision"],"filename":q["spec"]["filename"],"source_sha256":q["spec"]["sha256"],"context_length":2048,"method":q["selected_actual_edit"]})
        write(target/"README.md","This optional float32 seven-row head operates on the final hidden state of the pinned Gemma 4 E2B Q8_0 decoder. The separate decoder file is 4,967,491,520 bytes. Download the pinned model from https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF/tree/2a8538c2e2d4687953a86c422c7c623cfd022880. With llama.cpp b11960 or a compatible version, start llama-server -m gemma-4-E2B-it-Q8_0.gguf -ngl 0 -c 2048 -np 1 --embeddings --pooling last --host 127.0.0.1 --port 8080. Install requests, numpy and safetensors, then run python predict.py \"We therefore dismiss the appeal.\". See the report for comparison with native token logits and the unchanged copied readout.\n")
    for obsolete in ("prepare_matched.py","prepare_confirmation.py"):
        (KIT/"reproduce"/obsolete).unlink(missing_ok=True)
    from export_extra import build
    build(KIT,OUT)
    archive=OUT/"AttoFlow-judgment-role-edit.zip";stage=archive.with_suffix(".part")
    with zipfile.ZipFile(stage,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for p in sorted(KIT.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:z.write(p,p.relative_to(KIT))
    with zipfile.ZipFile(stage) as z:
        assert z.testzip() is None
        assert not any(n.endswith(".csv") or "prr-" in n or "marro" in n for n in z.namelist())
    stage.replace(archive)
    print("Built",len(profiles),"BERT profiles.",archive.stat().st_size,"portable bytes.",flush=True)
if __name__=="__main__":main()
