"""Rebuild a checksum-pinned local seven-role profile from its public donor and bundled head."""
import argparse,json,os,shutil,tempfile,urllib.request
from pathlib import Path
import numpy as np
from weight_edit import file_hash,patched_checkpoint
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile",default="compact")
    parser.add_argument("--source",type=Path,help="Existing public donor file, or omit to download it.")
    parser.add_argument("--output-dir",type=Path,default=Path("model"))
    args=parser.parse_args()
    kit=Path(__file__).resolve().parent
    manifest=json.loads((kit/"profiles.json").read_text());profiles=manifest["profiles"]
    if args.profile not in profiles:parser.error("Choose one of "+", ".join(profiles))
    info=profiles[args.profile];parts=kit/"profiles"/args.profile
    dest=args.output_dir.resolve();dest.mkdir(parents=True,exist_ok=True)
    result=dest/"model.safetensors"
    if result.exists() and file_hash(result)!=info["target_sha256"]:raise SystemExit("The target already contains different weights. Choose another output directory.")
    if not result.exists():
        with tempfile.TemporaryDirectory(dir=dest) as stage:
            temp=Path(stage);source=args.source
            if source is None:
                source=temp/"source.safetensors"
                with urllib.request.urlopen(info["source_url"],timeout=120) as response,source.open("wb") as out:
                    shutil.copyfileobj(response,out,length=1024*1024)
            if file_hash(source)!=info["source_sha256"]:raise ValueError("Public source checksum mismatch.")
            if file_hash(parts/"head.npz")!=info["head_sha256"]:raise ValueError("Replacement head checksum mismatch.")
            with np.load(parts/"head.npz",allow_pickle=False) as head:
                tmp=temp/"model.safetensors"
                keys=info.get("replacement_keys",["classifier.weight","classifier.bias"])
                if set(keys)!=set(head.files):raise ValueError("Replacement tensor names differ from the profile.")
                patched_checkpoint(source,tmp,{k:head[k] for k in keys})
            if file_hash(tmp)!=info["target_sha256"]:raise ValueError("Rebuilt weights differ from the published profile hash.")
            os.replace(tmp,result)
    for n in ("config.json","tokenizer.json","tokenizer_config.json","special_tokens_map.json","vocab.txt","LICENSE","NOTICE","TOKENIZER-LICENSE.txt","predict.py","span_model.py","windowing.py","requirements.txt","README.md"):
        shutil.copyfile(kit/n,dest/n)
    for n in ("edit.json","sequence.safetensors","sequence.json"):
        if (parts/n).exists():shutil.copyfile(parts/n,dest/n)
    print("Verified",args.profile,"profile",info["target_sha256"])
if __name__=="__main__":main()
