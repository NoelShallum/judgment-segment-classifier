"""Execute the frozen packed/isolated comparison on one verified T4."""

import argparse
from pathlib import Path
import subprocess
from urllib.parse import quote
from zipfile import ZIP_DEFLATED, ZipFile

from encode import collect
from fit import evaluate, fit
from native import read, require_t4, sha, write


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source",type=Path)
    cli.add_argument("work",type=Path)
    cli.add_argument("--checkpoint",type=Path)
    cli.add_argument("--batch",type=int,default=24)
    args = cli.parse_args()
    data, work = args.source, args.work
    work.mkdir(parents=True,exist_ok=True)
    write(work/"device.json",require_t4())
    plan = read(data/"protocol.json")
    ckpt = args.checkpoint if args.checkpoint is not None else work/"source.safetensors"
    model = plan["source_model"]
    if not ckpt.exists():
        url = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{quote(model['path'])}"
        tmp = ckpt.with_suffix(".part")
        subprocess.run(["curl","--fail","--retry","3","--location","--silent","--show-error",url,"-o",str(tmp)],check=True)
        assert sha(tmp) == model["sha256"]
        tmp.replace(ckpt)
    assert sha(ckpt) == model["sha256"]
    features = work/"features"
    output = work/"run-corrected"
    collect(data,ckpt,features,("fit","select"),"cuda",args.batch,threads=2)
    if not (output/"freeze.json").exists():
        fit(data,features,output,ckpt)
    else:
        frozen = read(output/"freeze.json")
        assert frozen["serialized_weights_roundtrip"]
        assert frozen["plan_sha256"] == sha(data/"protocol.json") and frozen["base_checkpoint_sha256"] == sha(ckpt)
    collect(data,ckpt,features,("shared",),"cuda",args.batch,threads=2)
    evaluate(data,features,output)
    archive = work/"packed-results-corrected.zip"
    weak = archive.with_suffix(".part")
    with ZipFile(weak,"w",compression=ZIP_DEFLATED,compresslevel=3) as bundle:
        for path in sorted(output.iterdir()):
            bundle.write(path,path.name)
        bundle.write(work/"device.json","device.json")
        bundle.write(data/"protocol.json","protocol.json")
        for keys in plan["groups"].values():
            for key in keys:
                receipt = features/(key+".json")
                bundle.write(receipt,"feature-receipts/"+receipt.name)
    weak.replace(archive)
    print("Packed study complete",archive,sha(archive),flush=True)


if __name__ == "__main__":
    main()
