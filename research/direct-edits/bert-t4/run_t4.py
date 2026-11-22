from pathlib import Path
import subprocess
import sys
from urllib.parse import quote
from zipfile import ZIP_DEFLATED, ZipFile

from native import read, require_t4, sha, write


def main(data, work, stage):
    note = require_t4()
    write(work/"device.json", note)
    plan = read(data/"review.json")
    assert sha(data/"protocol.json") == plan["protocol_sha256"]
    checkpoint = work/"inlegalbert.safetensors"
    src = plan["model"]
    if not checkpoint.exists() or sha(checkpoint) != src["sha256"]:
        link=f"https://huggingface.co/{src['repo']}/resolve/{src['revision']}/{quote(src['path'])}"
        temp=checkpoint.with_suffix(".part")
        subprocess.run(["curl","--fail","--retry","3","--location","--silent","--show-error",link,"-o",str(temp)],check=True)
        if sha(temp) != src["sha256"]:
            raise ValueError("Downloaded InLegalBERT failed the reference hash.")
        temp.replace(checkpoint)
    features=work/"features"
    output=work/("run-"+stage)
    fit_count,shared_count,groups = (96,16,("fit","select","shared")) if stage=="priority" else (384,48,("fit","select","shared","confirm"))
    for g in groups:
        take = ["--take",str(fit_count if g=="fit" else shared_count)] if g in ("fit","shared") else []
        subprocess.run([sys.executable,"-u",str(data/"native.py"),str(data),str(checkpoint),str(features),g,*take],check=True)
    args=[str(data),str(checkpoint),str(features),str(output)]
    subprocess.run([sys.executable,"-u",str(data/"fit.py"),"fit",*args,"--take",str(fit_count)],check=True)
    for g in groups[2:]:
        take = ["--take", str(shared_count)] if g=="shared" else []
        subprocess.run([sys.executable,"-u",str(data/"fit.py"),g,*args,*take],check=True)
    result=work/(stage+"-results.zip")
    temp=result.with_suffix(".part")
    with ZipFile(temp,"w",compression=ZIP_DEFLATED,compresslevel=3) as z:
        for file in sorted(output.iterdir()):
            z.write(file, file.name)
        z.write(work/"device.json","device.json")
        z.write(data/"protocol.json","protocol.json")
        for g in groups:
            ids=read(data/"protocol.json")["groups"][g]
            ids=ids[:fit_count] if g=="fit" else ids[:shared_count] if g=="shared" else ids
            for key in ids:
                file=features/(key+".json")
                z.write(file,"feature-receipts/"+file.name)
    temp.replace(result)
    print(stage, "complete", result, sha(result), flush=True)


if __name__=="__main__":
    main(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3])
