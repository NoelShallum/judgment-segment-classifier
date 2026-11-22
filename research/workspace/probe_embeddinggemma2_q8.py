"""Probe the text-only EmbeddingGemma 2 GGUF on a local CPU llama.cpp server."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
from time import perf_counter,sleep
from urllib.error import URLError
from urllib.request import Request,urlopen

import numpy as np

ROOT=Path(__file__).resolve().parent
MODEL=ROOT/"references/embeddinggemma2-q8"
PROMPT="task: classification | query: "
OUTPUT=ROOT/"results/embeddinggemma2-q8-cpu-probe.json"


def request(url,packet=None):
    data=None if packet is None else json.dumps(packet).encode()
    req=Request(url,data=data,headers={"Content-Type":"application/json"})
    with urlopen(req,timeout=60) as response:
        return json.load(response)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-server",type=Path,required=True)
    args=parser.parse_args()
    binary=args.llama_server.resolve(strict=True)
    details=json.loads((MODEL/"provenance.json").read_text())
    artifact=MODEL/details["file"]
    assert artifact.stat().st_size==details["bytes"]
    with artifact.open("rb") as f:
        assert hashlib.file_digest(f,"sha256").hexdigest()==details["sha256"]
    build=subprocess.run([str(binary),"--version"],check=True,capture_output=True,text=True)
    version=(build.stdout+build.stderr).strip()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1",0))
        port=probe.getsockname()[1]
    host=f"http://127.0.0.1:{port}"
    run=[str(binary),"-m",str(artifact),"--embeddings","--pooling","mean","--embd-normalize","2",
         "--host","127.0.0.1","--port",str(port),"-ngl","0","-np","1","-c","512","-b","256","-ub","256","-t","3","-tb","3"]
    fixture=[
        "The appeal is dismissed. There shall be no order as to costs.",
        "The appeal succeeds and the challenged order is set aside.",
        "Counsel for the petitioner submits that notice was not served.",
        "The sale agreement was executed on the third day of May 2019.",
        "The question is whether the notice complied with the statute.",
    ]
    with (ROOT/"embeddinggemma2-q8-llama-server.log").open("w") as log:
        begin=perf_counter()
        server=subprocess.Popen(run,stdout=log,stderr=subprocess.STDOUT)
        try:
            for _ in range(90):
                if server.poll() is not None:
                    raise RuntimeError(f"llama-server exited with code {server.returncode}")
                try:
                    if request(host+"/health").get("status")=="ok":
                        break
                except (URLError,TimeoutError):
                    sleep(1.)
            else:
                raise TimeoutError("llama-server did not report healthy")
            startup=perf_counter()-begin
            def embeddings(text):
                rows=request(host+"/v1/embeddings",{"input":[PROMPT+t for t in text]})["data"]
                v=np.asarray([r["embedding"] for r in sorted(rows,key=lambda row:row["index"])],dtype=np.float32)
                assert v.shape==(len(text),768) and np.isfinite(v).all()
                return v
            first=embeddings(fixture)
            singles=np.concatenate([embeddings([s]) for s in fixture])
            norm=np.linalg.norm(first,axis=1)
            assert np.allclose(norm,1.,atol=1.e-5)
            cosine=(first*singles).sum(1)/(norm*np.linalg.norm(singles,axis=1))
            assert cosine.min()>.999
            result={}
            for n in (1,len(fixture)):
                seconds=[]
                for _ in range(3):
                    now=perf_counter()
                    embeddings(fixture[:n])
                    seconds.append(perf_counter()-now)
                result[str(n)]=float(np.median(seconds))
        finally:
            server.terminate()
            try:
                server.wait(timeout=8)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    OUTPUT.write_text(json.dumps({"checkpoint":details,"runtime":version,"cpu_threads":3,
                "output_dims":768,"startup_seconds":startup,"min_batched_singleton_cosine":float(cosine.min()),
                "median_seconds":result,"prompt":PROMPT,"evaluation":"synthetic CPU functionality probe"},indent=2)+"\n")
    print("EmbeddingGemma 2 GGUF probe",result,flush=True)


if __name__=="__main__":
    main()
